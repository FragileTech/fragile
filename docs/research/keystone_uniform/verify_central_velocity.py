"""Reproduce KUER.24--26, retaining the full Gaussian tail."""

import math

import mpmath as mp


iv = mp.iv
iv.dps = 85
one = iv.mpf(1)
pi = iv.pi
normalizer = iv.sqrt(2 * pi)
coefficients = [iv.mpf((-1) ** n) / ((2 * n + 1) * 2**n * math.factorial(n)) for n in range(161)]


def phi_cdf(x):
    # The exact endpoints are +/-5.1; allow their outward rounding.
    assert x.a >= -iv.mpf("5.101") and x.b <= iv.mpf("5.101")
    polynomial = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        polynomial = coefficient + x * x * polynomial
    return one / 2 + x * polynomial / normalizer + iv.mpf(["-1e-100", "1e-100"])


t = one / 50
c = iv.exp(-one / 25)
q2 = (one - iv.exp(-2 * one / 25)) / 2
b = t * (one + c)
eta = t * b
epsilon = one / 100
sigma = one / 10
radius = 4 * b
linear = c - 2 * eta
periodic = 40 * pi * pi * eta
sinc_min = iv.sin(2 * pi * radius) / (2 * pi * radius)
attenuation = iv.exp(-2 * pi * pi * t * t * q2)
second = iv.exp(-8 * pi * pi * t * t * q2)
assert (linear * attenuation).a > (periodic * second).b


def envelope(r):
    g = (one - 2 * eta) * r - 20 * pi * eta * iv.sin(2 * pi * r)
    angle = g + radius
    if angle.b <= one / 2:
        cosine = iv.cos(2 * pi * angle)
    else:
        assert angle.a >= one / 2
        cosine = -one
    values = [
        linear**2
        - 2 * linear * periodic * s * attenuation * cosine
        + periodic**2 * s * s / 2 * (one - second + 2 * second * cosine**2)
        for s in (sinc_min, one)
    ]
    return iv.mpf([max(x.a for x in values), max(x.b for x in values)])


def mass(r):
    return phi_cdf((r - epsilon) / sigma) - phi_cdf((-r - epsilon) / sigma)


persisting = envelope(epsilon)
jitter = sum(
    envelope(iv.mpf(j) / 100) * (mass(iv.mpf(j) / 100) - mass(iv.mpf(j - 1) / 100))
    for j in range(1, 51)
) + (linear + periodic) ** 2 * (one - mass(one / 2))
assert jitter.a > persisting.b
x2 = iv.sqrt(3 * (epsilon**2 + sigma**2))
u2 = 4 + t * (2 * x2 + 20 * pi * iv.sqrt(3))
w2 = iv.sqrt(c * c * u2 * u2 + 3 * q2)
a = iv.mpf(3) / 500
defect = 4 * a * iv.exp(-one / 2) * b * (one + a) * w2
coefficient = iv.sqrt(jitter) + a * (linear + periodic) + a * c + defect
for name, value in [
    ("persistent square", persisting),
    ("jitter square", jitter),
    ("OU L2", w2),
    ("graph defect", defect),
    ("velocity coefficient", coefficient),
]:
    print(name, value)
assert jitter.b < iv.mpf(".838")
assert coefficient.b < iv.mpf(".934")
print("All central velocity certificate margins passed")
