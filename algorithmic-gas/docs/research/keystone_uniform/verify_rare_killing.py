"""Reproduce KURC.17--21 with outward interval arithmetic (no simulation)."""

import math

import mpmath as mp


iv = mp.iv
iv.dps = 85
zero = iv.mpf(0)
one = iv.mpf(1)
two = iv.mpf(2)
pi = iv.pi
root2pi = iv.sqrt(2 * pi)
phi8 = iv.exp(-iv.mpf(32)) / root2pi
tail8 = phi8 / 8
K = 400
coef = [iv.mpf((-1) ** n) / (iv.mpf(2 * n + 1) * 2**n * math.factorial(n)) for n in range(K + 1)]
err = iv.mpf("1e-200")


def Phi(x):
    if x.a >= 8:
        return one - iv.mpf(["0", "6.4e-16"])
    if x.b <= -8:
        return iv.mpf(["0", "6.4e-16"])
    assert x.a >= -8 and x.b <= 8
    y = x * x
    p = coef[-1]
    for co in reversed(coef[:-1]):
        p = co + y * p
    return iv.mpf("0.5") + x * p / root2pi + iv.mpf(["-1e-200", "1e-200"])


h = one / 25
t = h / 2
c = iv.exp(-h)
b = t * (one + c)
eta = b * t
q2 = (one - iv.exp(-2 * h)) / 2
tau = iv.sqrt(t * t * q2 + one / 2500)
l = one - 2 * eta
A = 20 * pi * eta
Linner = 2 - 4 * b
s = zero
for j in range(400):
    zl = -8 + iv.mpf(j) / 25
    zr = -8 + iv.mpf(j + 1) / 25
    mass = Phi(zr) - Phi(zl)
    x = 2 + zr / 10
    gx = l * x - A * iv.sin(2 * pi * x)
    p = Phi((Linner - gx) / tau) - Phi((-Linner - gx) / tau)
    s += mass * p
p1 = s * s * s
print("single interval", s)
print("accepted p1 interval", p1)
k = b / tau
r0 = iv.mpf(103) / 50
z = k * r0 / iv.sqrt(3)
Q = Phi(-z)
phi = iv.exp(-z * z / 2) / root2pi
psi = Q**3
B = -3 * (k / iv.sqrt(3)) * Q**2 * phi / (2 * r0)
At = psi - B * r0 * r0
print("support A interval", At)
print("support at E2 interval", At + 4 * B)
print("count checks p1>A and A+4B>2e-6", p1.a > At.b, (At + 4 * B).a > iv.mpf("2e-6"))
# Truncated primitive Gaussian-column sum and explicit integral-tail upper bound.
C = 2 * iv.exp(2) + 9**3 * (
    1
    + sum(
        2 * iv.exp(-iv.mpf(35) / 128 * (iv.mpf(5) / 4) ** (2 * j))
        + iv.exp(-iv.mpf(3) / 8 * (iv.mpf(5) / 4) ** j)
        for j in range(40)
    )
)
r = iv.mpf(5) / 4
a = iv.mpf(35) / 128
bb = iv.mpf(3) / 8
C += 9**3 * (
    2 * iv.exp(-a * r**80) * (1 + 1 / (2 * a * iv.ln(r) * r**80))
    + iv.exp(-bb * r**40) * (1 + 1 / (bb * iv.ln(r) * r**40))
)
print("C40 interval", C, "C40 <7188", C.b < 7188)
VE = 2 * (1 - t * iv.mpf(3) / 10 + t * iv.mpf(3) / 10 * iv.sqrt(7188))
Qr = Phi(-k * VE / iv.sqrt(3))
print("row psi lower interval", Qr**3, "row >7e-11", (Qr**3).a > iv.mpf("7e-11"))
box_factor = (one - iv.exp(-8 / tau**2)) ** 3
assert p1.a > At.b
assert ((At + 4 * B) * box_factor).a > iv.mpf("2e-6")
assert C.b < 7188
assert (Qr**3 * box_factor).a > iv.mpf("7e-11")
zr = k * VE / iv.sqrt(3)
derivative = Qr**3 + 3 * zr * Qr**2 * iv.exp(-(zr**2) / 2) / (2 * root2pi)
assert derivative.b < p1.a
print("All survival certificate margins passed")
