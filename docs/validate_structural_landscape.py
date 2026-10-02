"""Check identities and exact numerical certificates in structural convergence.

Run with: uv run python docs/validate_structural_landscape.py
SymPy checks polynomial identities. Fraction checks prove the conservative
Rastrigin bounds using exact rational arithmetic. mpmath values at the end are
illustrations of the proved formulas, not substitutes for analytic bounds.
"""

from fractions import Fraction
from math import factorial

import mpmath as mp
import sympy as sp


def verify_identities():
    x, v, c, a, q, s, xi, zeta, force = sp.symbols("x v c a q s xi zeta force")
    v1 = v + c * force
    x1 = x + c * v1
    v2 = a * v1 + q * xi
    actual = x1 + c * v2 + s * zeta
    expected = x + c * (1 + a) * v + c**2 * (1 + a) * force + c * q * xi + s * zeta
    assert sp.expand(actual - expected) == 0

    # The signed complete-step kinetic polynomial (SCK.2) must equal the
    # actual quadratic difference before the radial cap, term by term.
    r, z, f, g, B, kin_eta, beta, gamma_p, alpha_p = sp.symbols(
        "r z f g B kin_eta beta gamma_p alpha_p"
    )
    R = r + B * z + kin_eta * f
    Z = a * z + a * c * f + c * g
    metric = lambda pos, vel: alpha_p * pos**2 + 2 * beta * pos * vel + gamma_p * vel**2
    signed = (
        2 * (alpha_p * B + beta * (a - 1)) * r * z
        + (alpha_p * B**2 + 2 * beta * a * B + gamma_p * (a**2 - 1)) * z**2
        + 2 * (alpha_p * kin_eta + beta * a * c) * r * f
        + 2 * beta * c * r * g
        + 2 * (alpha_p * B * kin_eta + beta * a * c * B
               + beta * a * kin_eta + gamma_p * a**2 * c) * z * f
        + 2 * (beta * B * c + gamma_p * a * c) * z * g
        + (alpha_p * kin_eta**2 + 2 * beta * a * c * kin_eta
           + gamma_p * a**2 * c**2) * f**2
        + 2 * (beta * c * kin_eta + gamma_p * a * c**2) * f * g
        + gamma_p * c**2 * g**2
    )
    assert sp.expand(metric(R, Z) - metric(r, z) - signed) == 0

    # For a matched component the shared Haar sign retains the relative
    # velocity cross term. Independent signs for unmatched components erase it.
    mu, nu, b, b_t, restitution = sp.symbols("mu nu b b_t restitution")
    matched = sum(
        (mu - nu + restitution * sign * (b - b_t))**2 for sign in (-1, 1)
    ) / 2
    unmatched = sum(
        (mu - nu + restitution * (sign * b - sign_t * b_t))**2
        for sign in (-1, 1) for sign_t in (-1, 1)
    ) / 4
    assert sp.expand(matched - (mu - nu)**2 - restitution**2 * (b - b_t)**2) == 0
    assert sp.expand(unmatched - (mu - nu)**2 - restitution**2 * (b**2 + b_t**2)) == 0

    # A row of the unsimplified Keystone plan: outgoing recipient error
    # cannot be retained without the corresponding incoming donor error.
    d0, d1, d2, cross1, cross2, common1, common2, residual1, residual2 = sp.symbols(
        "d0 d1 d2 cross1 cross2 common1 common2 residual1 residual2"
    )
    persist = 1 - common1 - common2 - residual1 - residual2
    post = (persist * d0**2 + common1 * d1**2 + common2 * d2**2
            + residual1 * cross1**2 + residual2 * cross2**2)
    net = (common1 * (d1**2 - d0**2) + common2 * (d2**2 - d0**2)
           + residual1 * (cross1**2 - d0**2)
           + residual2 * (cross2**2 - d0**2))
    assert sp.expand(post - d0**2 - net) == 0

    kappa, remainder = sp.symbols("kappa remainder")
    resonant = expected.subs(force, -kappa * x + remainder)
    resonant = sp.expand(resonant).subs(kappa, 1 / (c**2 * (1 + a)))
    target = c * (1 + a) * v + c**2 * (1 + a) * remainder + c * q * xi + s * zeta
    assert sp.simplify(resonant - target) == 0

    d = sp.symbols("d", integer=True, positive=True)
    sixth = 15 * d + 9 * d * (d - 1) + d * (d - 1) * (d - 2)
    assert sp.expand(sixth - d * (d + 2) * (d + 4)) == 0

    m, upper = sp.symbols("m upper", positive=True)
    eta = 2 / (m + upper)
    assert sp.simplify((1 - eta * m) - (upper - m) / (upper + m)) == 0
    assert sp.simplify((eta * upper - 1) - (upper - m) / (upper + m)) == 0

    n, wd, wc, gate = sp.symbols("n wd wc gate", positive=True)
    gain = (n - 1) * wc / (n - 2 + wc) * (n - 2) / (n - 2 + wd) * gate
    assert sp.limit(gain, n, sp.oo) == wc * gate
    print("PASS: BAOAB, signed kinetic and Keystone net-flux identities, Haar moments,")
    print("      resonance cancellation, Gaussian sixth moment,")
    print("      local-step optimization and singleton amplification limit.")


def verify_rastrigin_certificate():
    # pi > 3 and sqrt(2) > 7/5 imply m + M > 616.
    total_lower = 4 + 20 * (2 + Fraction(7, 5)) * 9
    assert total_lower == 616
    assert Fraction(16, 3) / total_lower < Fraction(1, 100)
    assert 3 - 2 * Fraction(7, 5) == Fraction(1, 5)

    radius, jitter = Fraction(1, 16), Fraction(1, 64)
    vc, B_upper = Fraction(2, 1000), Fraction(3, 40)
    gap_lower = radius - Fraction(1, 5) * (radius + jitter) - B_upper * vc
    assert gap_lower > Fraction(9, 200)
    assert Fraction(1, 100) + radius + jitter < Fraction(1, 8)
    jitter_exponent = jitter**2 / (2 * Fraction(1, 1000)**2)
    variance_upper = Fraction(401, 400 * 10**6)
    kinetic_exponent = Fraction(9, 200)**2 / (2 * variance_upper)
    assert jitter_exponent > 122
    assert kinetic_exponent > 1000

    # exp(122) exceeds any finite Taylor partial sum, whose terms are positive.
    exp_lower = sum((Fraction(122**k, factorial(k)) for k in range(201)), Fraction(0))
    residence_upper = Fraction(4 * 10**6 * 128) / exp_lower
    assert residence_upper < Fraction(6, 10**45)
    print("PASS: exact rational residence certificate < 6e-45 for N=128,")
    print("      d=1, 1,000,000 updates; no floating-point tails used.")


def illustrative_values():
    mp.mp.dps = 80
    m = 2 + 20 * mp.sqrt(2) * mp.pi**2
    upper = 2 + 40 * mp.pi**2
    eta = 2 / (m + upper)
    h = 2 * mp.sqrt(eta / mp.mpf("1.5"))
    print("Illustrative values (not interval certificates):")
    for name, value in (("h", h), ("rho", (upper - m) / (upper + m)),
                        ("residence upper formula", 4 * 10**6 * 128 * mp.exp(-122))):
        print(f"  {name}: {mp.nstr(value, 16)}")

    # Exact entering positions 0 and 1, common velocity 0, feature radius 2,
    # bandwidths 2, diversity floor .001 and standardizer .1.
    n = 128
    distance = mp.mpf(2) / 3
    weight = mp.exp(-distance**2 / 8)
    delta = mp.sqrt(distance**2 + mp.mpf(".001")**2) - mp.mpf(".001")
    pd = pc = weight / (n - 2 + weight)
    terms = []
    for k in range(n):
        t = mp.mpf(k + 1) / n
        scale = mp.sqrt(t * (1 - t) * delta**2 + mp.mpf(".1")**2)
        fminus = 2 / (1 + mp.exp(t * delta / scale)) + mp.mpf(".1")
        fplus = 2 / (1 + mp.exp(-(1 - t) * delta / scale)) + mp.mpf(".1")
        gate = min(mp.mpf(1), (fplus - fminus) / (fminus + mp.mpf(".000001")))
        terms.append(mp.binomial(n - 1, k) * pd**k * (1 - pd)**(n - 1 - k)
                     * (n - 1 - k) * gate)
    print("  singleton expected post-copy count:", mp.nstr(1 + pc * mp.fsum(terms), 16))



# Outward-rounded dyadic interval arithmetic. Each endpoint operation below is
# rational/integer arithmetic; transcendental enclosures use positive Taylor
# terms with a proved geometric remainder, and integer square-root bracketing.
class Interval:
    scale = 2**128

    def __init__(self, lo, hi=None):
        lo, hi = Fraction(lo), Fraction(lo if hi is None else hi)
        self.lo = Fraction((lo * self.scale).__floor__(), self.scale)
        self.hi = Fraction((hi * self.scale).__ceil__(), self.scale)
        assert self.lo <= self.hi

    @staticmethod
    def as_interval(value):
        return value if isinstance(value, Interval) else Interval(value)

    def __add__(self, other):
        other = self.as_interval(other)
        return Interval(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -self.as_interval(other)

    def __rsub__(self, other):
        return self.as_interval(other) + -self

    def __mul__(self, other):
        other = self.as_interval(other)
        values = [x * y for x in (self.lo, self.hi) for y in (other.lo, other.hi)]
        return Interval(min(values), max(values))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.as_interval(other)
        assert other.lo > 0 or other.hi < 0
        return self * Interval(1 / other.hi, 1 / other.lo)

    def __rtruediv__(self, other):
        return self.as_interval(other) / self

    def __pow__(self, power):
        assert isinstance(power, int) and power >= 0
        out = Interval(1)
        for _ in range(power):
            out = out * self
        return out

    def sqrt(self):
        from math import isqrt

        assert self.lo >= 0
        lower = isqrt((self.lo * self.scale**2).__floor__())
        upper = isqrt((self.hi * self.scale**2).__floor__()) + 1
        return Interval(Fraction(lower, self.scale), Fraction(upper, self.scale))

    @staticmethod
    def exp_point(x):
        x = Fraction(x)
        if x < 0:
            return 1 / Interval.exp_point(-x)
        assert x <= 8
        term = total = Interval(1)
        for k in range(1, 101):
            term = term * x / k
            total = total + term
        next_term = term * x / 101
        # Subsequent term ratios are at most x/102 < 1.
        remainder = next_term.hi / (1 - x / 102)
        return Interval(total.lo, total.hi + remainder)

    def exp(self):
        return Interval(self.exp_point(self.lo).lo, self.exp_point(self.hi).hi)


def verify_singleton_interval(reward_a=0, reward_b=0, reward_exponent=0):
    n = 128
    delta = (Interval(Fraction(4, 9)) + Fraction(1, 10**6)).sqrt() - Fraction(1, 1000)
    weight = Interval(-Fraction(1, 18)).exp()
    p = weight / (n - 2 + weight)
    probability = (1 - p)**(n - 1)
    mean_reward = Fraction((n - 1) * reward_a + reward_b, n)
    reward_variance = Fraction(n - 1, n**2) * (reward_b - reward_a)**2
    reward_scale = Interval(reward_variance + Fraction(1, 100)).sqrt()
    za = (Interval(reward_a) - mean_reward) / reward_scale
    zb = (Interval(reward_b) - mean_reward) / reward_scale
    ha = (2 / (1 + (-za).exp()) + Fraction(1, 10))**reward_exponent
    hb = (2 / (1 + (-zb).exp()) + Fraction(1, 10))**reward_exponent

    def acceptance(fitness, donor_fitness):
        ratio = (donor_fitness - fitness) / (fitness + Fraction(1, 10**6))
        return Interval(max(0, min(1, ratio.lo)), max(0, min(1, ratio.hi)))

    gain = Interval(0)
    for k in range(n):
        t = Interval(Fraction(k + 1, n))
        scale = (t * (1 - t) * delta**2 + Fraction(1, 100)).sqrt()
        fminus = 2 / (1 + (t * delta / scale).exp()) + Fraction(1, 10)
        fplus = 2 / (1 + (-(1 - t) * delta / scale).exp()) + Fraction(1, 10)
        discovered = hb * fplus
        near, far = ha * fminus, ha * fplus
        signed_gain = ((n - 1 - k) * (p * acceptance(near, discovered)
                        - acceptance(discovered, near) / (n - 1))
                      + k * (p * acceptance(far, discovered)
                        - acceptance(discovered, far) / (n - 1)))
        gain = gain + probability * signed_gain
        if k < n - 1:
            probability = probability * Fraction(n - 1 - k, k + 1) * p / (1 - p)
    result = 1 + gain
    if reward_exponent == 0:
        assert result.lo > Fraction(19060, 10000)
        assert result.hi < Fraction(19061, 10000)
        print("PASS: rational interval count 1.9060 < E[K] < 1.9061 (diversity only).")
    elif reward_b > reward_a:
        assert result.lo > Fraction(19460, 10000)
        assert result.hi < Fraction(19462, 10000)
        print("PASS: rational interval count 1.9460 < E[K] < 1.9462 (better site).")
    else:
        assert result.hi < Fraction(1, 10000)
        print("PASS: rational interval upper bound E[K] < 1e-4 (worse site).")


def verify_local_attraction():
    radius, vmax = Fraction(1, 16), Fraction(1, 1000)
    exponent = (4 * radius**2 + 4 * vmax**2) / 8
    assert exponent < Fraction(1, 2)
    rate = 2 * Fraction(1, 5)**2 * 3
    assert rate == Fraction(6, 25)
    variance = Fraction(401, 400 * 10**6)
    baseline = (
        2 * Fraction(1, 5)**2 / 10**6
        + 2 * Fraction(3, 40)**2 * Fraction(2, 1000)**2
        + variance
    )
    assert baseline == Fraction(451, 400 * 10**6)
    assert Fraction(61**8, factorial(8)) > 8 * 10**8
    additive = baseline + Fraction(1, 400 * 10**6)
    assert additive == Fraction(113, 10**8)
    moment8 = rate**8 / 256 + additive / (1 - rate)
    assert moment8 < Fraction(154, 10**8)
    assert moment8 / (1 - Fraction(1, 10**40)) < Fraction(16, 10**7)
    print("PASS: rational local-attraction rate < .24, additive term < 1.13e-6,")
    print("      and conditional mean-square radius after eight steps < 1.6e-6.")


def verify_trajectory_and_regeneration():
    """Symbolic identities and exact exponent bookkeeping for Sections 9--10."""
    d, c, kappa, a = sp.symbols("d c kappa a", positive=True)
    eighth = 16 * sp.prod(d / 2 + j for j in range(4))
    assert sp.expand(eighth - d * (d + 2) * (d + 4) * (d + 6)) == 0
    assert sp.simplify((1 - c**2 * kappa).subs(c**2, 1 / (kappa * (1 + a))) - a / (1 + a)) == 0
    # R=delta^(-1/32), r=delta^(1/2), bad mass O(delta^(1/4)).
    one = Fraction(1)
    q1_exponents = [Fraction(1, 2) - Fraction(1, 32), Fraction(1, 4), Fraction(3, 16)]
    q2_exponents = [Fraction(1, 2) - Fraction(3, 32), Fraction(1, 4), Fraction(1, 8)]
    assert min(q1_exponents) == Fraction(3, 16)
    assert min(q2_exponents) == Fraction(1, 8)
    assert Fraction(1, 8) - Fraction(2, 32) == Fraction(1, 16)
    assert Fraction(1, 16) - Fraction(1, 32) == Fraction(1, 32)
    assert Fraction(3, 16) - Fraction(1, 2) < 0  # finite-cell sampling vanishes
    # Harris choices R=4b, beta=epsilon/(R+2b), and the global K^2 minorization.
    b, eps = sp.symbols("b eps", positive=True)
    radius, beta = 4 * b, eps / (6 * b)
    outside = (2 + 2 * beta * b) / (2 + beta * radius)
    assert sp.simplify(1 - outside - eps / (2 * eps + 6)) == 0
    assert one - Fraction(1, 4) == Fraction(3, 4)
    print("PASS: eighth Gaussian moment, resonance minorization, truncation exponents,")
    print("      grid sampling exponents and explicit Harris/regeneration algebra.")


def verify_structural_flux():
    """Exact finite-source balance and parameter-only structural identities."""
    values = [Fraction(1), Fraction(4), Fraction(9)]
    donors = [[Fraction(0), Fraction(1, 3), Fraction(2, 3)],
              [Fraction(1, 4), Fraction(0), Fraction(3, 4)],
              [Fraction(2, 5), Fraction(3, 5), Fraction(0)]]
    gates = [[Fraction(0), Fraction(1, 2), Fraction(1, 5)],
             [Fraction(2, 3), Fraction(0), Fraction(0)],
             [Fraction(1), Fraction(1, 3), Fraction(0)]]
    copied = sum((1 - sum(donors[i][j] * gates[i][j] for j in range(3))) * values[i]
                 + sum(donors[i][j] * gates[i][j] * values[j] for j in range(3))
                 for i in range(3)) / 3
    flux = sum(donors[i][j] * gates[i][j] * (values[j] - values[i])
               for i in range(3) for j in range(3)) / 3
    assert copied == sum(values) / 3 + flux
    r0 = sp.symbols("r0", positive=True)
    tilt = (1 - r0) / (2 * r0)
    assert sp.simplify((1 + tilt) * r0 - (1 + r0) / 2) == 0
    d = sp.symbols("d", positive=True)
    radius = sp.symbols("radius", positive=True)
    assert sp.simplify(d * radius**(-d) * radius**(d + 8) / (d + 8)
                       - d * radius**8 / (d + 8)) == 0
    # Tail-to-core companion self-exclusion may only improve the lower bound.
    for n in range(2, 30):
        assert Fraction(n, n - 1) >= 1
    print("PASS: exact copying-flux balance, selection-versus-growth rate,")
    print("      uniform-ball initial moments and donor self-exclusion factors.")


def verify_regional_holder_closure():
    """Exact exponent checks supplement the all-parameter algebraic proof."""
    for alpha in [Fraction(1, 8), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1)]:
        threshold = 16 * alpha**2 / (1 + alpha)
        for fraction in [Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)]:
            power = threshold * fraction
            e1 = alpha / 2 - power / 32
            e2 = alpha * e1 - power / 32
            assert 0 < e2 <= e1 <= Fraction(1, 2)
    alpha, power = sp.symbols("alpha power")
    assert sp.expand(alpha * (alpha / 2 - power / 32) - power / 32
                     - (alpha**2 / 2 - power * (1 + alpha) / 32)) == 0
    for dim in range(1, 21):
        assert Fraction(3, 16) - Fraction(1, 2) <= -Fraction(1, 16 * dim)
    print("PASS: symbolic regional Holder threshold and exact exponent/grid checks.")


if __name__ == "__main__":
    verify_identities()
    verify_trajectory_and_regeneration()
    verify_structural_flux()
    verify_regional_holder_closure()
    verify_rastrigin_certificate()
    verify_local_attraction()
    verify_singleton_interval()
    verify_singleton_interval(-1, 0, 1)
    verify_singleton_interval(0, -1, 1)
    illustrative_values()
