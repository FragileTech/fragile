"""Certify actual finite-kernel structural expansion using rational intervals.

All arithmetic rounds outwards to multiples of 10**-40. Square roots use
integer bounds; exp uses dyadic reduction and a proved Taylor remainder.
The finite expression averages all retained measurement vectors before
integrating frozen donor/gate rows and Gaussian jitter moments. This is an
algebraic certificate, not a simulated gas trajectory or a statistical test.
Run this script to emit its complete certificate as JSON on standard output.
"""
import itertools
import json
import math
from fractions import Fraction


# Fixed denominator for every outward-rounded interval endpoint.
S = 10**40


class I:
    """Closed interval with integer endpoints measured in units of 1/S."""

    def __init__(self, l, h=None):
        self.l = l
        self.h = l if h is None else h

    @classmethod
    def rat(cls, a, b=1):
        return cls(a * S // b, -(-a * S // b))

    def __add__(self, o):
        o = cv(o)
        return I(self.l + o.l, self.h + o.h)

    __radd__ = __add__

    def __neg__(self):
        return I(-self.h, -self.l)

    def __sub__(self, o):
        return self + -cv(o)

    def __rsub__(self, o):
        return cv(o) + -self

    def __mul__(self, o):
        o = cv(o)
        v = [self.l * o.l, self.l * o.h, self.h * o.l, self.h * o.h]
        return I(min(v) // S, -(-max(v) // S))

    __rmul__ = __mul__

    def inv(self):
        assert self.l > 0
        return I(S * S // self.h, -(-S * S // self.l))

    def __truediv__(self, o):
        return self * cv(o).inv()

    def __rtruediv__(self, o):
        return cv(o) * self.inv()

    def sq(self):
        lo = 0 if self.l <= 0 <= self.h else min(self.l * self.l, self.h * self.h)
        hi = max(self.l * self.l, self.h * self.h)
        return I(lo // S, -(-hi // S))

    def sqrt(self):
        assert self.l >= 0
        lo = math.isqrt(self.l * S)
        hi = math.isqrt(self.h * S)
        return I(lo, hi + (hi * hi != self.h * S))

    def exp(self):
        if self.h < 0:
            return (-self).exp().inv()
        if self.l < 0:
            return I(I(self.l).exp().l, I(self.h).exp().h)
        k = 0
        y = self
        while y.h > S:
            y = y / 2
            k += 1
        t = cv(1)
        r = t
        for n in range(1, 81):
            t = t * y / n
            r = r + t
        # For 0 <= y <= 1, exp(y) < 3, so the Taylor remainder is
        # at most 3*y**81/81!, including all omitted positive terms.
        tail = t * y / 81 * 3
        r = I(r.l, r.h + tail.h)
        for _ in range(k):
            r = r * r
        return r

    def clip01(self):
        return I(max(0, min(S, self.l)), max(0, min(S, self.h)))

    def endpoints(self):
        return [str(Fraction(self.l, S)), str(Fraction(self.h, S))]


def cv(a):
    return a if isinstance(a, I) else I.rat(a)


def frac(a, b):
    return I.rat(a, b)


def logistic(z):
    return frac(1, 10) + 2 / (cv(1) + (-z).exp())


def calc(x):
    """Enclose the complete marked measurement average of the cloning variance."""
    n = len(x)
    r = [-t.sq() / 2 for t in x]

    def standard(a):
        m = sum(a) / n
        var = sum(((v - m).sq() for v in a)) / n + frac(1, 100)
        return [(v - m) / var.sqrt() for v in a]

    # Endpoint absolute values are used only for these exact rational inputs.
    assert all(t.l == t.h for t in x)
    sq = [2 * t / (2 + I(abs(t.l))) for t in x]
    dd = [[(a - b).sq() for b in sq] for a in sq]
    d = [[(v + frac(1, 10 ** 6)).sqrt() for v in row] for row in dd]
    w = [[(-dd[i][j] / 8).exp() if i != j else cv(0) for j in range(n)] for i in range(n)]
    kernel = [[v / sum(row) for v in row] for row in w]
    rg = [logistic(z) for z in standard(r)]
    bar = sum(x) / n
    var = sum(((v - bar).sq() for v in x)) / n
    drift = cv(0)
    mass = cv(0)
    flux = cv(0)
    activity = cv(0)
    for marks in itertools.product(*[[j for j in range(n) if j != i] for i in range(n)]):
        prob = cv(1)
        for i, j in enumerate(marks):
            prob = prob * kernel[i][j]
        standardized = standard([d[i][j] for i, j in enumerate(marks)])
        fitness = [a * logistic(z) for a, z in zip(rg, standardized, strict=True)]
        bij = [
            [
                kernel[i][j]
                * ((fitness[j] - fitness[i]) / (fitness[i] + frac(1, 10**6))).clip01()
                for j in range(n)
            ]
            for i in range(n)
        ]
        ps = [sum(row) for row in bij]
        t = [sum((row[j] * (x[j] - x[i]) for j in range(n))) for i, row in enumerate(bij)]
        av = [
            sum(row[j] * (x[j] - x[i]).sq() for j in range(n)) - t[i].sq()
            for i, row in enumerate(bij)
        ]
        fl = sum(
            bij[i][j] * ((x[j] - bar).sq() - (x[i] - bar).sq())
            for i in range(n)
            for j in range(n)
        ) / n
        dr = (
            fl - (sum(t) / n).sq() - sum(av) / (n * n)
            + frac(n - 1, n) * frac(1, 100) * sum(ps) / n
        )
        drift = drift + prob * dr
        mass = mass + prob
        flux = flux + prob * fl
        activity = activity + prob * sum(ps) / n
    return {
        "var": var,
        "drift": drift,
        "output_var": var + drift,
        "mass": mass,
        "flux": flux,
        "activity": activity,
    }


def main():
    """Emit exact rational enclosures for the cloning and full-step proofs."""
    a = calc([frac(-3, 2)] * 3 + [cv(1)])
    b = calc([frac(-3, 200)] * 3 + [frac(1, 100)])
    inp = frac(99, 100).sq() * a["var"]
    out = (a["output_var"].sqrt() - b["output_var"].sqrt()).sq()
    answer = {
        "method": (
            "Directed fixed-denominator rational intervals, scale 10^40; "
            "square root by integer isqrt; exponential positive series through order 80 "
            "with tail <= 3*x^81/81! after dyadic range reduction."
        ),
        "a": {k: v.endpoints() for k, v in a.items()},
        "b": {k: v.endpoints() for k, v in b.items()},
        "input_structural": inp.endpoints(),
        "output_structural_lower_bound": out.endpoints(),
        "increment_lower_bound": (out - inp).endpoints(),
    }
    h = frac(1, 25)
    c = h / 2
    decay = (-h).exp()
    q2 = (1 - (-2 * h).exp()) / 2
    t = 1 - c.sq() * (1 + decay)
    nu = c.sq() * q2 + frac(1, 2500)
    apre = t.sq() * a["output_var"] + frac(3, 4) * nu
    bpre = t.sq() * b["output_var"] + frac(3, 4) * nu
    # Source-conditioned Gaussian tails control the actual terminal boundary.
    sigma2 = frac(21, 2000)
    assert (t.sq() * frac(1, 100) + nu).h < sigma2.l
    delta_a = 8 * (-frac(1, 4) / (2 * sigma2)).exp()
    delta_b = 8 * (-frac(397, 200).sq() / (2 * sigma2)).exp()
    fourth = frac(3, 2).sq().sq() + 6 * frac(3, 2).sq() * sigma2 + 3 * sigma2.sq()
    assert delta_a.h < frac(55, 10 ** 6).l
    assert delta_b.h < frac(1, 10 ** 30).l
    assert fourth.h < frac(1041, 200).l
    # Retain the normalization by the probability of nonextinction.
    alive_a = apre - (frac(1041, 200) * frac(55, 10 ** 6)).sqrt()
    alive_b = (bpre + 4 * frac(1, 10 ** 30)) / (1 - frac(1, 10 ** 30))
    # Q(x,v) = (399/400)*x*x + (v+.05*x)**2.
    full_lower = frac(399, 400) * (alive_a.sqrt() - alive_b.sqrt()).sq()
    assert (full_lower - inp).l > frac(8, 100).h
    answer["full_step"] = {
        "scope": (
            "N=4 canonical quadratic BAOAB, h=.04, B=1, gamma=1, position diffusion=.1, "
            "cap=2, terminal absorption; Q=|x|^2+|v|^2+.1*x*v; conditional on nonextinction."
        ),
        "preboundary_a_variance": apre.endpoints(),
        "preboundary_b_variance": bpre.endpoints(),
        "a_exit_union_probability_bound": delta_a.endpoints(),
        "b_exit_union_probability_bound": delta_b.endpoints(),
        "a_alive_variance_lower_bound": alive_a.endpoints(),
        "b_alive_variance_upper_bound": alive_b.endpoints(),
        "structural_output_lower_bound": full_lower.endpoints(),
        "structural_increment_lower_bound": (full_lower - inp).endpoints(),
    }
    assert (out - inp).l > frac(12, 100).h
    print(json.dumps(answer, indent=2))


if __name__ == "__main__":
    main()
