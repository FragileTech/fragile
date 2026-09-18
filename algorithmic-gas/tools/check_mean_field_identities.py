"""Independent finite-kernel algebra checks for the repaired entropy proofs.

These small kernels test universal measure identities, not Euclidean Gas trajectories.
The Rust kinetic and collision tests separately check the implemented gas stages.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def expectation(p, f):
    return sum(x * y for x, y in zip(p, f, strict=True))


def normalize(p):
    return [x / sum(p) for x in p]


def output(p, kernel):
    return [sum(p[i] * kernel[i][j] for i in range(2)) for j in range(2)]


def entropy(p, q):
    return sum(x * math.log(x / y) for x, y in zip(p, q, strict=True) if x)


def variance(p, f):
    mean = expectation(p, f)
    return expectation(p, [(x - mean) ** 2 for x in f])


def check():
    maximum_identity_error = 0.0
    maximum_cubic_ratio = 0.0
    cases = 0
    for kernel in [[[0.6, 0.3], [0.2, 0.4]], [[0.15, 0.65], [0.55, 0.2]]]:
        k = [sum(row) for row in kernel]
        a, b = kernel[0]
        c, d = kernel[1]
        eigenvalue = (a + d + math.sqrt((a - d) ** 2 + 4 * b * c)) / 2
        nu = normalize([c, eigenvalue - a])
        e = [b, eigenvalue - a]
        e = [x / expectation(nu, e) for x in e]
        doob = [[kernel[i][j] * e[j] / (eigenvalue * e[i]) for j in range(2)] for i in range(2)]
        delta = sum(min(doob[i][j] for i in range(2)) for j in range(2))
        for p in [[0.1, 0.9], [0.35, 0.65], [0.8, 0.2]]:
            reference = [0.45, 0.55]
            ap, ar = expectation(p, k), expectation(reference, k)
            pp, rp = normalize(output(p, kernel)), normalize(output(reference, kernel))
            ps = [p[i] * k[i] / ap for i in range(2)]
            rs = [reference[i] * k[i] / ar for i in range(2)]
            backward_loss = 0.0
            for j in range(2):
                pb = normalize([p[i] * kernel[i][j] for i in range(2)])
                rb = normalize([reference[i] * kernel[i][j] for i in range(2)])
                backward_loss += pp[j] * entropy(pb, rb)
            error = abs(entropy(ps, rs) - entropy(pp, rp) - backward_loss)
            maximum_identity_error = max(maximum_identity_error, error)
            logf = [math.log(p[i] / reference[i]) for i in range(2)]
            covariance = expectation(p, [k[i] * logf[i] for i in range(2)]) - ap * expectation(p, logf)
            source = expectation(pp, [math.log(rp[i] / reference[i]) for i in range(2)])
            rhs = covariance / ap + math.log(ar / ap) - backward_loss + source
            maximum_identity_error = max(maximum_identity_error, abs(entropy(pp, reference) - entropy(p, reference) - rhs))
            evolved = p
            for n in range(21):
                bound = (max(e) / min(e)) ** 2 * (1 - delta) ** n * entropy(p, nu)
                assert entropy(evolved, nu) <= bound + 1e-14
                evolved = normalize(output(evolved, kernel))
            cases += 1
        observable = [-1.0, 2.0]
        conditional_mean = [expectation(kernel[i], observable) / k[i] for i in range(2)]
        conditional_variance = [
            expectation(kernel[i], [x * x for x in observable]) / k[i] - conditional_mean[i] ** 2
            for i in range(2)
        ]
        tilted = [nu[i] * k[i] / eigenvalue for i in range(2)]
        budget = expectation(tilted, conditional_variance) + variance(tilted, conditional_mean)
        maximum_identity_error = max(maximum_identity_error, abs(variance(nu, observable) - budget))
        g = [1.0, -nu[0] / nu[1]]
        size = max(map(abs, g))
        g = [x / size for x in g]
        ag = [sum(nu[i] * kernel[i][j] * g[i] for i in range(2)) / (eigenvalue * nu[j]) for j in range(2)]
        quadratic = (variance(nu, ag) - expectation(nu, [x * x for x in g])) / 2
        for epsilon in [-0.1, -0.01, 0.01, 0.1]:
            p = [nu[i] * (1 + epsilon * g[i]) for i in range(2)]
            pp = normalize(output(p, kernel))
            residual = abs(entropy(pp, nu) - entropy(p, nu) - epsilon * epsilon * quadratic)
            maximum_cubic_ratio = max(maximum_cubic_ratio, residual / abs(epsilon) ** 3)
            assert residual <= 12 * abs(epsilon) ** 3
            cases += 1
    assert maximum_identity_error < 2e-14
    return {
        "kind": "independent_finite_kernel_algebra",
        "cases": cases,
        "maximum_identity_error": maximum_identity_error,
        "maximum_cubic_remainder_ratio": maximum_cubic_ratio,
        "allowed_cubic_remainder_ratio": 12,
        "identities": ["survivor_chain_rule", "product_reference_recurrence", "qsd_variance_budget", "doob_entropy_decay", "qsd_forcing_cancellation"],
    }


if __name__ == "__main__":
    report = check()
    path = Path(__file__).resolve().parents[1] / "validation" / "mean-field-proof-identities.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
