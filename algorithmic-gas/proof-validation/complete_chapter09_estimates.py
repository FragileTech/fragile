"""Exact finite-law, normalized measurement and survival algebra for Chapter 9.

Every source expression receives its own disposition. A proof obligation or
operator definition never inherits numerical credit from its owning theorem.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

from chapter09_residual_estimates import residual_checks
import numpy as np
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/source/2_fractal_gas/convergence_program/09_propagation_chaos.md"
INVENTORY = Path(__file__).with_name("chapter09_inventory.json")
SCOPE = (
    "Exact finite probabilities and explicitly supplied analytic fixtures. These compare the "
    "whole recorded operands in the indicated expression. A finite fixture does not establish "
    "a native QSD, uniform attraction, a Poincare constant or an infinite-horizon limit."
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def algebra(inventory):
    expressions = inventory["quantitative_expressions"]
    checks = []
    source_lines = SOURCE.read_text().splitlines()
    formal = inventory["formal_items"]

    def add(label, needle, name, observed, bound=0.0, relation="equal", inputs=None):
        item = next(i for i in formal if i["id"] == label)
        start = item["source_line"]
        end = min(
            [i["source_line"] for i in formal if i["source_line"] > start] or [len(source_lines)]
        )
        matches = [
            e for e in expressions if start <= e["source_line"] < end and needle in e["formula"]
        ]
        if not matches and needle not in "\n".join(source_lines[start - 1 : end]):
            raise ValueError(f"Exact source expression missing: {label}: {needle}")
        observed, bound = float(observed), float(bound)
        tolerance = 2e-11 * max(1, abs(bound), abs(observed))
        passed = (
            abs(observed - bound) <= tolerance
            if relation == "equal"
            else observed <= bound + tolerance
        )
        checks.append({
            "id": name,
            "source_label": label,
            "source_quote": needle,
            "source_line_only_evidence": [
                {"line": index + 1, "exact_line": line}
                for index, line in enumerate(source_lines)
                if start <= index + 1 < end and needle in line
            ]
            if not matches
            else [],
            "source_expressions": [
                {"id": e["id"], "formula": e["formula"], "line": e["source_line"]} for e in matches
            ],
            "observed": observed,
            "bound": bound,
            "relation": relation,
            "tolerance": tolerance,
            "passed": passed,
            "inputs": inputs or {},
            "scope": SCOPE,
        })

    # Sampling indices address a permutation-invariant swarm; every possible
    # address tuple is integrated. No persistent physical labels are introduced.
    for n in [2, 3, 4, 6]:
        values = np.array([math.sin(i * 1.7) for i in range(n)])
        mean = float(values.mean())
        for i in range(n):
            exclusion = abs(float(np.delete(values, i).mean()) - mean)
            add(
                "lem-empirical-convergence",
                "\\frac2N",
                f"exclude-N{n}-{i}",
                exclusion,
                2 / n,
                "upper",
                {"values": values.tolist(), "removed_address": i, "N": n},
            )
        for order in range(2, min(n, 4) + 1):
            with_replacement = []
            distinct = []
            collision = 0
            for indices in itertools.product(range(n), repeat=order):
                product = math.prod(values[i] for i in indices)
                with_replacement.append(product)
                if len(set(indices)) == order:
                    distinct.append(product)
                else:
                    collision += 1
            exact = collision / n**order
            add(
                "lem-empirical-convergence",
                "l(l-1)/(2N)",
                f"collision-N{n}-l{order}",
                exact,
                order * (order - 1) / (2 * n),
                "upper",
                {"N": n, "l": order, "all_tuples": n**order},
            )
            add(
                "lem-empirical-convergence",
                "l(l-1)/N",
                f"product-N{n}-l{order}",
                abs(np.mean(distinct) - np.mean(with_replacement)),
                order * (order - 1) / n,
                "upper",
                {
                    "N": n,
                    "l": order,
                    "without_replacement": distinct,
                    "with_replacement": with_replacement,
                },
            )
        square = sum(values[i] * values[j] for i in range(n) for j in range(n)) / n**2
        pair = sum(values[i] * values[j] for i in range(n) for j in range(n) if i != j) / (
            n * (n - 1)
        )
        add(
            "lem-empirical-convergence",
            "\\mathbb E(L_N\\varphi)^2",
            f"exchangeable-square-N{n}",
            square,
            float(np.mean(values**2)) / n + (n - 1) / n * pair,
            inputs={"uniform_permutation_law": values.tolist()},
        )

    # Normalized weighted kernels, joint laws, reward moments, alive ratio.
    for fixture in range(24):
        n = 5
        rho = np.array([1 + (i * 7 + fixture) % 11 for i in range(n)], dtype=float)
        eta = np.array([1 + (i * 3 + fixture * 2) % 13 for i in range(n)], dtype=float)
        rho /= rho.sum()
        eta /= eta.sum()
        reward = np.sin(np.arange(n) * 2.1 + fixture)
        weight = 0.2 + 0.8 * np.array([
            [math.exp(-abs(i - j) / (fixture + 1)) for j in range(n)] for i in range(n)
        ])
        kr, ke = weight * rho, weight * eta
        kr /= kr.sum(axis=1, keepdims=True)
        ke /= ke.sum(axis=1, keepdims=True)
        jr, je = rho[:, None] * kr, eta[:, None] * ke
        delta = float(np.abs(rho - eta).sum())
        mu_r, mu_e = float(rho @ reward), float(eta @ reward)
        var_r, var_e = float(rho @ (reward - mu_r) ** 2), float(eta @ (reward - mu_e) ** 2)
        inputs = {
            "rho": rho.tolist(),
            "eta": eta.tolist(),
            "reward": reward.tolist(),
            "weight": weight.tolist(),
            "a": 0.2,
            "full_variation": delta,
        }
        add(
            "lem-uniqueness-lipschitz-moments",
            "|\\rho R-\\eta R|",
            f"reward-mean-{fixture}",
            abs(mu_r - mu_e),
            delta,
            "upper",
            inputs,
        )
        add(
            "lem-uniqueness-lipschitz-moments",
            "|\\operatorname{Var}_\\rho R",
            f"reward-var-{fixture}",
            abs(var_r - var_e),
            3 * delta,
            "upper",
            inputs,
        )
        add(
            "lem-uniqueness-lipschitz-moments",
            "\\sup_z\\|K_\\rho",
            f"kernel-{fixture}",
            float(np.abs(kr - ke).sum(axis=1).max()),
            10 * delta,
            "upper",
            inputs,
        )
        add(
            "lem-uniqueness-lipschitz-moments",
            "\\|J_\\rho-J_\\eta",
            f"joint-{fixture}",
            float(np.abs(jr - je).sum()),
            11 * delta,
            "upper",
            inputs,
        )
        f, g = rho * (0.3 + fixture / 50), eta * (0.4 + fixture / 100)
        add(
            "lem-uniqueness-lipschitz-moments",
            "\\frac f{\\int f}",
            f"alive-normalize-{fixture}",
            float(np.abs(f / f.sum() - g / g.sum()).sum()),
            2 / 0.3 * float(np.abs(f - g).sum()),
            "upper",
            {"f": f.tolist(), "g": g.tolist(), "minimum_alive_mass": 0.3},
        )
        s1, s2 = math.sqrt(var_r + 0.1**2), math.sqrt(var_e + 0.1**2)
        for raw in reward:
            add(
                "lem-uniqueness-lipschitz-fitness-potential",
                "\\left|\\frac{u-m_1}",
                f"standardization-{fixture}-{raw}",
                abs((raw - mu_r) / s1 - (raw - mu_e) / s2),
                abs(mu_r - mu_e) / 0.1 + abs(raw - mu_e) / 0.1**2 * abs(s1 - s2),
                "upper",
                {"u": raw, "m1": mu_r, "m2": mu_e, "s1": s1, "s2": s2, "s_floor": 0.1},
            )

    # Exactly enumerate independent increasing outgoing edges: N<=6, at most
    # one edge per row. This is the conditional ordered forest law, not a
    # replacement of the native canonical operator.
    c = 0.8
    m2 = (1 + 2 * c) * math.exp(4 * c)
    m3 = (1 + 6 * c + 3 * c * c) * math.exp(8 * c)
    for n in range(2, 7):
        first = np.zeros(n)
        second = np.zeros(n)
        third = np.zeros(n)
        tails = np.zeros((n, n))
        sizes = np.zeros((n, n + 1))
        same_component = 0.0
        probability = 0.0
        choices = [[None, *range(i + 1, n)] for i in range(n)]
        for edges in itertools.product(*choices):
            prob = math.prod(
                1 - c * (n - i - 1) / n if j is None else c / n for i, j in enumerate(edges)
            )
            probability += prob
            adjacent = [[] for _ in range(n)]
            for i, j in enumerate(edges):
                if j is not None:
                    adjacent[i].append(j)
                    adjacent[j].append(i)
            for root in range(n):
                seen = {root: 0}
                queue = [root]
                for i in queue:
                    for j in adjacent[i]:
                        if j not in seen:
                            seen[j] = seen[i] + 1
                            queue.append(j)
                size = len(seen)
                first[root] += prob * size
                second[root] += prob * size**2
                third[root] += prob * size**3
                radius = max(seen.values())
                for r in range(1, n):
                    tails[root, r] += prob * (radius >= r)
                for k in range(1, n + 1):
                    sizes[root, k] += prob * (size > k)
                same_component += prob * (size - 1) / (n * (n - 1))
        for root in range(n):
            operands = {
                "N": n,
                "C": c,
                "root_address": root,
                "edge_probability": c / n,
                "all_forests": math.factorial(n),
                "probability_sum": probability,
            }
            add(
                "lem-chaos-component-truncation",
                "e^{2C}",
                f"forest-first-N{n}-r{root}",
                first[root],
                math.exp(2 * c),
                "upper",
                operands,
            )
            for p, value, bound in [(2, second[root], m2), (3, third[root], m3)]:
                add(
                    "lem-chaos-component-moments",
                    "M_p(C)=",
                    f"forest-moment-N{n}-r{root}-p{p}",
                    value,
                    bound,
                    "upper",
                    {**operands, "p": p},
                )
            for radius in range(1, n):
                add(
                    "lem-chaos-component-truncation",
                    "\\frac{(2C)^r}{r!}",
                    f"forest-radius-N{n}-root{root}-r{radius}",
                    tails[root, radius],
                    (2 * c) ** radius / math.factorial(radius),
                    "upper",
                    {**operands, "radius": radius},
                )
            for k in range(1, n + 1):
                add(
                    "lem-chaos-component-truncation",
                    "e^{2C}/K",
                    f"forest-size-N{n}-root{root}-K{k}",
                    sizes[root, k],
                    math.exp(2 * c) / k,
                    "upper",
                    {**operands, "K": k},
                )
        add(
            "thm-chaos-rooted-collision-limit",
            "\\frac{\\mathbb E(|\\mathcal C_N(I)|-1)}{N-1}",
            f"two-root-N{n}",
            same_component,
            (float(first.mean()) - 1) / (n - 1),
            inputs={"N": n, "C": c, "root_uniform": True},
        )

    # Efron--Stein exact finite innovation product law. Includes common output
    # dependence; outputs are not counted as independent samples.
    for n in [2, 4, 8]:
        outcomes = list(itertools.product([0, 1], repeat=n))
        vals = np.array([sum(x) / n + 0.1 * math.prod(x) for x in outcomes])
        influence = 0.0
        for row in range(n):
            for x in outcomes:
                for bit in [0, 1]:
                    y = list(x)
                    y[row] = bit
                    fy = sum(y) / n + 0.1 * math.prod(y)
                    influence += (sum(x) / n + 0.1 * math.prod(x) - fy) ** 2 / (len(outcomes) * 2)
        add(
            "lem-chaos-innovation-variance",
            "\\frac12\\sum",
            f"efron-stein-N{n}",
            float(vals.var()),
            influence / 2,
            "upper",
            {
                "N": n,
                "independent_innovations": n,
                "uniform_binary_law": True,
                "observable": "mean + .1*all-one interaction",
            },
        )

    # Ratio inequality includes zero-alive outputs via their declared auxiliary
    # probability. No dead physical displacement enters this observable.
    for mass in [0.1, 0.3, 0.8, 1.0]:
        for u0 in [-mass, 0.0, mass]:
            for v in [0.0, 0.1, 0.5, 1.0]:
                for u in sorted({-v, 0.0, v}):
                    actual = abs(u / v - u0 / mass) if v else 1 + abs(u0 / mass)
                    add(
                        "thm-chaos-conditioned-quantitative-map",
                        "\\left|\\frac Uv-\\frac um",
                        f"alive-ratio-{mass}-{u0}-{v}-{u}",
                        actual,
                        (abs(u - u0) + abs(v - mass)) / mass,
                        "upper",
                        {
                            "U": u,
                            "v": v,
                            "u": u0,
                            "m": mass,
                            "zero_alive_auxiliary": "worst bounded test",
                        },
                    )

    # Explicit nonconstant-survival finite QSD. The kernel is constructed from
    # a Doob transform and has known positive eigenfunction/invariant law.
    pmat = np.array([[0.8, 0.2], [0.3, 0.7]])
    eig = np.array([1.0, 1.2])
    alpha = 0.8
    kernel = alpha * eig[:, None] * pmat / eig[None, :]
    nu = np.array([0.6, 0.4]) / eig
    nu /= nu.sum()
    q = kernel.sum(axis=1)
    h = np.array([-0.8, 0.7])
    tilted = nu * q / alpha
    r = (kernel @ h) / q
    s = (kernel @ (h * h)) / q - r * r

    def variance(law, f):
        return float(law @ ((f - law @ f) ** 2))

    add(
        "thm-chaos-qsd-variance-budget",
        "\\widetilde\\nu_Ns_N+",
        "qsd-total-variance",
        variance(nu, h),
        float(tilted @ s) + variance(tilted, r),
        inputs={
            "Q": kernel.tolist(),
            "nu": nu.tolist(),
            "alpha": alpha,
            "H": h.tolist(),
            "scope": "explicit two-state substochastic fixture, not a fitted native QSD",
        },
    )
    delta = float((1 - q).max())
    add(
        "thm-chaos-qsd-variance-budget",
        "\\|\\widetilde\\nu_N-\\nu_N\\|_1",
        "qsd-reweighted-full-variation",
        float(np.abs(tilted - nu).sum()),
        2 * delta / alpha,
        "upper",
        {"Q": kernel.tolist(), "delta": delta, "alpha": alpha},
    )
    add(
        "thm-chaos-qsd-variance-budget",
        "|\\operatorname{Var}_\\rho(f)",
        "qsd-variance-variation",
        abs(variance(nu, r) - variance(tilted, r)),
        3 * 0.8**2 * float(np.abs(nu - tilted).sum()),
        "upper",
        {"rho": nu.tolist(), "eta": tilted.tolist(), "f": r.tolist(), "M": 0.8},
    )
    # Exact filtering differs from repeatedly applying normalized kernel rows.
    eta = np.array([0.4, 0.6])
    for step in range(1, 17):
        next_full = eta @ kernel
        extinction = 1 - float(next_full.sum())
        survivor = next_full / next_full.sum()
        complete = np.append(next_full, extinction)
        cond = np.append(survivor, 0.0)
        add(
            "thm-chaos-survival-uniform-floor",
            "=e_n^{\\dagger}",
            f"filter-TV-{step}",
            float(np.abs(complete - cond).sum() / 2),
            extinction,
            inputs={
                "eta_n": eta.tolist(),
                "Q": kernel.tolist(),
                "own_survival_denominator": float(next_full.sum()),
            },
        )
        eta = survivor

    for a0, p in [(0.2, 0.3), (0.7, 0.4), (1.0, 1.0)]:
        n0 = math.ceil(max(8 / p, 16 / a0) * math.log(4))
        delta = min(1, math.exp(-p * n0 / 8) + math.exp(-a0 * n0 / 16))
        add(
            "def-chaos-survival-filter",
            "N_{\\mathrm{surv}}=",
            f"survival-threshold-{a0}-{p}",
            delta,
            0.5,
            "upper",
            {"a0": a0, "p": p, "N_surv": n0},
        )
        for n in [2, 8, 32, 128, 512]:
            hazard = min(1, math.exp(-p * n / 8) + math.exp(-a0 * n / 16))
            decay = min(p / 8, a0 / 16)
            for steps in [1, 8, 128]:
                exact = 1 - (1 - hazard) ** steps
                add(
                    "thm-chaos-conditioned-propagation",
                    "1-(1-\\delta_N)^T",
                    f"path-hazard-{a0}-{p}-{n}-{steps}",
                    exact,
                    steps * hazard,
                    "upper",
                    {"N": n, "a0": a0, "p": p, "delta_N": hazard, "T": steps},
                )
            c = decay / 2
            steps = math.floor(math.exp(min(c * n, 600)))
            exact = -math.expm1(steps * math.log1p(-hazard)) if hazard < 1 else 1.0
            add(
                "thm-chaos-conditioned-propagation",
                "2e^{-(c_{\\rm surv}-c)N}",
                f"exponential-window-{a0}-{p}-{n}",
                exact,
                2 * math.exp(-(decay - c) * n),
                "upper",
                {"N": n, "a0": a0, "p": p, "c": c, "T": steps},
            )

    # Gaussian safe-center upper tails and exact independent terminal extinction.
    for d in [1, 2, 4, 8]:
        for sigma in [0.02, 0.2, 1.0]:
            box = 1.0
            qtau = 1 - (2 * ndtr(box / sigma) - 1) ** d
            for n in [2, 4, 16]:
                for center in [0.0, 0.25, 0.7, 1.2]:
                    inside = (ndtr((box - center) / sigma) - ndtr((-box - center) / sigma)) ** d
                    hazard = (1 - inside) ** n
                    add(
                        "cor-chaos-exact-hazard-recovery-window",
                        "q_\\tau^N\\le h_N(S)",
                        f"gaussian-product-d{d}-s{sigma}-n{n}-c{center}",
                        qtau**n,
                        hazard,
                        "upper",
                        {
                            "d": d,
                            "tau": sigma,
                            "N": n,
                            "centers": center,
                            "box": [-1, 1],
                            "actual_P_D_tau": inside,
                        },
                    )
    # Finite Doob/eigenfunction identities and uniform conditioned convergence
    # on this exact substochastic fixture. These are algebraic checks of the
    # QSD theorem, not a native QSD eigenvalue estimated from finite runs.
    epsilon = 2 * float(kernel.min())
    theta = np.array([0.5, 0.5])
    delta_doob = epsilon * float(theta @ eig) / (alpha * float(eig.max()))
    theta_hat = theta * eig / (theta @ eig)
    condition_constant = 4 * (eig.max() / eig.min()) ** 2
    doob = kernel * eig[None, :] / (alpha * eig[:, None])
    add(
        "thm-chaos-canonical-finite-n-qsd",
        "P_N(k,dl)=",
        "finite-Doob-kernel",
        float(np.abs(doob - pmat).max()),
        inputs={"Q": kernel.tolist(), "e": eig.tolist(), "alpha": alpha, "P": pmat.tolist()},
    )
    add(
        "thm-chaos-canonical-finite-n-qsd",
        "\\nu_N(dl)=",
        "finite-eigen-law",
        float(np.abs(nu @ kernel - alpha * nu).max()),
        inputs={"Q": kernel.tolist(), "e": eig.tolist(), "alpha": alpha, "nu": nu.tolist()},
    )
    add(
        "thm-chaos-canonical-finite-n-qsd",
        "\\delta_N=",
        "finite-Doob-minorization",
        float(np.max(delta_doob * theta_hat[None, :] - doob)),
        0,
        "upper",
        {
            "epsilon": epsilon,
            "theta": theta.tolist(),
            "theta_hat": theta_hat.tolist(),
            "delta_N": delta_doob,
            "P": doob.tolist(),
        },
    )
    for step in [1, 2, 4, 8, 16, 32]:
        power = np.linalg.matrix_power(kernel, step)
        doob_power = np.linalg.matrix_power(doob, step)
        for weight in np.linspace(0, 1, 11):
            eta = np.array([weight, 1 - weight])
            alive = eta @ power
            conditioned = alive / alive.sum()
            add(
                "thm-chaos-canonical-finite-n-qsd",
                "\\leq C_Nr_N^n",
                f"finite-qsd-convergence-{step}-{weight}",
                float(np.abs(conditioned - nu).sum()),
                float(condition_constant * (1 - delta_doob) ** step),
                "upper",
                {
                    "eta": eta.tolist(),
                    "Q": kernel.tolist(),
                    "nu": nu.tolist(),
                    "C_N": condition_constant,
                    "r_N": 1 - delta_doob,
                    "step": step,
                    "variation_convention": "full signed-measure norm as explicitly declared in this proof",
                },
            )
            left = float(eta @ power @ h)
            right = float(alpha**step * (eta * eig) @ doob_power @ (h / eig))
            add(
                "thm-chaos-canonical-finite-n-qsd",
                "\\eta Q_N^nf=",
                f"finite-Doob-semigroup-{step}-{weight}",
                left,
                right,
                inputs={
                    "eta": eta.tolist(),
                    "Q": kernel.tolist(),
                    "e": eig.tolist(),
                    "alpha": alpha,
                    "P": doob.tolist(),
                    "f": h.tolist(),
                    "step": step,
                },
            )
    add(
        "def-sequence-of-qsds",
        "\\nu_NQ_N=",
        "finite-qsd-definition",
        float(np.abs(nu @ kernel - alpha * nu).max()),
        inputs={
            "Q": kernel.tolist(),
            "nu": nu.tolist(),
            "alpha": alpha,
            "model": "exact finite fixture",
        },
    )
    add(
        "rem-qsd-vs-true-stationarity",
        "\\nu_N(Q_N-I)H=",
        "finite-qsd-stationary-defect",
        float(nu @ (kernel - np.eye(2)) @ h),
        -(1 - alpha) * float(nu @ h),
        inputs={"Q": kernel.tolist(), "nu": nu.tolist(), "alpha": alpha, "H": h.tolist()},
    )
    add(
        "thm-extinction-rate-vanishes",
        "1-\\alpha_N=",
        "finite-qsd-extinction-identity",
        1 - alpha,
        float(nu @ (1 - kernel.sum(axis=1))),
        inputs={"Q": kernel.tolist(), "nu": nu.tolist(), "alpha": alpha, "delta_N": delta},
    )

    # Exact entropic concentration under correlated Bernoulli joint laws,
    # including the product law and a fully correlated all-zero/all-one law.
    for n in [2, 4, 6, 8]:
        outcomes = list(itertools.product([0, 1], repeat=n))
        product = np.repeat(1 / len(outcomes), len(outcomes))
        phi = np.array([sum(2 * x - 1 for x in row) / n for row in outcomes])
        correlated = np.zeros(len(outcomes))
        correlated[0] = correlated[-1] = 0.5
        for mix in [0.0, 0.1, 0.5, 1.0]:
            joint = (1 - mix) * product + mix * correlated
            positive = joint > 0
            entropy = float(np.sum(joint[positive] * np.log(joint[positive] / product[positive])))
            observed = float(joint @ (phi * phi))
            bound = 4 / n * (entropy + 0.5 * math.log(2))
            add(
                "lem-chaos-concentration-criterion",
                "H_N+\\tfrac12\\log2",
                f"joint-entropy-N{n}-mix{mix}",
                observed,
                bound,
                "upper",
                {
                    "N": n,
                    "rho": "Bernoulli(1/2)",
                    "joint_mixture": mix,
                    "joint_relative_entropy": entropy,
                    "M": 1,
                    "full_states": len(outcomes),
                },
            )

    # Exact heterogeneous Poisson-binomial final alive counts after preparation.
    # Safe centers have guaranteed margin r. No preparation probability is
    # silently inferred from a sampled fraction.
    for d in [1, 2, 4]:
        for sigma in [0.0, 0.2, 1.0]:
            for n in [2, 4, 8]:
                r = 0.2
                epsilon_tail = min(1, 2 * d * float(ndtr(-r / sigma))) if sigma else 0.0
                for safe in sorted({1, n // 2, n}):
                    centers = [0.7] * (safe) + [1.3] * (n - safe)
                    probs = [
                        (
                            float(ndtr((1 - z) / sigma) - ndtr((-1 - z) / sigma)) ** d
                            if sigma
                            else float(abs(z) <= 1)
                        )
                        for z in centers
                    ]
                    count = np.array([1.0])
                    for prob in probs:
                        count = np.convolve(count, [1 - prob, prob])
                    for k in range(1, safe + 1):
                        binomial = sum(
                            math.comb(safe, j)
                            * (1 - epsilon_tail) ** j
                            * epsilon_tail ** (safe - j)
                            for j in range(k)
                        )
                        observed = float(count[:k].sum())
                        add(
                            "prop-chaos-safe-center-noise",
                            "B_{g,k}(p_r",
                            f"safe-count-d{d}-s{sigma}-N{n}-g{safe}-k{k}",
                            observed,
                            binomial,
                            "upper",
                            {
                                "d": d,
                                "tau": sigma,
                                "r": r,
                                "g": safe,
                                "k": k,
                                "centers": centers,
                                "safe_center_preparation_probability": 1,
                                "epsilon_r": epsilon_tail,
                                "full_alive_count_law": count.tolist(),
                            },
                        )
                    add(
                        "prop-chaos-safe-center-noise",
                        "\\epsilon_r(\\tau)^g",
                        f"safe-extinction-d{d}-s{sigma}-N{n}-g{safe}",
                        count[0],
                        epsilon_tail**safe,
                        "upper",
                        {
                            "d": d,
                            "tau": sigma,
                            "r": r,
                            "g": safe,
                            "centers": centers,
                            "eta_preparation": 0,
                        },
                    )

    # Uniform integrability/tightness tails use unbounded values, not a box
    # replacing the mathematical state space.
    values = np.array([0.0, 1.0, 4.0, 16.0, 64.0])
    law = np.array([0.5, 0.25, 0.125, 0.0625, 0.0625])
    first = float(law @ values)
    for radius in [1, 2, 8, 32, 128]:
        probability = float(law @ (values > radius))
        add(
            "thm-qsd-marginals-are-tight",
            "C_W/R",
            f"tightness-tail-R{radius}",
            probability,
            first / radius,
            "upper",
            {"W_values": values.tolist(), "law": law.tolist(), "C_W": first, "R": radius},
        )
        eps = 0.5
        tail = float(law @ (values * (values > radius)))
        moment = float(law @ (values ** (1 + eps)))
        add(
            "lem-uniform-integrability",
            "R^{-\\epsilon}",
            f"uniform-integrability-R{radius}",
            tail,
            radius ** (-eps) * moment,
            "upper",
            {
                "Y": values.tolist(),
                "law": law.tolist(),
                "epsilon": eps,
                "moment_1_plus_epsilon": moment,
                "R": radius,
            },
        )

    # Conditional stationary variance budgets on the explicit substochastic
    # fixture. A is computed from the full output including an extinction
    # value of zero. No stationary law is estimated from native trajectories.
    r = (kernel @ h) / q
    s = (kernel @ (h * h)) / q - r * r
    N_fixture = 2
    M = float(np.abs(h).max())
    conditional_full_var = kernel @ (h * h) - (kernel @ h) ** 2
    A_fixture = N_fixture * float(conditional_full_var.max())
    one_step = float(tilted @ s)
    base = A_fixture / (N_fixture * alpha) + M * M * delta / alpha
    common_inputs = {
        "Q": kernel.tolist(),
        "nu": nu.tolist(),
        "alpha": alpha,
        "N": N_fixture,
        "H": h.tolist(),
        "M": M,
        "delta_N": delta,
        "A_phi": A_fixture,
        "scope": "Finite substochastic fixture with independently computed full-output conditional variances, not a native population stationary law",
    }
    add(
        "thm-chaos-qsd-variance-budget",
        "0\\leq \\widetilde\\nu_Ns_N",
        "qsd-injected-variance",
        one_step,
        base,
        "upper",
        common_inputs,
    )
    add(
        "thm-chaos-qsd-variance-budget",
        "-\\operatorname{Var}_{\\nu_N}(r_N)",
        "qsd-variance-balance",
        abs(variance(nu, h) - variance(nu, r)),
        base + 6 * M * M * delta / alpha,
        "upper",
        common_inputs,
    )
    add(
        "thm-chaos-qsd-variance-budget",
        "\\nu_NH=",
        "qsd-first-second-moment",
        abs(float(nu @ h - tilted @ r)) + abs(float(nu @ (h * h) - tilted @ (s + r * r))),
        inputs=common_inputs,
    )
    for i in range(2):
        mean = float(kernel[i] @ h)
        full_var = float(kernel[i] @ (h * h) - mean**2)
        restricted = float(kernel[i] @ ((h - mean) ** 2))
        add(
            "thm-chaos-qsd-variance-budget",
            "q_N(S)s_N(S)",
            f"qsd-conditional-var-row{i}",
            float(q[i] * s[i]),
            restricted,
            "upper",
            {
                **common_inputs,
                "row": i,
                "full_variance": full_var,
                "restricted_variance": restricted,
            },
        )
    add(
        "thm-chaos-qsd-variance-budget",
        "\\int|q_N-\\alpha_N|",
        "qsd-survival-tilt-budget",
        float(nu @ np.abs(q - alpha)),
        2 * delta,
        "upper",
        common_inputs,
    )
    for t in np.linspace(0, 1, 21):
        g = (1 - t) * h + t * r
        fg = float(nu @ np.abs(h - g))
        add(
            "thm-chaos-qsd-variance-budget",
            "|\\operatorname{Var}(f)-\\operatorname{Var}(g)|",
            f"qsd-function-variance-change-{t}",
            abs(variance(nu, h) - variance(nu, g)),
            4 * M * fg,
            "upper",
            {"f": h.tolist(), "g": g.tolist(), "nu": nu.tolist(), "M": M, "E_abs_difference": fg},
        )

    # Conditioning a bad alive-count event subtracts extinction before
    # division, so the alive-floor bound is not inflated by 1/(1-delta).
    for delta_bound in [0.01, 0.2, 0.8, 1.0]:
        for extinction in np.linspace(0, min(delta_bound, 0.9), 9):
            filtered = (delta_bound - extinction) / (1 - extinction)
            add(
                "thm-chaos-survival-uniform-floor",
                "\\frac{\\delta_N-e}{1-e}\\le\\delta_N",
                f"floor-conditioning-{delta_bound}-{extinction}",
                filtered,
                delta_bound,
                "upper",
                {
                    "P_bad": delta_bound,
                    "extinction_subset_bad": extinction,
                    "own_survival_denominator": 1 - extinction,
                },
            )
            add(
                "cor-chaos-noise-conditioned-law",
                "\\frac{\\beta_n^k-e_n}{1-e_n}\\le\\beta_n^k",
                f"safe-conditioning-{delta_bound}-{extinction}",
                filtered,
                delta_bound,
                "upper",
                {"beta_k": delta_bound, "e_n": extinction, "U_n": delta_bound},
            )

    for n in [2, 8, 32, 128]:
        for B in [0.0, 0.2, 1.0, 3.0]:
            probability = min(B / n, 1.0)
            eq2 = (1 + (n - 1) * probability) ** 2 + (n - 1) * probability * (1 - probability)
            add(
                "lem-chaos-canonical-innovation-replacement",
                "\\mathbb EQ^2",
                f"exceptional-rows-N{n}-B{B}",
                eq2,
                (1 + B) ** 2 + B,
                "upper",
                {
                    "N": n,
                    "B": B,
                    "row_probability": probability,
                    "Q": "1+Binomial(N-1,q)",
                    "independent_rows": True,
                },
            )
            if probability <= 0.5:
                C = 0.8
                add(
                    "lem-chaos-canonical-innovation-replacement",
                    "\\frac{C/N}{1-q_i}",
                    f"conditional-edge-N{n}-B{B}",
                    (C / n) / (1 - probability),
                    2 * C / n,
                    "upper",
                    {"N": n, "C": C, "q_i": probability, "conditional_no_change": 1 - probability},
                )

    for ell in range(13):
        C = 0.8
        observed = sum(
            C**ell / (math.factorial(a) * math.factorial(ell - a)) for a in range(ell + 1)
        )
        add(
            "lem-chaos-component-truncation",
            "\\sum_{a=0}^{\\ell}",
            f"ordered-path-count-{ell}",
            observed,
            (2 * C) ** ell / math.factorial(ell),
            inputs={"C": C, "ell": ell},
        )
    for n in [4, 8, 16, 64]:
        C = 0.8
        for p in [1, 2, 3, 4, 8]:
            for count in range(1, n + 1):
                prob = min(C * count / n, 1.0)
                expected = count**p + prob * ((count + 1) ** p - count**p)
                upper = (1 + (2**p - 1) * C / n) * count**p
                add(
                    "lem-chaos-component-moments",
                    "[1+(2^p-1)C/N]s^p",
                    f"component-scan-N{n}-p{p}-s{count}",
                    expected,
                    upper,
                    "upper",
                    {"N": n, "C": C, "p": p, "s": count, "joining_probability": prob},
                )

    return checks


def constant_comparisons(inventory, path):
    """Independent 70-digit evaluation of each full Rust constant formula."""
    import mpmath as mp

    mp.mp.dps = 70
    data = json.loads(path.read_text())
    checks = []
    expressions = inventory["quantitative_expressions"]
    source = SOURCE.read_text()
    specifications = {
        "l_q": ("L_q=", "lem-chaos-canonical-innovation-replacement"),
        "h_s": ("H_s=", "lem-chaos-canonical-innovation-replacement"),
        "l_0": ("L_0=H_sL_q", "lem-chaos-canonical-innovation-replacement"),
        "l_a": ("L_a=\\max", "lem-chaos-canonical-innovation-replacement"),
        "b": ("B=C+2L_aL_0", "lem-chaos-canonical-innovation-replacement"),
        "a_t": ("A_T=", "thm-chaos-canonical-quantitative-bias"),
        "l_t": ("L_T=2L_aH_s", "thm-chaos-canonical-quantitative-bias"),
        "d_d": ("D_D=", "thm-chaos-canonical-quantitative-bias"),
        "a": ("A=1+C+D_D", "thm-chaos-canonical-quantitative-bias"),
        "n_0": ("N_0=\\lceil", "thm-chaos-canonical-quantitative-bias"),
        "log_m1_c": ("M_p(C)=", "lem-chaos-component-moments"),
        "log_m2_c": ("M_2(C)=", "lem-chaos-component-moments"),
        "log_m3_c": ("M_3(C)=", "lem-chaos-component-moments"),
        "log_a_d": ("A_D=", "lem-chaos-canonical-innovation-replacement"),
        "log_a_phi_unit": ("A_\\varphi=", "thm-chaos-canonical-conditional-variance"),
        "log_b_star": ("B_*=", "thm-chaos-canonical-quantitative-bias"),
        "log_c_update": ("C_{\\mathrm{upd}}=", "thm-chaos-conditioned-quantitative-map"),
        "log_k_ordered": ("K_{\\rm ord}(C)=", "thm-chaos-ordered-star-quantitative"),
        "log_a_ordered_unit": ("A_{\\rm ord,\\varphi}=", "thm-chaos-ordered-star-quantitative"),
        "log_b_ordered": ("B_{\\rm ord}=", "thm-chaos-ordered-star-quantitative"),
    }
    profiles = list(data["profiles"])
    if data.get("native_population_comparisons"):
        profiles.extend(data["native_population_comparisons"]["cases"])
    for profile_index, profile in enumerate(profiles):
        record = profile["constants"]
        q = {k: mp.mpf(str(v)) for k, v in record["input"].items()}
        m = q["minimum_alive_fraction"]
        s = q["separation_upper"]
        sigma = q["separation_scale_floor"]
        c = 2 / (q["clone_weight_lower"] * m)
        dd = 2 / (q["measurement_weight_lower"] * m)
        lq = s / (m * sigma) + 3 * s**3 / (2 * m * sigma**3)
        gr = q["reward_map_floor"] + q["reward_map_amplitude"]
        gs = q["separation_map_floor"] + q["separation_map_amplitude"]
        er = q["reward_exponent"]
        es = q["separation_exponent"]
        hs = (
            gr**er
            * q["separation_map_amplitude"]
            / 4
            * es
            * max(q["separation_map_floor"] ** (es - 1), gs ** (es - 1))
            if es
            else mp.mpf(0)
        )
        l0 = hs * lq
        fmin = q["reward_map_floor"] ** er * q["separation_map_floor"] ** es
        fmax = gr**er * gs**es
        la = max(
            1 / (q["acceptance_saturation"] * (fmin + q["acceptance_epsilon"])),
            (fmax + q["acceptance_epsilon"])
            / (q["acceptance_saturation"] * (fmin + q["acceptance_epsilon"]) ** 2),
        )
        b = c + 2 * la * l0
        at = (1 / m + dd**2) * (2 * s**2 / sigma**2 + 5 * s**6 / sigma**6)
        lt = 2 * la * hs
        a = 1 + c + dd
        n0 = mp.ceil((8 * a) ** (mp.mpf(6) / 5))

        def mm(cc, p):
            return (
                mp.exp(2 * cc)
                if p == 1
                else (1 + 2 * cc) * mp.exp(4 * cc)
                if p == 2
                else (1 + 6 * cc + 3 * cc**2) * mp.exp(8 * cc)
            )

        ad = 9 * mm(2 * c, 2) * ((1 + b) ** 2 + b) + max(1, 4 * b**2)
        avar = 2 * (ad + 10 * mm(c, 2) + 1)
        common = 3 * mm(2 * c, 1) * lt * mp.sqrt(at) + 4 * lt**2 * at
        bs = common + 64 * a**2 + 16 * mm(c, 3) + mp.sqrt(n0)
        k = 27 + 30 * c + 6 * c**2
        independent = {
            "l_q": lq,
            "h_s": hs,
            "l_0": l0,
            "l_a": la,
            "b": b,
            "a_t": at,
            "l_t": lt,
            "d_d": dd,
            "a": a,
            "n_0": n0,
            "log_m1_c": mp.log(mm(c, 1)),
            "log_m2_c": mp.log(mm(c, 2)),
            "log_m3_c": mp.log(mm(c, 3)),
            "log_a_d": mp.log(ad),
            "log_a_phi_unit": mp.log(avar),
            "log_b_star": mp.log(bs),
            "log_c_update": mp.log(avar + 4 * bs**2) / 2,
            "log_k_ordered": mp.log(k),
            "log_a_ordered_unit": mp.log(2 * (ad + k + 1)),
            "log_b_ordered": mp.log(common + 128 * a**2 + 16 * mm(c, 3) + mp.sqrt(n0)),
        }
        for key, (needle, label) in specifications.items():
            matches = [e for e in expressions if needle in e["formula"]]
            if not matches and needle not in source:
                raise ValueError(f"Missing constant source: {needle}")
            observed = record[key]
            expected = float(independent[key])
            relative = abs(observed - expected) / max(1, abs(expected))
            checks.append({
                "id": f"independent-constant-{profile_index}-{key}",
                "source_label": label,
                "source_quote": needle,
                "source_expressions": [
                    {"id": e["id"], "formula": e["formula"], "line": e["source_line"]}
                    for e in matches
                ],
                "observed": observed,
                "bound": expected,
                "relation": "equal",
                "relative_residual": relative,
                "passed": relative < 2e-12,
                "inputs": record["input"],
                "independent_precision_digits": 70,
                "scope": "Independent high precision whole constant evaluation. Constant definitions are tested; their global influence/map applicability remains subject to the stated analytic hypotheses. Ordered-star expressions refer only to the separate operator.",
            })
    # Independent integer-coefficient forward difference of falling factorials:
    # E[(Poisson(C)+1)^p-Poisson(C)^p] = sum k*S(p,k)*C^(k-1).
    for profile_index, profile in enumerate(data["profiles"]):
        c = mp.mpf(str(profile["constants"]["c"]))
        for moment in profile.get("component_log_moments", []):
            p = moment["order"]
            row = [1]
            for j in range(1, p + 1):
                row = [0] + [
                    (k * row[k] if k < len(row) else 0) + row[k - 1] for k in range(1, j + 1)
                ]
            polynomial = sum(mp.mpf(k * row[k]) * c ** (k - 1) for k in range(1, p + 1))
            expected = mp.mpf(2) ** p * c + mp.log(polynomial)
            observed = moment["log_upper"]
            relative = abs(observed - float(expected)) / max(1, abs(float(expected)))
            match = next(e for e in expressions if "M_p(C)=" in e["formula"])
            checks.append({
                "id": f"general-moment-{profile_index}-p{p}",
                "source_label": "lem-chaos-component-moments",
                "source_expressions": [
                    {"id": match["id"], "formula": match["formula"], "line": match["source_line"]}
                ],
                "observed": observed,
                "bound": float(expected),
                "relation": "equal",
                "relative_residual": relative,
                "passed": relative < 2e-12,
                "inputs": {
                    "C": float(c),
                    "p": p,
                    "exact_forward_difference_coefficients": [k * row[k] for k in range(1, p + 1)],
                },
                "independent_precision_digits": 70,
                "scope": "Arbitrary moment series evaluated by independent exact-integer falling-factorial forward-difference coefficients, then 70-digit arithmetic. No component observations or assumptions inferred here.",
            })
    return checks


def native_evidence(inventory, path):
    data = json.loads(path.read_text())
    native = data.get("native_population_comparisons")
    if not native:
        return []
    records = []
    archive_index = json.loads((path.parent / "archive-index.json").read_text())
    entries = {e["path"]: e for e in archive_index["entries"]}
    ratio_expression = next(
        e
        for e in inventory["quantitative_expressions"]
        if "\\left|\\frac Uv-\\frac um" in e["formula"]
    )
    for case in native["cases"]:
        for c in case["checks"]:
            needle = (
                "A_\\varphi=2"
                if c["kind"] == "conditional-variance"
                else (
                    "2\\|\\varphi\\|_\\infty B_*"
                    if c["kind"] == "conditional-bias"
                    else "A_\\varphi+4"
                )
            )
            matches = [e for e in inventory["quantitative_expressions"] if needle in e["formula"]]
            if not matches:
                message = f"Missing native source expression for {needle}"
                raise ValueError(message)
            records.append({
                **c,
                "id": f"native-{case['id']}-{c['kind']}-test{c['test_index']}",
                "source_expressions": [
                    {"id": e["id"], "formula": e["formula"], "line": e["source_line"]}
                    for e in matches
                ],
                "inputs": {
                    "N": case["N"],
                    "d": case["d"],
                    "profile": case["profile"],
                    "constants": case["constants"],
                    "native_seed_denominator": case["native_replicas"],
                    "independent_root_denominator": case["root_draws"],
                    "lossless_operands": case["operands"],
                    "input_archive_index_sha256": native["input_index_sha256"],
                },
            })
        relative = case["operands"]
        file = path.parent / relative
        if sha(file) != entries[relative]["sha256"]:
            message = "Saved native probability-normalized operands checksum mismatch"
            raise ValueError(message)
        with gzip.open(file, "rt") as handle:
            operands = json.load(handle)
        mass = case["reference_mean"][3]
        if mass <= 0:
            message = "Independent finite reference has zero alive mass: increase reference draws"
            raise ValueError(message)
        for row_index, values in enumerate(operands["native_values"]):
            alive_mass = values[3]
            for j in range(3):
                U = values[4 + j]
                u = case["reference_mean"][4 + j]
                physical = U / alive_mass if alive_mass > 0 else 0.0
                reference_physical = u / mass
                observed = abs(physical - reference_physical)
                bound = (abs(U - u) + abs(alive_mass - mass)) / mass
                records.append({
                    "id": f"alive-ratio-native-{case['id']}-seed{row_index}-test{j}",
                    "source_label": "thm-chaos-conditioned-quantitative-map",
                    "source_expressions": [
                        {
                            "id": ratio_expression["id"],
                            "formula": ratio_expression["formula"],
                            "line": ratio_expression["source_line"],
                        }
                    ],
                    "kind": "pathwise normalized alive ratio",
                    "observed": observed,
                    "bound": bound,
                    "relation": "upper",
                    "passed": observed <= bound + 1e-12,
                    "inputs": {
                        "N": case["N"],
                        "d": case["d"],
                        "profile": case["profile"],
                        "seed_row": row_index,
                        "alive_submass_test": U,
                        "native_alive_mass": alive_mass,
                        "reference_submass_test": u,
                        "reference_alive_mass": mass,
                        "native_physical_alive_test": physical,
                        "reference_physical_alive_test": reference_physical,
                        "lossless_operands": relative,
                    },
                    "scope": "Exact pathwise ratio inequality using the independent finite Monte Carlo reference law as an explicit comparison probability. Every physical alive test uses its own alive mass and ignores dead physical coordinates; zero-alive outputs use auxiliary test zero. The unknown population-map means retain the separate Monte Carlo uncertainty, so this pathwise algebra is not a certificate of a global alive floor or stationary law.",
                })
    return records


def external_evidence(inventory, paths, memory):
    """Import independent comparisons with individual expression bindings."""
    expressions = {e["id"]: e for e in inventory["quantitative_expressions"]}
    checks = []
    for path in paths:
        report = json.loads(path.read_text())
        report_sha256 = sha(path)
        if report["source_sha256"] != inventory["source_sha256"]:
            message = "Independent expression evidence has stale source SHA"
            raise ValueError(message)
        is_replacement = any(
            key in report.get("summary", {})
            for key in [
                "native_measurement_replacements",
                "original_parameter_native_measurement_replacements",
            ]
        )
        for row in report["checks"]:
            # Ingredient or replay checks never inherit the expectation bound
            # merely from a shared lemma reference. Only literal matching
            # operands, or sufficient pathwise certificates, receive credit.
            if "prerequisite" in row["scope"]:
                continue
            if is_replacement:
                eligible_suffixes = (
                    "/unconditional-mean",
                    "/all-root-pathwise-certificate",
                    "/conditional-mean",
                    "/conditional-pathwise-certificate",
                    "/fitness-all",
                    "/q-all",
                )
                if not row["id"].endswith(eligible_suffixes):
                    continue
            refs = [expressions[e] for e in row["source_expressions"]]
            prefix = (
                "independent-native-replacement-" if is_replacement else "independent-variation-"
            )
            checks.append({
                **row,
                "id": prefix + row["id"],
                "source_label": refs[0]["source_label"],
                "source_expressions": [
                    {"id": e["id"], "formula": e["formula"], "line": e["source_line"]}
                    for e in refs
                ],
                "input_report": str(path),
                "input_report_sha256": report_sha256,
            })
    if memory:
        report = json.loads(memory.read_text())
        memory_sha256 = sha(memory)
        for number, key in [
            (798, "maximum_inverse_cap_error"),
            (806, "maximum_ou_cancellation_error"),
        ]:
            e = expressions[f"chapter09-expression-{number:04d}"]
            checks.append({
                "id": f"independent-memory-native-identity-{number}",
                "source_label": e["source_label"],
                "source_expressions": [
                    {"id": e["id"], "formula": e["formula"], "line": e["source_line"]}
                ],
                "observed": report["summary"][key],
                "bound": 1e-10,
                "relation": "upper",
                "passed": report["summary"][key] <= 1e-10,
                "inputs": {
                    "lossless_raw_batches": report["raw_batches"],
                    "consumed_native_archives": report["consumed_archives"],
                    "independent_root_comparisons": report["summary"]["comparisons"],
                    "retained_frames": report["summary"]["retained_quadratic_native_frames"],
                },
                "input_report": str(memory),
                "input_report_sha256": memory_sha256,
                "scope": report["scope"]
                + " This aggregates the maximum separately reconstructed primitive residual over every retained physical coordinate, rather than giving all source expressions credit from a total check count.",
            })
        archive_index = json.loads((memory.parent / "archive-index.json").read_text())
        entries = {e["path"]: e for e in archive_index["entries"]}
        operator = expressions["chapter09-expression-0799"]
        for batch in report["raw_batches"]:
            file = memory.parent / batch
            if sha(file) != entries[batch]["sha256"]:
                message = "Independent native memory raw batch checksum mismatch"
                raise ValueError(message)
            with gzip.open(file, "rt") as handle:
                raw = json.load(handle)
            maximum = 0.0
            for row in raw["rows"]:
                t = np.array(row["frequency"])
                position = np.array(row["physical_output_x"])
                inverse = np.array(row["inverse_physical_cap_velocity"])
                coefficient = (1 - (0.04 / 2) ** 2) / (0.04 / 2)
                reconstructed = math.cos(float(t @ (inverse - coefficient * position)))
                maximum = max(maximum, abs(reconstructed - row["observed"]))
            checks.append({
                "id": "independent-memory-observable-" + batch,
                "source_label": operator["source_label"],
                "source_expressions": [
                    {
                        "id": operator["id"],
                        "formula": operator["formula"],
                        "line": operator["source_line"],
                    }
                ],
                "observed": maximum,
                "bound": 1e-11,
                "relation": "upper",
                "passed": maximum <= 1e-11,
                "inputs": {
                    "raw_batch": batch,
                    "raw_batch_sha256": sha(file),
                    "native_raw_rows": len(raw["rows"]),
                    "h": 0.04,
                    "coefficient": coefficient,
                },
                "scope": "Complete source observable reconstructed for every saved native memory raw row; actual frozen-input gas h=.04. This is a native operator identity, while its conditional expectation has separate Gaussian concentration checks.",
            })
        expression = expressions["chapter09-expression-0801"]
        for row in report["comparisons"]:
            checks.append({
                **row,
                "id": "independent-memory-" + row["id"],
                "source_label": expression["source_label"],
                "source_expressions": [
                    {
                        "id": expression["id"],
                        "formula": expression["formula"],
                        "line": expression["source_line"],
                    }
                ],
                "observed": abs(row["residual"]),
                "bound": row["gaussian_concentration_allowance"],
                "relation": "upper",
                "scope": report["scope"],
                "input_report": str(memory),
                "input_report_sha256": memory_sha256,
                "lossless_raw_batches": report["raw_batches"],
            })
    return checks


def residual_disposition(expression):
    """Keep each unmatched finite estimate visible with its reason."""
    number = int(expression["id"].rsplit("-", 1)[1])
    formula = expression["formula"]
    contracts = {
        24,
        73,
        84,
        91,
        109,
        111,
        116,
        117,
        125,
        132,
        143,
        150,
        245,
        260,
        284,
        285,
        328,
        333,
        365,
        366,
        384,
        395,
        398,
        400,
        419,
        433,
        445,
        446,
        457,
        491,
        503,
        506,
        521,
        524,
        569,
        572,
        578,
        580,
        585,
        591,
        597,
        610,
        623,
        627,
        630,
        638,
        640,
        649,
        673,
        674,
        682,
        689,
        690,
        705,
        716,
        722,
        726,
        742,
        764,
        797,
    }
    if number in {167, 254, 611, 754, 769}:
        return (
            "analytic_hypothesis_or_domain_contract",
            "The exact source formula is an admitted probability/parameter bound or an additional global functional-inequality assumption. It is not inferred from a finite trajectory.",
        )
    if number == 133:
        return (
            "definition_or_exact_proof_expression",
            "This line displays the tail-expectation operand only, without a separate inequality. The complete adjacent uniform-integrability bound is individually compared.",
        )
    if number in contracts or expression.get("expression_role") in {
        "parameter_or_domain_condition",
        "domain_or_operator_contract",
    }:
        return (
            "analytic_hypothesis_or_domain_contract",
            "This exact formula prescribes an admitted parameter/domain or an additional analytic assumption. A finite observation cannot establish its uniform quantifier.",
        )
    specific = {
        209: "Requires a coupled native measurement replacement with identical other innovations and recomputed global normalizers. Independent seed archives cannot identify the counterfactual changed-row count D_r. The exact A_D formula is independently computed, and its surrounding normalization/gate bounds are separately tested.",
        226: "Requires the conditional changed-row square under fixed exceptional outcomes and both measurement arrays. A marginal or unconditional component sample cannot identify this conditional law. Its component-moment and exceptional-count inputs are separately tested.",
        293: "Applies to the separately configured ordered-star collision operator. Current native archives use whole-component Haar collision. Its priority readout, degree moments and exact constants are tested in separate finite fixtures; these archives do not test the ordered-star full-output variance.",
        294: "Ordered-star population reference and full-output bias/MSE require this alternate configured operator. Native Haar one-step comparisons are not assigned to this formula.",
        323: "This brackets the true native QSD eigenvalue alpha_N. It is not estimated from a finite transient run. Positive landing and Gaussian extinction constants are separately evaluated, and the finite-kernel QSD eigenvalue bracket is tested under its own exact law.",
        485: "Uniform collision-velocity envelope over every prepared component is an analytic consequence of center-of-mass and orthogonal rotation bounds; a finite run checks only its visited components.",
        487: "Uniform operator norm of the kinetic noise coefficient is an analytic contract; its explicit isotropic tau_x constant is computed separately.",
        665: "Includes an infinite-N stationary identification limit for unknown native QSD empirical laws. The finite stationary transfer is tested with a supplied exact substochastic kernel, while native one-step datasets do not certify a native QSD.",
        776: "Full actual population-map variation stability requires integrating its entire rooted component and kinetic law across input laws. The explicit L_step constants, input distance-law perturbation and pre-collision tagged-source/jitter variation are tested separately. Finite empirical Gaussian samples cannot estimate full total variation of continuous laws.",
        789: "Global maximal-coupling mismatch probability for the entire unbounded rooted component is an analytic coupling estimate. Its edge, intensity and moment constants are separately evaluated, but finite sampled root components do not expose this counterfactual coupling.",
    }
    if number in specific:
        return "finite_estimate_retained_analytic", specific[number]
    if any(
        token in formula
        for token in ["\\Rightarrow", "\\longrightarrow", "\\to0", "\\to 0", "\\lim", "\\infty"]
    ):
        return (
            "analytic_limit_or_population_map_obligation",
            "Limit or uniformly quantified law statement, retained with complete source assumptions. No finite dataset is assigned an infinite-law certificate.",
        )
    if expression.get("expression_role") == "quantitative_bound" or (
        ("\\le" in formula or "\\ge" in formula)
        and any(
            token in formula
            for token in ["\\mathbb E", "\\Pr", "\\operatorname{Var}", "|L_N", "D_i\\le"]
        )
    ):
        return (
            "finite_estimate_retained_analytic",
            "This exact finite estimate remains covered by the source proof rather than a matched operand experiment. It is explicitly listed in the residual table; no nearby check gives it numerical credit.",
        )
    return (
        "definition_or_exact_proof_expression",
        "Source-bound definition, algebraic intermediate, or measure/map construction. It has no independent numerical comparison; see the exact formula and its role rather than a parent theorem pass.",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--constants", type=Path)
    parser.add_argument("--resonance", type=Path)
    parser.add_argument("--external-evidence", type=Path, action="append", default=[])
    parser.add_argument("--memory", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = json.loads(INVENTORY.read_text())
    if inventory["source_sha256"] != sha(SOURCE):
        message = "Chapter09 source inventory is stale"
        raise ValueError(message)
    checks = algebra(inventory)
    checks += residual_checks(inventory, args.resonance)
    checks += external_evidence(inventory, args.external_evidence, args.memory)
    if args.constants:
        checks += constant_comparisons(inventory, args.constants)
        checks += native_evidence(inventory, args.constants)
    if len({c["id"] for c in checks}) != len(checks):
        message = "Every source-ledger assertion must have a unique check ID"
        raise ValueError(message)
    by_expression = {}
    for c in checks:
        for e in c["source_expressions"]:
            by_expression.setdefault(e["id"], []).append(c["id"])
    checks_by_id = {c["id"]: c for c in checks}
    ledger = []
    for e in inventory["quantitative_expressions"]:
        refs = by_expression.get(e["id"], [])
        if refs:
            kinds = {checks_by_id[r].get("validation_kind") for r in refs}
            if "conservative_full_law_certificate" in kinds:
                status = "scoped_conservative_full_law_certificate"
                scope = "Zero-gate complete-kernel coupling certificate only. Its operand upper-bounds full law variation via an exact maximal root coupling; it is not an empirical full-TV estimate or an active selected-map certificate."
            elif (
                "finite_prefix_only_analytic_limit_retained" in kinds
                or "\\longrightarrow" in e["formula"]
            ):
                status = "finite_comparison_with_analytic_limit_clause"
                scope = "The recorded finite inequality/operand is compared under its saved finite model. The limit clause and applicability to unknown native stationary laws remain analytic, with no numerical limit certificate."
            else:
                status = "exact_finite_law_comparison"
                scope = SCOPE
        else:
            status, scope = residual_disposition(e)
        ledger.append({
            **e,
            "validation_status": status,
            "check_ids": refs,
            "evidence_scope": scope,
        })
    native_identity_rechecks = []
    estimate_keys = set()
    all_keys = set()
    for c in checks:
        payload = {
            "source_expressions": [e["id"] for e in c["source_expressions"]],
            "source_line_only": c.get("source_line_only_evidence", []),
            "observed": c.get("observed", c.get("lhs")),
            "bound": c.get("bound", c.get("rhs")),
            "relation": c.get("relation"),
            "inputs": c.get("inputs", c.get("operands", {})),
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        all_keys.add(key)
        is_identity = c["id"].startswith((
            "independent-memory-native-identity-",
            "independent-memory-observable-",
        )) or (c["id"].startswith("residual-") and "resonance-d" in c["id"])
        if is_identity:
            native_identity_rechecks.append(c["id"])
        else:
            estimate_keys.add(key)
    report = {
        "chapter": 9,
        "source_sha256": sha(SOURCE),
        "provenance": {
            "source_path": str(SOURCE),
            "inventory_path": str(INVENTORY),
            "inventory_sha256": sha(INVENTORY),
            "runner_sha256": sha(Path(__file__)),
            "residual_runner_sha256": sha(
                Path(__file__).with_name("chapter09_residual_estimates.py")
            ),
            "constants_report": str(args.constants) if args.constants else None,
            "constants_report_sha256": sha(args.constants) if args.constants else None,
            "memory_report": str(args.memory) if args.memory else None,
            "memory_report_sha256": sha(args.memory) if args.memory else None,
            "resonance_report": str(args.resonance / "report.json") if args.resonance else None,
            "resonance_report_sha256": sha(args.resonance / "report.json")
            if args.resonance
            else None,
        },
        "checks": checks,
        "expression_ledger": ledger,
        "summary": {
            "formal_items": len(inventory["formal_items"]),
            "source_expressions": len(ledger),
            "checks": len(checks),
            "failed": sum(not c["passed"] for c in checks),
            "expressions_with_exact_comparisons": len(by_expression),
            "expression_dispositions": dict(Counter(e["validation_status"] for e in ledger)),
            "new_native_steps": 0,
            "unique_exact_assertions": len(all_keys),
            "unique_estimate_assertions_excluding_native_identity_rechecks": len(estimate_keys),
            "native_identity_recheck_records": len(native_identity_rechecks),
            "finite_estimates_retained_analytic": sum(
                e["validation_status"] == "finite_estimate_retained_analytic" for e in ledger
            ),
            "checks_with_exact_source_line_only": sum(
                bool(c.get("source_line_only_evidence")) for c in checks
            ),
        },
        "scope": SCOPE,
        "formal_items": inventory["formal_items"],
        "native_identity_recheck_ids": native_identity_rechecks,
        "external_evidence": [
            {
                "path": str(path),
                "sha256": sha(path),
                "scope": json.loads(path.read_text()).get(
                    "scope",
                    "The exact scopes are retained in each imported comparison; this report provides public-stage probes, not complete engine updates.",
                ),
                "native_primitive_summary": json.loads(path.read_text()).get("summary", {}),
                "import_policy": "Source-matching numerical operands only; normalizer, replay, permutation and graph ingredients retain prerequisite status. Primitive native comparison counts are not added to this source-ledger count.",
            }
            for path in args.external_evidence
        ],
        "count_scope": "Source-ledger assertions are reported separately from upstream native primitive comparisons. Retained identity maxima and raw primitive checks are aliases of the same data, never additive new experiments.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "expression-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    residuals = [
        e for e in ledger if e["validation_status"] == "finite_estimate_retained_analytic"
    ]
    (args.output / "finite-estimate-residuals.json").write_text(
        json.dumps(residuals, indent=2) + "\n"
    )
    residual_lines = [
        "# Explicit finite estimates without matched numerical operands",
        "",
        "| Expression | Exact source formula | Why empirical credit is withheld |",
        "|---|---|---|",
    ]
    for e in residuals:
        formula = e["formula"].replace("\n", " ").replace("|", "\\|")
        residual_lines.append(
            f"| {e['id']} (line {e['source_line']}) | `{formula}` | {e['evidence_scope']} |"
        )
    (args.output / "finite-estimate-residuals.md").write_text("\n".join(residual_lines) + "\n")
    (args.output / "source.md").write_bytes(SOURCE.read_bytes())
    (args.output / "runner.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "residual_runner.py").write_bytes(
        Path(__file__).with_name("chapter09_residual_estimates.py").read_bytes()
    )
    print(json.dumps(report["summary"]))
    if report["summary"]["failed"]:
        message = "A source-scoped estimate comparison failed"
        raise SystemExit(message)


if __name__ == "__main__":
    main()
