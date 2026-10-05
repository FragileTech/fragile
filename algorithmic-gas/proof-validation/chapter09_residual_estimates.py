"""Source-specific finite operands for the remaining Chapter 9 estimates.

The finite substochastic and ordered-forest fixtures are explicit mathematical
models. They do not stand in for a stationary law of the native gas.
"""

from collections import defaultdict
import gzip
import hashlib
import itertools
import json
import math
import operator

import numpy as np
from scipy.optimize import linprog
from scipy.special import gammaln, ndtr
from scipy.stats import chi2, ncx2


SCOPE = (
    "Exact finite supplied model; the input kernel, law and constants are saved. "
    "This does not establish the analytic hypotheses for an unknown native QSD."
)


def residual_checks(inventory, resonance=None):
    expressions = {e["id"]: e for e in inventory["quantitative_expressions"]}
    checks = []

    def add(number, needle, name, observed, bound=0.0, inputs=None, equal=False, scope=SCOPE):
        e = expressions[f"chapter09-expression-{number:04}"]
        if needle not in e["formula"]:
            raise ValueError(f"Source binding changed for {e['id']}: {needle}")
        observed, bound = float(observed), float(bound)
        tolerance = 3e-11 * max(1, abs(observed), abs(bound))
        passed = abs(observed - bound) <= tolerance if equal else observed <= bound + tolerance
        checks.append({
            "id": f"residual-{number:04}-{name}",
            "source_label": e["source_label"],
            "source_expressions": [
                {"id": e["id"], "formula": e["formula"], "line": e["source_line"]}
            ],
            "source_quote": e["formula"],
            "observed": observed,
            "bound": bound,
            "relation": "equal" if equal else "upper",
            "passed": passed,
            "tolerance": tolerance,
            "inputs": inputs or {},
            "scope": scope,
        })

    # A positive substochastic kernel with a separately known eigenpair.
    P = np.array([[0.8, 0.2], [0.3, 0.7]])
    eigen = np.array([1.0, 1.2])
    alpha = 0.8
    Q = alpha * eigen[:, None] * P / eigen[None, :]
    nu = np.array([0.6, 0.4]) / eigen
    nu /= nu.sum()
    q = Q.sum(axis=1)
    delta = float((1 - q).max())
    fixture = {"Q": Q.tolist(), "nu": nu.tolist(), "alpha": alpha, "delta": delta}
    f = np.array([-0.8, 0.7])
    add(
        37,
        "Q_Nf",
        "kernel-row-continuity",
        abs(float((Q @ f)[0] - (Q @ f)[1])),
        float(np.abs(Q[0] - Q[1]).sum()),
        fixture,
    )
    add(
        51,
        "Q_N(k,A)",
        "finite-minorization",
        float(np.max(2 * Q.min() * 0.5 - Q)),
        0,
        {**fixture, "epsilon": float(2 * Q.min()), "theta": [0.5, 0.5]},
    )
    V = np.array([1.0, 2.0])
    r = 0.7
    B = float(np.max(Q @ V - r * V))
    add(
        85,
        "Q_NV_N",
        "finite-drift",
        float(np.max(Q @ V - r * V)),
        B,
        {**fixture, "V": V.tolist(), "r": r, "B": B, "W": V.tolist(), "a": 1.0, "b": 0.0},
    )
    add(
        92,
        "B_N/(",
        "qsd-Lyapunov-transfer",
        float(nu @ V),
        B / (alpha - r),
        {**fixture, "V": V.tolist(), "r": r, "B": B},
    )
    add(364, "1-\\alpha_N", "qsd-lower-hazard", float((1 - q).min()), 1 - alpha, fixture)
    add(
        482,
        "a_0",
        "eigenvalue-bracket",
        max(float(q.min()) - alpha, alpha - float(q.max())),
        0,
        {**fixture, "a0": float(q.min()), "qD_power_N": float((1 - q).min())},
    )
    add(
        733,
        "B_W/",
        "qsd-moment-transfer",
        float(nu @ V),
        float((Q @ V).max()) / alpha,
        {**fixture, "W": V.tolist(), "B_W": float((Q @ V).max())},
    )
    # Choose the good set and verify its input hypothesis with the entire kernel.
    bad = np.array([False, True])
    d_bad = float((Q @ bad).max())
    add(
        731,
        "\\alpha_N\\nu_N",
        "good-set-stationary-equation",
        alpha * float(nu @ bad),
        float(nu @ Q @ bad),
        {**fixture, "bad": bad.tolist(), "delta_N": d_bad},
        equal=True,
    )
    for number in [463, 711, 721]:
        add(
            number,
            "\\nu_N(G_N^c)",
            "good-set-stationary-bound",
            float(nu @ bad),
            d_bad,
            {
                **fixture,
                "bad": bad.tolist(),
                "uniform_outgoing_bad_mass": d_bad,
                "extinction_subset_bad_required": False,
                "scope": "This specific kernel happens to satisfy the conclusion; it is not a fixture for the extinction-subset proof.",
            },
        )
    for n in [1, 2, 4, 8, 16]:
        power = np.linalg.matrix_power(Q, n)
        add(
            44,
            "Q_N^n",
            f"survival-lower-{n}",
            float(q.min()) ** n,
            float((power @ np.ones(2)).min()),
            {**fixture, "n": n, "a_N": float(q.min())},
        )
        add(
            312,
            "T_\\dagger",
            f"stationary-union-{n}",
            1 - alpha**n,
            n * delta,
            {**fixture, "n": n},
        )
        add(
            313,
            "1-\\alpha_N^n",
            f"stationary-geometric-{n}",
            1 - alpha**n,
            n * (1 - alpha),
            {**fixture, "n": n},
        )
        eta = np.array([0.4, 0.6]) @ power
        eta /= eta.sum()
        add(
            461,
            "\\eta_n(G_N^c)",
            f"filtered-good-set-{n}",
            float(eta @ bad),
            d_bad,
            {**fixture, "eta_n": eta.tolist(), "bad": bad.tolist(), "delta_N": d_bad},
        )
        hazard = 1 - float(np.array([0.4, 0.6]) @ power @ np.ones(2))
        at_risk = sum(
            float(np.array([0.4, 0.6]) @ np.linalg.matrix_power(Q, j) @ (1 - q)) for j in range(n)
        )
        add(600, "H_{N,T}=", f"at-risk-sum-{n}", hazard, at_risk, {**fixture, "T": n}, equal=True)
        add(
            601,
            "H_{N,T}=",
            f"whole-path-TV-{n}",
            hazard,
            at_risk,
            {**fixture, "T": n, "TV_half_norm": True},
            equal=True,
        )
        upper_sum = sum(
            float(np.array([0.4, 0.6]) @ np.linalg.matrix_power(Q, j) @ np.ones(2)) * delta
            for j in range(n)
        )
        add(
            602,
            "H_{N,T}\\le",
            f"at-risk-upper-{n}",
            hazard,
            upper_sum,
            {**fixture, "T": n, "b0_all_states": delta},
        )
        add(
            604,
            "1-(1-\\bar b_N)^T",
            f"whole-path-hazard-{n}",
            hazard,
            1 - (1 - delta) ** n,
            {**fixture, "T": n, "bar_b_N": delta},
        )
        add(
            606,
            "1-(1-\\delta_N)^T",
            f"whole-path-uniform-{n}",
            hazard,
            1 - (1 - delta) ** n,
            {**fixture, "T": n},
        )
        # Conditioned row mixture: first compare Y with its full-output mean,
        # then account for extinction and the previous survival tilt explicitly.
    # A genuine bound on conditional drift fluctuation, keeping survival tilt.
    h = f
    conditional = (Q @ h) / q
    g = 0.8 * conditional
    M = 0.8

    def var(v):
        return float(nu @ (v * v) - (nu @ v) ** 2)

    A = 2 * float((Q @ (h * h) - (Q @ h) ** 2).max())
    bN = float(nu @ np.abs(conditional - g))
    add(
        703,
        "b_N\\le",
        "finite-comparison-drift-bias",
        bN,
        2 * M * 1 / math.sqrt(2) + 4 * M * delta,
        {
            **fixture,
            "N": 2,
            "H": h.tolist(),
            "r_N": conditional.tolist(),
            "comparison_map": g.tolist(),
            "B_star": 1.0,
        },
    )
    add(
        704,
        "\\operatorname{Var}",
        "finite-comparison-variance-budget",
        abs(var(h) - var(g)),
        A / (2 * alpha) + 7 * M * M * delta / alpha + 4 * M * bN,
        {
            **fixture,
            "N": 2,
            "H": h.tolist(),
            "r_N": conditional.tolist(),
            "comparison_map": g.tolist(),
            "A_phi": A,
            "M": M,
            "b_N": bN,
        },
        scope=SCOPE
        + " The comparison map is the saved finite function g; it is not the native full population map.",
    )

    a0 = 0.75
    B_star = 1.0
    budget = A / (2 * a0) + 8 * M * M * B_star / math.sqrt(2)
    budget += M * M * (7 / a0 + 16) * delta
    add(
        706,
        "\\frac{A_\\varphi}{Na_0}",
        "closed-three-term-variance-budget",
        abs(var(h) - var(g)),
        budget,
        {
            **fixture,
            "N": 2,
            "a0": a0,
            "M": M,
            "B_star": B_star,
            "A_phi": A,
            "actual_b_N": bN,
            "H": h.tolist(),
            "comparison_map": g.tolist(),
            "term_conditional_variance": A / (2 * a0),
            "term_population_bias": 8 * M * M * B_star / math.sqrt(2),
            "term_survival": M * M * (7 / a0 + 16) * delta,
        },
    )

    # A supplied independent single-parent ordered forest. Each row has a
    # categorical choice with every admissible edge probability C/N. The full
    # sampled ancestor chain is retained as the conditioning sigma-field.
    C = 0.8
    for n in [2, 3, 4, 5]:
        chain_keys = []
        chain_lengths = []
        component_sizes = []
        probabilities = []
        options = [[-1, *range(i)] for i in range(n)]
        for parents in itertools.product(*options):
            prob = math.prod(
                1 - i * C / n if parent < 0 else C / n for i, parent in enumerate(parents)
            )
            edges = [(i, parent) for i, parent in enumerate(parents) if parent >= 0]
            chain = [n - 1]
            while parents[chain[-1]] >= 0:
                chain.append(parents[chain[-1]])
            component = {n - 1}
            while True:
                new = (
                    component
                    | {j for i, j in edges if i in component}
                    | {i for i, j in edges if j in component}
                )
                if new == component:
                    break
                component = new
            chain_keys.append(tuple(chain))
            chain_lengths.append(len(chain))
            component_sizes.append(len(component))
            probabilities.append(prob)
        probs = np.array(probabilities)
        lengths = np.array(chain_lengths)
        sizes = np.array(component_sizes)
        for k in range(n):
            add(
                155,
                "C^k/k!",
                f"ancestor-tail-{n}-{k}",
                float(probs @ (lengths >= k + 1)),
                C**k / math.factorial(k),
                {
                    "N": n,
                    "C": C,
                    "k": k,
                    "model": "independent single-parent ordered categorical forest",
                },
            )
        for p in [1, 2, 3, 4]:
            for chain in sorted(set(chain_keys)):
                take = np.array([key == chain for key in chain_keys])
                a = len(chain)
                condition_mass = float(probs[take].sum())
                if not condition_mass:
                    continue
                moment = float(probs[take] @ (sizes[take] ** p) / condition_mass)
                add(
                    159,
                    "\\mid\\text",
                    f"descendant-moment-{n}-p{p}-chain{'-'.join(map(str, chain))}",
                    moment,
                    math.exp((2**p - 1) * C) * a**p,
                    {
                        "N": n,
                        "C": C,
                        "p": p,
                        "chain_size": a,
                        "full_sampled_chain": list(chain),
                        "conditioning": "exact full sampled ancestor chain",
                        "probability_condition": condition_mass,
                        "parent_choice_probabilities": [
                            [1 - i * C / n, *([C / n] * i)] for i in range(n)
                        ],
                    },
                )
            for size in range(1, n + 1):
                add(
                    156,
                    "(s+1)^p",
                    f"scan-increment-{n}-{p}-{size}",
                    (size + 1) ** p - size**p,
                    (2**p - 1) * size ** (p - 1),
                    {"p": p, "s": size},
                )
        for j in range(1, n):
            p = C / n
            EY = (n - j) * p
            EY2 = EY**2 + (n - j) * p * (1 - p)
            add(
                299,
                "\\mathbb EY_j",
                f"incoming-mean-{n}-{j}",
                EY,
                C,
                {"N": n, "C": C, "trials": n - j, "p": p},
            )
            add(
                300,
                "\\mathbb EY_j^2",
                f"incoming-second-{n}-{j}",
                EY2,
                C + C * C,
                {"N": n, "C": C, "trials": n - j, "p": p},
            )
            K = 27 + 30 * C + 6 * C * C
            add(
                303,
                "K_{\\rm ord}(C)",
                f"ordered-degree-influence-{n}-{j}",
                3 * (1 + 2 * (4 + 4 * EY + EY2)),
                K,
                {
                    "N": n,
                    "C": C,
                    "E_Y": EY,
                    "E_Y2": EY2,
                    "operator": "separate ordered-star, not canonical Haar",
                },
            )

    # Independent innovations: exact replacement differences and Efron--Stein.
    for n in [2, 4, 6]:
        rows = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        values = rows.mean(axis=1)

        def physical_outputs(row):
            return row.copy()

        influences = []
        for j in range(n):
            support_squared = []
            for row in rows:
                for fresh in [-1.0, 1.0]:
                    changed = row.copy()
                    changed[j] = fresh
                    support_size = int(
                        np.count_nonzero(physical_outputs(row) != physical_outputs(changed))
                    )
                    support_squared.append(support_size * support_size)
            influences.append(float(np.mean(support_squared)))
        add(
            204,
            "\\sum_j",
            f"all-innovation-physical-support-{n}",
            sum(influences),
            0.5 * n,
            {
                "N": n,
                "exact_independent_Bernoulli_innovations": True,
                "physical_output_map": "each row retains its own fresh mark; permutation equivariant",
                "D_j_definition": "count of changed physical output coordinates",
                "bounded_test": "identity on the physical output interval [-1,1]",
                "E_D_j_squared": influences,
                "C": 0.5,
            },
        )
        add(
            239,
            "\\operatorname{Var}",
            f"variance-secondmoment-{n}",
            float(values.var()),
            float(np.mean(values**2)),
            {"N": n, "all_outputs": values.tolist()},
        )
    for c in [0.1, 0.8, 2.0]:
        add(
            240,
            "1/(ce)",
            f"exponential-influence-{c}",
            (1 / c) * math.exp(-c * (1 / c)),
            1 / (c * math.e),
            {"c": c, "maximizer": 1 / c},
            equal=True,
        )

    # Sampled measurement moments with no fit: independent two-point separation
    # marks have known exact first/second moments. Native limits are distinct.
    for n in [2, 4, 8]:
        S, sigma, m, DD = 2.0, 0.2, 0.5, 2.0
        values = np.array(list(itertools.product([0.2, 1.9], repeat=n)))
        mu = 1.05
        second = (0.2**2 + 1.9**2) / 2
        d1 = values.mean(axis=1) - mu
        d2 = (values**2).mean(axis=1) - second
        dv = ((values**2).mean(axis=1) - values.mean(axis=1) ** 2) - (second - mu**2)
        T = abs(d1) / sigma + S * abs(dv) / (2 * sigma**3)
        AT = (1 / m + DD * DD) * (2 * S * S / sigma**2 + 5 * S**6 / sigma**6)
        add(
            250,
            "\\mathbb E\\Delta_1^2",
            f"sampled-mark-two-moments-{n}",
            max(float(np.mean(d1 * d1)) / (S * S), float(np.mean(d2 * d2)) / (S**4)),
            (1 / m + DD * DD) / n,
            {
                "N": n,
                "S_star": S,
                "m_star": m,
                "D_D": DD,
                "E_delta1_squared": float(np.mean(d1 * d1)),
                "E_delta2_squared": float(np.mean(d2 * d2)),
                "mu": mu,
                "second": second,
            },
        )
        add(
            251,
            "|\\Delta_v|",
            f"sampled-variance-difference-{n}",
            float(np.max(abs(dv) - abs(d2) - 2 * S * abs(d1))),
            0,
            {
                "N": n,
                "delta1": d1.tolist(),
                "delta2": d2.tolist(),
                "delta_v": dv.tolist(),
                "S_star": S,
            },
        )
        add(
            252,
            "A_T/N",
            f"measurement-T-secondmoment-{n}",
            float(np.mean(T * T)),
            AT / n,
            {
                "N": n,
                "S_star": S,
                "sigma_s": sigma,
                "m_star": m,
                "D_D": DD,
                "A_T": AT,
                "T": T.tolist(),
            },
        )
    for n in [2, 8, 32, 128, 512, 2048]:
        K = math.floor(n ** (1 / 6))
        add(269, "K\\geq", f"truncation-radius-floor-{n}", n ** (1 / 6) / 2, K, {"N": n, "K": K})

    # Exact finite killed-path model. The whole path law has an explicit first
    # death distribution, so conditioning TV is evaluated with all path atoms.
    for hazard in [0.001, 0.2, 0.8]:
        for T in [1, 2, 4, 8, 32]:
            survival = (1 - hazard) ** T
            death = 1 - survival
            paths = np.array([hazard * (1 - hazard) ** j for j in range(T)] + [survival])
            conditioned = np.zeros(T + 1)
            conditioned[-1] = 1
            TV = float(abs(paths - conditioned).sum() / 2)
            x = {
                "hazard": hazard,
                "T": T,
                "all_path_atoms": paths.tolist(),
                "own_survival": survival,
            }
            add(
                322,
                "\\Pr(\\tau_N>n)",
                f"geometric-extinction-{hazard}-{T}",
                survival,
                math.exp(-T * hazard),
                {**x, "E_tau": 1 / hazard, "E_alive_mass": survival},
            )
            add(
                361,
                "1-b_N",
                f"geometric-single-step-{hazard}-{T}",
                1 - hazard,
                math.exp(-hazard),
                x,
            )
            add(
                438,
                "\\zeta_k",
                f"stopped-first-hit-{hazard}-{T}",
                death,
                float(paths[:-1].sum()),
                x,
                equal=True,
            )
            add(440, "T\\bar b", f"stopped-union-bound-{hazard}-{T}", death, T * hazard, x)
            add(
                441,
                "(1-\\bar b)^T",
                f"stopped-survival-{hazard}-{T}",
                (1 - hazard) ** T,
                survival,
                x,
                equal=True,
            )
            add(447, "\\mathsf P_T-", f"stopped-path-TV-{hazard}-{T}", TV, death, x, equal=True)
            add(
                396,
                "\\delta_N^{-1}",
                f"survival-lifetime-bracket-{hazard}-{T}",
                max(
                    (1 - 0.9) ** T - survival,
                    survival - (1 - 0.0001) ** T,
                    1 / 0.9 - 1 / hazard,
                    1 / hazard - 1 / 0.0001,
                ),
                0,
                {**x, "delta_N": 0.9, "q_tau_power_N": 0.0001, "E_tau": 1 / hazard},
            )
            # bounded errors on each complete path atom, no assumed independence.
            errors = np.linspace(0, 1, T + 1)
            E = float(paths @ errors)
            add(
                625,
                "E_{N,T}+H",
                f"survival-conditioned-error-{hazard}-{T}",
                float(conditioned @ errors),
                min(1, E + death),
                {**x, "path_errors": errors.tolist(), "E_N_T": E, "H_N_T": death},
            )
            add(
                654,
                "H_{N,n}\\le",
                f"horizon-monotonicity-{hazard}-{T}",
                1 - (1 - hazard) ** (T // 2),
                death,
                x,
            )
        a0 = 0.2
        sep = math.ceil(math.log(2 / a0) / (-math.log1p(-hazard)))
        safe_sep = math.ceil(math.log(2 / a0) / hazard)
        for number, n in [(344, sep), (345, safe_sep), (368, sep)]:
            add(
                number,
                "a_0",
                f"separation-time-{hazard}",
                (1 - hazard) ** n,
                a0 / 2,
                {"qD_power_N": hazard, "a0": a0, "n": n},
            )
        for n in [1, 4, 32]:
            # Alive mass is 1 on survivors and 0 on death; target population .2.
            s = (1 - hazard) ** n
            error = s * abs(1 - a0) + (1 - s) * a0
            for number in [343, 367]:
                add(
                    number,
                    "a_0",
                    f"unconditioned-population-error-{hazard}-{n}",
                    a0 - s,
                    error,
                    {"n": n, "a0": a0, "alive_law": [1 - s, s], "target_mass": a0},
                )

    # Gaussian terminal hazards and safe-center recovery, retaining dimension.
    for d in [1, 2, 4]:
        for tau in [0.2, 0.7, 1.0]:
            inside = float((2 * ndtr(1 / tau) - 1) ** d)
            qD = 1 - inside
            for n in [2, 4, 8]:
                add(
                    318,
                    "p_D=",
                    f"box-death-definition-{d}-{tau}-{n}",
                    qD**n,
                    (1 - inside) ** n,
                    {"d": d, "widths": [2.0] * d, "s": tau, "pD": inside, "qD": qD, "N": n},
                    equal=True,
                )
                probabilities = [
                    (float(ndtr((1 - z) / tau) - ndtr((-1 - z) / tau))) ** d for z in [0.7] * n
                ]
                hazard = math.prod(1 - p for p in probabilities)
                for number in [357, 414]:
                    add(
                        number,
                        "q_",
                        f"terminal-product-lower-{d}-{tau}-{n}",
                        qD**n,
                        hazard,
                        {
                            "d": d,
                            "tau": tau,
                            "N": n,
                            "centers": [[0.7] * d] * n,
                            "per_row_alive_probabilities": probabilities,
                        },
                    )
                epsilon = min(1, 2 * d * float(ndtr(-0.2 / tau)))
                add(
                    407,
                    "1-P_{D,\\tau}",
                    f"safe-center-tail-{d}-{tau}-{n}",
                    1 - probabilities[0],
                    epsilon,
                    {"d": d, "r": 0.2, "tau": tau, "center": [0.7] * d, "epsilon_r": epsilon},
                )
                # Two possible preparation outcomes, including an unsafe state.
                badprep = 0.1
                fullhazard = badprep * 0.9 + (1 - badprep) * hazard
                add(
                    404,
                    "\\Pr_S^{\\rm prep}",
                    f"safe-preparation-mixture-{d}-{tau}-{n}",
                    fullhazard,
                    min(1, badprep + epsilon**n),
                    {
                        "N": n,
                        "rho": 1.0,
                        "eta_preparation": badprep,
                        "epsilon_r": epsilon,
                        "good_preparation_hazard": hazard,
                        "bad_preparation_hazard": 0.9,
                        "delta_N": 1.0,
                    },
                )
                count = np.array([1.0])
                for p in probabilities:
                    count = np.convolve(count, [1 - p, p])
                for k in [1, n]:
                    binomial = sum(
                        math.comb(n, j) * (1 - epsilon) ** j * epsilon ** (n - j) for j in range(k)
                    )
                    beta_k = min(1, badprep + binomial)
                    beta_0 = min(1, badprep + epsilon**n)
                    U = badprep + (1 - badprep) * float(count[:k].sum())
                    e = badprep * 0.9 + (1 - badprep) * float(count[0])
                    inputs = {
                        "N": n,
                        "d": d,
                        "tau": tau,
                        "g": n,
                        "k": k,
                        "eta_preparation": badprep,
                        "epsilon_r": epsilon,
                        "binomial_tail": binomial,
                        "beta_k": beta_k,
                        "beta_0": beta_0,
                        "U": U,
                        "e": e,
                    }
                    add(
                        563,
                        "u_k(S)",
                        f"safe-noise-filter-definitions-{d}-{tau}-{n}-{k}",
                        beta_k,
                        min(1, badprep + binomial),
                        inputs,
                        equal=True,
                    )
                    add(
                        564,
                        "0\\le e_n",
                        f"safe-noise-filter-order-{d}-{tau}-{n}-{k}",
                        max(-e, e - beta_0, beta_0 - beta_k, beta_k - 1),
                        0,
                        inputs,
                    )
                    for number in [582, 590]:
                        add(
                            number,
                            "e_n\\le",
                            f"extinction-filter-{d}-{tau}-{n}-{k}",
                            e,
                            beta_0,
                            inputs,
                        )
                    add(581, "U_n\\le", f"bad-count-filter-{d}-{tau}-{n}-{k}", U, beta_k, inputs)
                    add(
                        583,
                        "(\\beta_n^k-e_n)",
                        f"conditional-filter-algebra-{d}-{tau}-{n}-{k}",
                        (beta_k - e) / (1 - e),
                        beta_k,
                        inputs,
                    )
                    add(
                        588,
                        "\\eta_n(G_N^c)",
                        f"conditional-bad-count-{d}-{tau}-{n}-{k}",
                        (U - e) / (1 - e),
                        beta_k,
                        inputs,
                    )
                    for number in [422, 423]:
                        needle = "\\eta_{N,g,r}" if number == 422 else "B_{g,k}"
                        add(
                            number,
                            needle,
                            f"preparation-definition-{d}-{tau}-{n}-{k}",
                            badprep if number == 422 else binomial,
                            0.1
                            if number == 422
                            else float(
                                sum(
                                    math.comb(n, j) * (1 - epsilon) ** j * epsilon ** (n - j)
                                    for j in range(k)
                                )
                            ),
                            inputs,
                            equal=True,
                        )
            for gamma in [0.0, 1.0]:
                h = 0.1
                c = h / 2
                sh2 = -math.expm1(-2 * gamma * h) / (2 * gamma) if gamma else h
                # Independently integrate the exponential OU variance.
                independent = (
                    (math.exp(-2 * gamma * h) * math.expm1(2 * gamma * h) / (2 * gamma))
                    if gamma
                    else h
                )
                add(
                    175 if gamma else 176,
                    "s_h^2=",
                    f"OU-variance-{d}-{tau}-{gamma}",
                    sh2,
                    independent,
                    {"gamma": gamma, "h": h},
                    equal=True,
                )
                add(
                    388,
                    "q^2=",
                    f"combined-Gaussian-variance-{d}-{tau}-{gamma}",
                    c * c * sh2 + tau * tau,
                    c * c * independent + tau * tau,
                    {"gamma": gamma, "h": h, "sigma_v": 1.0, "s": tau, "c0": c},
                    equal=True,
                )

    # Explicit landing/minorization constant: compare the volume*density lower
    # bound against a separately integrated noncentral chi-square ball mass.
    for d in [1, 2, 4]:
        h = 0.1
        gamma = 1.0
        cap = 0.2
        restitution = 0.5
        RD = 1.0
        J = 0.1
        G = 1.0
        LU = 2.0
        BU = 0.2
        ch = math.exp(-gamma * h)
        sh = math.sqrt(-math.expm1(-2 * gamma * h) / (2 * gamma))
        sigmaJ = 0.05
        Bnoise = 0.2
        s = 0.8
        r0 = 0.1
        xD = 0.0
        W = (1 + 2 * restitution) * cap
        FJ = LU * (RD + J) + BU
        L = RD + J + h / 2 * (1 + ch) * (W + h / 2 * FJ) + h / 2 * sh * Bnoise * G
        pJ = float(chi2.cdf((J / sigmaJ) ** 2, d))
        pG = float(chi2.cdf(G * G, d))
        ballvolume = math.exp(d / 2 * math.log(math.pi) + d * math.log(r0) - gammaln(1 + d / 2))
        a0 = (
            pJ
            * pG
            * ballvolume
            * (2 * math.pi * s * s) ** (-d / 2)
            * math.exp(-((L + abs(xD) + r0) ** 2) / (2 * s * s))
        )
        exact = pJ * pG * float(ncx2.cdf((r0 / s) ** 2, d, (L / s) ** 2))
        add(
            332,
            "a_0=",
            f"Gaussian-landing-lower-d{d}",
            a0,
            exact,
            {
                "d": d,
                "h": h,
                "gamma": gamma,
                "V": cap,
                "alpha": restitution,
                "RD": RD,
                "J": J,
                "G": G,
                "LU": LU,
                "BU": BU,
                "sigmaJ": sigmaJ,
                "Bnorm": Bnoise,
                "s": s,
                "r0": r0,
                "xD": xD,
                "c_h": ch,
                "s_h": sh,
                "W": W,
                "F_J": FJ,
                "L": L,
                "pJ": pJ,
                "pG": pG,
                "ball_volume": ballvolume,
                "a0": a0,
                "independent_noncentral_ball_mass": exact,
            },
        )
        for profile in ["quadratic", "rastrigin"]:
            for radius in [0.0, 0.3, 4.0, 20.0]:
                x = np.repeat(radius / math.sqrt(d), d)
                force = (
                    2 * x
                    if profile == "quadratic"
                    else 2 * x + 20 * math.pi * np.sin(2 * math.pi * x)
                )
                BU_profile = 0.0 if profile == "quadratic" else 20 * math.pi * math.sqrt(d)
                for number in [342, 476]:
                    add(
                        number,
                        "L_U",
                        f"force-growth-{d}-{profile}-{radius}",
                        float(np.linalg.norm(force)),
                        2 * float(np.linalg.norm(x)) + BU_profile,
                        {
                            "d": d,
                            "profile": profile,
                            "x": x.tolist(),
                            "actual_force": force.tolist(),
                            "LU": 2.0,
                            "BU": BU_profile,
                        },
                    )
        c = h / 2
        b = c * (1 + ch)
        eta = c * c * (1 + ch)
        Ax = 1 + eta * LU
        taux = math.sqrt(c * c * sh * sh * Bnoise * Bnoise + s * s)
        add(
            477,
            "A_x=",
            f"moment-affine-constants-d{d}",
            Ax,
            1 + (h * h / 4) * (1 + math.exp(-gamma * h)) * LU,
            {"h": h, "gamma": gamma, "LU": LU, "c": c, "b": b, "eta": eta, "Ax": Ax, "W": W},
            equal=True,
        )
        add(
            478,
            "\\tau_x^2",
            f"moment-Gaussian-constants-d{d}",
            taux * taux,
            c * c * sh * sh * Bnoise * Bnoise + s * s,
            {
                "c": c,
                "s_h": sh,
                "Bnorm": Bnoise,
                "sigma_x": s / math.sqrt(h),
                "h": h,
                "tau_x": taux,
            },
            equal=True,
        )
        # Choose a valid deterministic preparation and affine confining force.
        mean = (1 + eta * LU) * RD + b * W + eta * BU
        for p in [1, 2, 4, 8]:
            gd = math.exp((p / 2 * math.log(2) + gammaln((d + p) / 2) - gammaln(d / 2)) / p)
            Kr = (Ax * (RD + sigmaJ * gd) + b * W + eta * BU + taux * gd) ** p
            independent = (
                float(ncx2.expect(lambda z: z ** (p / 2), args=(d, (mean / taux) ** 2))) * taux**p
            )
            inp = {
                "d": d,
                "r": p,
                "RD": RD,
                "sigmaJ": sigmaJ,
                "Ax": Ax,
                "b": b,
                "W": W,
                "eta": eta,
                "BU": BU,
                "tau_x": taux,
                "g_d_r": gd,
                "K_r": Kr,
                "mean_norm": mean,
                "independent_noncentral_moment": independent,
                "H_x": (2 * math.pi * s * s) ** (-d / 2),
            }
            add(
                479,
                "K_r=",
                f"position-moment-constant-{d}-{p}",
                Kr,
                (Ax * (RD + sigmaJ * gd) + b * W + eta * BU + taux * gd) ** p,
                inp,
                equal=True,
            )
            add(488, "K_r", f"position-moment-bound-{d}-{p}", independent, Kr, inp)
            # Survival conditioning with a real positive Gaussian ball event.
            survive = float(ncx2.cdf((2 / taux) ** 2, d, (mean / taux) ** 2))
            conditioned = (
                float(
                    ncx2.expect(
                        lambda z: z ** (p / 2),
                        args=(d, (mean / taux) ** 2),
                        lb=0,
                        ub=(2 / taux) ** 2,
                    )
                )
                * taux**p
                / survive
            )
            add(
                481,
                "K_r/a_0",
                f"conditioned-position-moment-{d}-{p}",
                conditioned,
                Kr / survive,
                {**inp, "a0": survive, "event": "|X|<=2", "conditional_denominator": survive},
            )
            if p == 2:
                add(
                    680,
                    "K_2/a_0",
                    f"empirical-conditioned-moment-{d}",
                    conditioned,
                    Kr / survive,
                    {**inp, "a0": survive},
                )
                for N in [2, 8, 32]:
                    radius = 4.0
                    probability = (
                        1 - float(ncx2.cdf((radius / taux) ** 2, d, (mean / taux) ** 2)) ** N
                    )
                    add(
                        494,
                        "NK_2",
                        f"maximum-tail-{d}-{N}",
                        probability,
                        N * Kr / (survive * radius * radius),
                        {
                            **inp,
                            "N": N,
                            "R": radius,
                            "a0": survive,
                            "unconditioned_probability_used": probability,
                        },
                    )
        # Independent density/volume bound in an interior small box.
        halfwidth = 0.05
        center = np.zeros(d)
        center[0] = mean
        probability = math.prod(
            float(ndtr((halfwidth - z) / taux) - ndtr((-halfwidth - z) / taux)) for z in center
        )
        Hx = (2 * math.pi * s * s) ** (-d / 2)
        volume = (2 * halfwidth) ** d
        add(
            684,
            "H_x/a_0",
            f"boundary-volume-density-{d}",
            probability / survive,
            Hx * volume / survive,
            {
                "d": d,
                "tau_x": taux,
                "H_x": Hx,
                "a0": survive,
                "volume": volume,
                "box_half_width": halfwidth,
                "test_set": "small interior box, demonstrating the same density-volume estimate",
            },
        )

    # Cap concavity, resonance recurrence and whole energy after averaging.
    for cap in [0.5, 2.0]:
        for initial in [0.01, 0.3, cap]:
            velocity = initial
            for n in range(1, 33):
                velocity = -cap * velocity / (cap + abs(velocity))
                closed = (-1) ** n * cap * initial / (cap + n * initial)
                add(
                    71,
                    "v_{n+1}=",
                    f"resonance-closed-{cap}-{initial}-{n}",
                    abs(velocity - closed),
                    0,
                    {
                        "V": cap,
                        "v0": initial,
                        "n": n,
                        "actual_iterate": velocity,
                        "closed": closed,
                    },
                )
                add(
                    525,
                    "V/(n+1)",
                    f"resonance-speed-{cap}-{initial}-{n}",
                    abs(velocity),
                    cap / (n + 1),
                    {"V": cap, "v0": initial, "n": n},
                )
                add(
                    526,
                    "V^2",
                    f"resonance-energy-{cap}-{initial}-{n}",
                    velocity * velocity,
                    cap * cap / (n + 1) ** 2,
                    {"V": cap, "v0": initial, "n": n},
                )
            epsilon = initial / 3
            threshold = math.ceil(cap * (1 / epsilon - 1 / initial))
            add(
                512,
                "\\epsilon",
                f"resonance-threshold-{cap}-{initial}",
                cap * initial / (cap + threshold * initial),
                epsilon,
                {"V": cap, "v0": initial, "epsilon": epsilon, "n": threshold},
            )
        for energy in [0.001, 0.1, 1.0, 4.0]:
            function = energy / (1 + math.sqrt(energy) / cap) ** 2
            derivative = (1 + math.sqrt(energy) / cap) ** -3
            curvature = -3 / (2 * cap * math.sqrt(energy)) * (1 + math.sqrt(energy) / cap) ** -4
            step = energy * 1e-3

            def fcap(x):
                return x / (1 + math.sqrt(x) / cap) ** 2

            first = (fcap(energy + step) - fcap(energy - step)) / (2 * step)
            second = (fcap(energy + step) - 2 * function + fcap(energy - step)) / (step * step)
            add(
                522,
                "f''(t)",
                f"cap-derivatives-{cap}-{energy}",
                max(abs(first - derivative), abs(second - curvature)) / max(1, abs(curvature)),
                2e-6,
                {
                    "V": cap,
                    "t": energy,
                    "f": function,
                    "f_prime": derivative,
                    "f_second": curvature,
                    "finite_difference_step": step,
                },
            )
        velocities = np.array([-0.3, 0.1, 0.4]) * cap
        restitution = 0.5
        prepared = restitution * velocities + (1 - restitution) * velocities.mean()
        energy = float(np.mean(velocities**2))
        Eprepared = float(np.mean(prepared**2))
        capped = -prepared / (1 + abs(prepared) / cap)

        def fcap(t):
            return t / (1 + math.sqrt(t) / cap) ** 2

        add(
            520,
            "E_v(S^c)",
            f"component-energy-{cap}",
            Eprepared,
            energy,
            {
                "V": cap,
                "velocities": velocities.tolist(),
                "alpha": restitution,
                "prepared": prepared.tolist(),
            },
        )
        add(
            523,
            "f(E_v(S^c))",
            f"cap-Jensen-energy-{cap}",
            float(np.mean(capped * capped)),
            fcap(Eprepared),
            {
                "V": cap,
                "velocities": velocities.tolist(),
                "prepared": prepared.tolist(),
                "output": capped.tolist(),
                "Eprepared": Eprepared,
                "Einitial": energy,
                "second_inequality_residual": fcap(Eprepared) - fcap(energy),
            },
        )

    # Alive normalization and finite products of an unlabeled empirical cloud.
    for n in [2, 4, 8]:
        z = np.sin(np.arange(n) * 1.7)
        mu = 0.2
        for ell in [1, 2]:
            distinct = [
                math.prod(z[i] for i in idx) for idx in itertools.permutations(range(n), ell)
            ]
            error = abs(float(np.mean(distinct)) - mu**ell)
            add(
                632,
                "\\ell(\\ell-1)",
                f"finite-product-transfer-{n}-{ell}",
                error,
                ell * (ell - 1) / n + ell * abs(float(z.mean()) - mu),
                {"N": n, "ell": ell, "swarm_tests": z.tolist(), "target_test": mu},
            )
        for m in sorted({1, n // 2, n}):
            v = m / n
            U = float(z[:m].sum() / n)
            target_mass = 0.6
            u = 0.1
            physical = U / v
            bound = (abs(U - u) + abs(v - target_mass)) / target_mass
            for number in [661, 637]:
                needle = "|U/v" if number == 661 else "\\rho_N\\psi"
                add(
                    number,
                    needle,
                    f"physical-alive-error-{n}-{m}",
                    abs(physical - u / target_mass),
                    bound,
                    {"N": n, "M": m, "U": U, "v": v, "u": u, "m": target_mass, "a0": target_mass},
                )
            for number in [553, 660]:
                add(
                    number,
                    "|U|",
                    f"physical-submass-{n}-{m}",
                    abs(U),
                    v,
                    {"N": n, "M": m, "U": U, "v": v},
                )
            ell = 2 if m >= 2 else 1
            add(
                663,
                "\\ell(\\ell-1)/M",
                f"alive-sampling-exclusion-{n}-{m}",
                ell * (ell - 1) / m,
                ell * (ell - 1) / (v * n),
                {"N": n, "M": m, "m_star": v, "ell": ell},
            )
    for N in [2, 8, 32, 128]:
        for epsilon in [0.1, 0.7]:
            rho = 0.25
            add(
                646,
                "\\epsilon_r",
                f"exponential-ceiling-{N}-{epsilon}",
                epsilon ** math.ceil(rho * N),
                math.exp(-rho * N * math.log(1 / epsilon)),
                {"N": N, "rho": rho, "epsilon_r": epsilon},
            )
            Ceta = 2.0
            keta = 0.2
            kstar = min(keta, rho * math.log(1 / epsilon))
            c = kstar / 2
            T = math.floor(math.exp(c * N))
            b = min(1, Ceta * math.exp(-keta * N) + epsilon ** math.ceil(rho * N))
            hazard = -math.expm1(T * math.log1p(-b)) if b < 1 else 1.0
            add(
                618,
                "C_\\eta+1",
                f"safe-exponential-window-{N}-{epsilon}",
                hazard,
                (Ceta + 1) * math.exp(-(kstar - c) * N),
                {
                    "N": N,
                    "rho": rho,
                    "epsilon_r": epsilon,
                    "Ceta": Ceta,
                    "keta": keta,
                    "kstar": kstar,
                    "c": c,
                    "T": T,
                    "b": b,
                },
            )

    # Product Poincare fixture with its own Dirichlet form, not a native LSI.
    for n in [2, 4, 6]:
        rows = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        F = rows.mean(axis=1)
        dirichlet = 0.0
        for j in range(n):
            changed = rows.copy()
            changed[:, j] *= -1
            dirichlet += float(np.mean((changed.mean(axis=1) - F) ** 2)) / 4
        add(
            746,
            "C_P",
            f"product-Poincare-{n}",
            float(F.var()),
            dirichlet,
            {
                "N": n,
                "CP": 1.0,
                "Cj": 1.0,
                "Dirichlet": dirichlet,
                "variance": float(F.var()),
                "second_inequality_residual": dirichlet - 1 / n,
                "law": "uniform independent Rademacher product",
            },
        )

    # Exact proof subexpressions, with operands saved separately from the parent
    # inequality. These receive their own checks rather than inherited credit.
    for fixture_index in range(12):
        rho = np.array([1 + fixture_index, 2.0, 3.0, 4.0])
        rho /= rho.sum()
        eta = np.array([4.0, 3.0, 2.0, 1 + fixture_index])
        eta /= eta.sum()
        reward = np.sin(np.arange(4) + fixture_index)
        variation = float(abs(rho - eta).sum())
        inp = {
            "rho": rho.tolist(),
            "eta": eta.tolist(),
            "reward": reward.tolist(),
            "M_R": 1.0,
            "full_variation": variation,
        }
        add(
            119,
            "M_R^2",
            f"reward-second-component-{fixture_index}",
            abs(float((rho - eta) @ (reward * reward))),
            variation,
            inp,
        )
        add(
            120,
            "2M_R^2",
            f"reward-square-mean-component-{fixture_index}",
            abs(float(rho @ reward) ** 2 - float(eta @ reward) ** 2),
            2 * variation,
            inp,
        )
        add(
            718,
            "M^2",
            f"variance-second-component-{fixture_index}",
            abs(float((rho - eta) @ (reward * reward))),
            variation,
            inp,
        )
        add(
            124,
            "\\int f-\\int g",
            f"normalizer-mass-difference-{fixture_index}",
            abs(float(rho.sum() - eta.sum())),
            variation,
            inp,
        )
        # Conditioning a good set changes its original law by precisely its
        # complement mass in the half variation convention.
        good = np.array([True, True, False, True])
        badmass = float(rho @ ~good)
        filtered = rho * good
        filtered /= filtered.sum()
        add(
            593,
            "\\zeta_n-\\eta_n",
            f"good-fraction-TV-{fixture_index}",
            float(abs(filtered - rho).sum() / 2),
            badmass,
            {
                "eta": rho.tolist(),
                "conditioned": filtered.tolist(),
                "good": good.tolist(),
                "delta_N": badmass,
            },
        )
    for radius in [0.0, 0.1, 1.0, 4.0, 20.0]:
        for d in [1, 2, 4]:
            x = np.repeat(radius / math.sqrt(d), d)
            R = -float(np.sum(x * x + 10 * (1 - np.cos(2 * math.pi * x))))
            CR = max(1, 20 * d)
            add(
                139,
                "C_R",
                f"reward-growth-d{d}-r{radius}",
                abs(R),
                CR * (1 + float(x @ x)),
                {
                    "d": d,
                    "x": x.tolist(),
                    "reward": R,
                    "CR": CR,
                    "profile": "quadratic plus bounded Rastrigin perturbation",
                },
            )
            add(
                145,
                "2C_R^2",
                f"reward-square-growth-d{d}-r{radius}",
                R * R,
                2 * CR * CR * (1 + float(x @ x) ** 2),
                {"d": d, "x": x.tolist(), "reward": R, "CR": CR},
            )
    for bound in [0.5, 2.0]:
        for radius in [0.0, 0.1, 1.0, 10.0]:
            x = np.array([radius, -radius / 2])
            u = bound * x / (bound + np.linalg.norm(x))
            inverse = bound * u / (bound - np.linalg.norm(u))
            add(
                18,
                "S_R(x)",
                f"compactification-{bound}-{radius}",
                float(np.linalg.norm(u)),
                bound,
                {"R": bound, "x": x.tolist(), "u": u.tolist()},
            )
            add(
                23,
                "x=Ru",
                f"compactification-inverse-{bound}-{radius}",
                float(np.max(abs(inverse - x))),
                0,
                {"R": bound, "x": x.tolist(), "u": u.tolist(), "inverse": inverse.tolist()},
            )
    for n in [2, 4, 8, 32]:
        y = n**0.5
        second = (y * y) / n
        add(
            820,
            "\\mu_N=",
            f"escaping-law-{n}",
            second,
            1.0,
            {"N": n, "atoms": [0.0, y], "weights": [1 - 1 / n, 1 / n]},
            equal=True,
        )
        add(
            821,
            "W_2^2",
            f"escaping-W2-{n}",
            second,
            1.0,
            {"N": n, "unique_coupling_to_delta0_cost": second},
            equal=True,
        )
        # Efron--Stein uses exact independent replacement, saved full differences.
        if n <= 8:
            rows = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
            values = rows.mean(axis=1)
            influences = []
            for j in range(n):
                delta_values = []
                for row in rows:
                    for fresh in [-1.0, 1.0]:
                        changed = row.copy()
                        changed[j] = fresh
                        delta_values.append(n * abs(float(changed.mean() - row.mean())) / 2)
                influences.append(float(np.mean(np.square(delta_values))))
            add(
                203,
                "\\operatorname{Var}(F",
                f"exact-Efron-Stein-{n}",
                float(values.var()),
                2 / n**2 * sum(influences),
                {"N": n, "b": 1.0, "D_j_definition": "N|F-F_j|/(2b)", "E_D_j_squared": influences},
            )
        alive = np.arange(n + 1) / n
        add(
            379,
            "M'/N",
            f"alive-indicator-{n}",
            float(np.max(alive - (alive > 0))),
            0,
            {"N": n, "all_possible_alive_fractions": alive.tolist()},
        )
        p = 0.7
        probs = np.array([math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(n + 1)])
        a0 = p
        mstar = a0 / 4
        delta = min(1, math.exp(-n / 8) + math.exp(-a0 * n / 16))
        mass = float(probs @ alive)
        survive = 1 - probs[0]
        inp = {
            "N": n,
            "alive_count_law": probs.tolist(),
            "a0": a0,
            "p": 1.0,
            "m_star": mstar,
            "delta_N": delta,
        }
        add(376, "m_*=", f"alive-floor-constants-{n}", mstar, a0 / 4, inp, equal=True)
        for number in [377, 405]:
            add(
                number,
                "P_N(S,G_N^c)",
                f"uniform-alive-floor-{n}",
                float(probs[alive < mstar].sum()),
                delta,
                inp,
            )
        add(378, "q_N(S)", f"alive-survival-floor-{n}", a0, survive, inp)
        add(380, "\\mathbb E_S", f"alive-firstmoment-floor-{n}", a0, mass, inp)
        add(491, "\\eta q_N", f"mixture-survival-floor-{n}", a0, survive, inp)
        add(472, "e\\ge", f"extinction-lower-{n}", (1 - p) ** n, probs[0], inp, equal=True)
        add(473, "\\delta_N\\ge", f"extinction-floor-order-{n}", (1 - p) ** n, delta, inp)
        g = math.ceil(0.25 * n)
        add(
            420,
            "g=",
            f"safe-center-ceiling-{n}",
            g,
            math.ceil(0.25 * n),
            {"N": n, "rho": 0.25},
            equal=True,
        )
        add(
            428,
            "M^+\\ge",
            f"deterministic-safe-centers-{n}",
            g,
            n,
            {"N": n, "tau": 0.0, "g": g, "safe_centers_count": n},
        )
    # The priority/rank collision readout is checked only as its own configured
    # operator. It is not the canonical component Haar collision experiment.
    for restitution in [0.0, 0.5, 1.0]:
        vt = 0.3
        incoming = [-0.2, 0.8]
        vu = -0.4
        parent_incoming = [0.1]
        mt = float(np.mean([vt, *incoming]))
        mu = float(np.mean([vu, vt, *parent_incoming]))
        inp = {
            "v_t": vt,
            "incoming": incoming,
            "v_u": vu,
            "parent_other_incoming": parent_incoming,
            "alpha": restitution,
            "operator": "separate configured ordered-star, no native Haar attribution",
        }
        add(
            280,
            "m_t=",
            f"ordered-star-means-{restitution}",
            mt,
            (vt + sum(incoming)) / (1 + len(incoming)),
            {
                **inp,
                "m_t": mt,
                "m_u": mu,
                "parent_mean_residual": mu
                - (vu + vt + sum(parent_incoming)) / (2 + len(parent_incoming)),
            },
            equal=True,
        )
        for has_parent, root_rank, parent_rank, children in [
            (False, 1.0, 0.0, incoming),
            (True, 2.0, 1.0, incoming),
            (True, 1.0, 2.0, incoming),
            (True, 1.0, 2.0, []),
            (False, 1.0, 0.0, []),
        ]:
            root_mean = float(np.mean([vt, *children]))
            # Independent dispatcher: construct event centers and take the
            # highest-priority accepted event containing the tagged root.
            events = []
            if children:
                events.append((root_rank, root_mean))
            if has_parent:
                events.append((parent_rank, mu))
            expected = vt if not events else restitution * vt + (1 - restitution) * max(events)[1]
            direct = (
                restitution * vt + (1 - restitution) * root_mean
                if children and (not has_parent or root_rank > parent_rank)
                else restitution * vt + (1 - restitution) * mu
                if has_parent
                else vt
            )
            add(
                281,
                "v_t^{\\rm ord}",
                f"ordered-star-priority-{restitution}-{has_parent}-{root_rank}-{len(children)}",
                direct,
                expected,
                {
                    **inp,
                    "has_parent": has_parent,
                    "root_rank": root_rank,
                    "parent_rank": parent_rank,
                    "children": children,
                },
                equal=True,
            )
    # Exact harmonic BAOAB arithmetic for independent prescribed innovations;
    # the parent separately tests these against native archived stages.
    for h in [0.04, 0.5, 2.0]:
        c = h / 2
        k = 1 - c * c
        gamma = 1.0
        q = 0.3
        s = 0.2
        X = np.array([0.1, -0.4])
        V = np.array([0.2, 0.3])
        xi = np.array([1.1, -0.7])
        zeta = np.array([-0.2, 0.8])
        v1 = V - c * X
        x1 = X + c * v1
        v2 = math.exp(-gamma * h) * v1 + q * xi
        x2 = x1 + c * v2
        v3 = v2 - c * x2
        xplus = x2 + s * zeta
        cap = 2.0
        vplus = v3 / (1 + np.linalg.norm(v3) / cap)
        inverse = cap * vplus / (cap - np.linalg.norm(vplus))
        inp = {
            "h": h,
            "c": c,
            "k": k,
            "gamma": gamma,
            "q": q,
            "s": s,
            "X": X.tolist(),
            "V": V.tolist(),
            "xi": xi.tolist(),
            "zeta": zeta.tolist(),
            "v1": v1.tolist(),
            "x1": x1.tolist(),
            "v2": v2.tolist(),
            "x2": x2.tolist(),
            "v3": v3.tolist(),
            "xplus": xplus.tolist(),
            "vplus": vplus.tolist(),
        }
        add(
            27,
            "a=e",
            f"harmonic-OU-factor-{h}",
            math.exp(-gamma * h),
            math.exp(-gamma * h),
            inp,
            equal=True,
        )
        add(
            29,
            "v_2&=",
            f"harmonic-affine-noise-{h}",
            float(np.max(abs(v3 - ((k / c) * x2 - (k / c) * X - V)))),
            0,
            inp,
        )
        noise_matrix = np.block([
            [c * q * np.eye(2), s * np.eye(2)],
            [q * k * np.eye(2), np.zeros((2, 2))],
        ])
        output_noise = np.concatenate([xplus, v3]) - np.concatenate([
            k * X + c * V + c * math.exp(-gamma * h) * (V - c * X),
            k * math.exp(-gamma * h) * (V - c * X) - c * (k * X + c * V),
        ])
        add(
            30,
            "L=",
            f"harmonic-noise-matrix-{h}",
            float(np.max(abs(output_noise - noise_matrix @ np.concatenate([xi, zeta])))),
            0,
            {**inp, "L": noise_matrix.tolist()},
        )
        add(802, "v_1=", f"harmonic-stages-{h}", float(np.max(abs(x1 - (k * X + c * V)))), 0, inp)
        add(
            804,
            "x^+=",
            f"harmonic-final-position-{h}",
            float(np.max(abs(xplus - x2 - s * zeta))),
            0,
            inp,
        )
        add(
            806,
            "C_{V_{\\max}}^{-1}",
            f"harmonic-memory-cancellation-{h}",
            float(np.max(abs(inverse - (k / c) * xplus + (k / c) * X + V + (k * s / c) * zeta))),
            0,
            inp,
        )

    # Finite composition transfers with a completely specified Markov population
    # map. These test the source conditioning/transport algebra, not an unknown
    # iterated native mean-field trajectory.
    q = Q.sum(axis=1)
    delta = float((1 - q).max())
    full = np.column_stack((Q, 1 - q))
    vertices = np.eye(3)

    def transport(law1, law2, cost):
        nr, nc = cost.shape
        constraints = []
        values = []
        for row in range(nr):
            constraint = np.zeros((nr, nc))
            constraint[row, :] = 1
            constraints.append(constraint.ravel())
            values.append(law1[row])
        for column in range(nc):
            constraint = np.zeros((nr, nc))
            constraint[:, column] = 1
            constraints.append(constraint.ravel())
            values.append(law2[column])
        result = linprog(
            cost.ravel(), A_eq=constraints, b_eq=values, bounds=(0, None), method="highs"
        )
        if not result.success:
            message = "Finite transport linear program failed"
            raise ValueError(message)
        return float(result.fun)

    eta = np.array([0.4, 0.6])
    target = np.append(eta, 0.0)
    full_kernel = np.vstack((full, [0.0, 0.0, 1.0]))
    for n in range(1, 9):
        previous = eta.copy()
        full_output = previous @ full
        survival = float(full_output[:2].sum())
        eta = full_output[:2] / survival
        cost = np.array([
            [float(abs(vertex - row).sum() / 2) for row in full] for vertex in vertices
        ])
        epsilon = float(max(sum(full[i, j] * cost[j, i] for j in range(3)) for i in range(2)))
        W = transport(np.append(eta, 0), previous, cost)
        inputs = {
            "Q": Q.tolist(),
            "previous_filtered": previous.tolist(),
            "next_filtered": eta.tolist(),
            "map_rows_including_extinction": full.tolist(),
            "metric": "half full-variation on finite probability simplex",
            "epsilon_N": epsilon,
            "delta_N": delta,
            "n": n,
            "population_map": "explicit finite Markov map only, not native gas Fh",
        }
        add(533, "W_d", f"conditioned-map-transport-{n}", W, epsilon + 2 * delta, inputs)
        add(
            543,
            "\\varepsilon_N+",
            f"bad-set-transfer-{n}",
            epsilon,
            epsilon + delta,
            {**inputs, "good_set": "all nonextinct inputs"},
        )
        target_full = target @ full_kernel
        empirical_error = float(
            sum(eta[i] * float(abs(vertices[i] - target_full).sum() / 2) for i in range(2))
        )
        map_error = float(
            sum(previous[i] * float(abs(full[i] - target_full).sum() / 2) for i in range(2))
        )
        add(
            535,
            "\\mu_{n+1}",
            f"conditioned-map-triangle-{n}",
            empirical_error,
            epsilon + 2 * delta + map_error,
            {**inputs, "target_full_next": target_full.tolist(), "map_error": map_error},
        )
        alive_map = Q / q[:, None]
        cost_alive = np.array([
            [float(abs(np.eye(2)[j] - row).sum() / 2) for row in alive_map] for j in range(2)
        ])
        Wa = transport(eta, previous, cost_alive)
        epsilon_a = float(
            max(sum(Q[i, j] * cost_alive[j, i] for j in range(2)) for i in range(2))
        ) + float((1 - q).max())
        add(
            540,
            "W_{d_a}",
            f"conditioned-alive-map-transport-{n}",
            Wa,
            epsilon_a + 2 * delta,
            {**inputs, "epsilon_N_a": epsilon_a, "alive_map_rows": alive_map.tolist()},
        )
        add(
            575,
            "C_{\\rm upd}",
            f"noise-sensitive-map-transport-{n}",
            W,
            epsilon + 2 * delta,
            {
                **inputs,
                "beta_previous_k": delta,
                "beta_current_zero": delta,
                "min_Cupd_over_2sqrtN": epsilon,
                "N_fixture": 1,
                "Cupd_fixed_from_full_kernel_supremum": 2 * epsilon,
            },
        )
        # The source finite horizon recurrence assumes a modulus; here the
        # finite Markov map is a contraction with omega(r)=r, D*=1.
        error_before = float(
            sum(previous[i] * float(abs(vertices[i] - target).sum() / 2) for i in range(2))
        )
        r = 0.5
        error_after = empirical_error
        add(
            192,
            "\\omega(r)",
            f"finite-horizon-modulus-{n}",
            error_after,
            epsilon + delta + r + error_before / r,
            {
                **inputs,
                "r": r,
                "omega_r": r,
                "D_star": 1.0,
                "previous_error": error_before,
                "one_step_error_bound": epsilon + delta,
                "target_previous": target.tolist(),
                "target_next": target_full.tolist(),
                "full_3_state_kernel": full_kernel.tolist(),
            },
        )
        target = target_full
    # Stationary finite population-map defect: the QSD is known, and the
    # comparison operator is again the entire saved Markov row including death.
    cost = np.array([[float(abs(vertex - row).sum() / 2) for row in full] for vertex in vertices])
    W = transport(np.append(nu, 0), nu, cost)
    epsilon = float(max(sum(full[i, j] * cost[j, i] for j in range(3)) for i in range(2)))
    add(
        671,
        "\\min",
        "finite-stationary-map-defect",
        W,
        epsilon + 2 * delta,
        {
            **fixture,
            "map_rows_including_extinction": full.tolist(),
            "epsilon": epsilon,
            "beta_k": delta,
            "beta_0": delta,
            "operator": "explicit finite Markov population map; does not certify native stationary phases",
        },
    )

    add(
        665,
        "\\varepsilon_N+2",
        "finite-stationary-defect-prefix",
        W,
        epsilon + 2 * delta,
        {
            **fixture,
            "map_rows_including_extinction": full.tolist(),
            "epsilon": epsilon,
            "operator": "known finite substochastic QSD and explicit full Markov population map",
        },
        scope="Finite inequality prefix only of source 0665, evaluated under the exact supplied two-state QSD. Neither its N-to-infinity limit clause nor the existence/attraction of a native QSD is inferred from these operands.",
    )
    checks[-1]["validation_kind"] = "finite_prefix_only_analytic_limit_retained"

    # Single sampled-separation replacement with the native Global convention:
    # variance denominator M and sigma=sqrt(variance+sigma_min^2).
    for n in [2, 4, 8, 32, 128]:
        separations = np.linspace(0.1, 1.9, n)
        sigma = 0.5
        S = 2.0
        mstar = 1.0
        C = 2.5
        reward_factor = 1.3
        Hs = 2.1 * 0.5
        Lq = S / sigma + 3 * S**3 / (2 * sigma**3)
        L0 = Hs * Lq
        Fmin = 0.01
        Fmax = 2.1**2
        epsilon = 0.01
        saturation = 1.0
        La = max(
            1 / (saturation * (Fmin + epsilon)),
            (Fmax + epsilon) / (saturation * (Fmin + epsilon) ** 2),
        )
        B = C + 2 * La * L0

        def fitness(array):
            standardized = (array - array.mean()) / math.sqrt(float(array.var()) + sigma * sigma)
            return reward_factor * (0.1 + 2 / (1 + np.exp(-standardized)))

        original = fitness(separations)
        for row in [0, n - 1]:
            changed = separations.copy()
            changed[row] = 1.7 if row == 0 else 0.2
            replacement = fitness(changed)
            unchanged = np.arange(n) != row
            discrepancy = float(abs(original[unchanged] - replacement[unchanged]).max())
            inp = {
                "N": n,
                "replaced_measurement_address": row,
                "separations": separations.tolist(),
                "replacement_separations": changed.tolist(),
                "original_fitness": original.tolist(),
                "replacement_fitness": replacement.tolist(),
                "reward_factor": reward_factor,
                "sigma_s": sigma,
                "S_star": S,
                "m_star": mstar,
                "L_q": Lq,
                "H_s": Hs,
                "L_0": L0,
                "F_min": Fmin,
                "F_max": Fmax,
                "epsilon": epsilon,
                "saturation": saturation,
                "L_a": La,
                "C": C,
                "B": B,
                "operator": "native Global normalizer and Logistic powers1, with explicit finite marks and frozen reward factors",
            }
            add(218, "|F_i-", f"single-measurement-fitness-{n}-{row}", discrepancy, L0 / n, inp)
            # Coupled self-excluded uniform donor/gate: integrate the gate
            # disagreement probability across all donors for each unchanged row.
            probabilities = []
            for recipient in range(n):
                if recipient == row:
                    continue
                total = 0.0
                for donor in range(n):
                    if donor == recipient:
                        continue
                    a1 = min(
                        1,
                        max(
                            0,
                            (original[donor] - original[recipient])
                            / (original[recipient] + epsilon),
                        ),
                    )
                    a2 = min(
                        1,
                        max(
                            0,
                            (replacement[donor] - replacement[recipient])
                            / (replacement[recipient] + epsilon),
                        ),
                    )
                    total += abs(a1 - a2) / (n - 1)
                probabilities.append(total)
            add(
                220,
                "q_i",
                f"single-measurement-gate-{n}-{row}",
                max(probabilities),
                B / n,
                {
                    **inp,
                    "gate_disagreement_by_unchanged_row": probabilities,
                    "integration": "exact common-uniform gate and self-excluded uniform donor",
                },
            )

    values = np.array([0.0, 1.0, 4.0, 16.0, 64.0])
    probabilities = np.array([0.5, 0.25, 0.125, 0.0625, 0.0625])
    CW = float(probabilities @ values)
    for radius in [1.0, 2.0, 8.0, 32.0, 128.0]:
        add(
            96,
            "C_W/R",
            f"stationary-cloud-Markov-tail-{radius}",
            float(probabilities @ (values > radius)),
            CW / radius,
            {
                "population_law_atoms": "clouds whose average W equals the listed values",
                "mean_W_values": values.tolist(),
                "law": probabilities.tolist(),
                "C_W": CW,
                "R": radius,
            },
        )
        eps = 0.5
        tail = float(probabilities @ (values**2 * (values > radius)))
        moment = float(probabilities @ (values ** (2 + eps)))
        add(
            819,
            "R^{-\\epsilon}",
            f"W2-uniform-integrability-tail-{radius}",
            tail,
            radius ** (-eps) * moment,
            {
                "absolute_z": values.tolist(),
                "law": probabilities.tolist(),
                "epsilon": eps,
                "moment2plus_epsilon": moment,
                "R": radius,
            },
        )
    for n in [4, 8, 32]:
        C = 0.8
        for j in range(1, n):
            if j not in {1, n - 1}:
                continue
            categorical = np.array([1 - j * C / n] + [C / n] * j)
            for k in range(1, j + 1):
                attached = float(categorical[1 : k + 1].sum())
                for number in [164, 170]:
                    add(
                        number,
                        "Ck/N",
                        f"ordered-attachment-probability-{n}-{j}-{k}",
                        attached,
                        C * k / n,
                        {
                            "N": n,
                            "C": C,
                            "exposed_size": k,
                            "potential_earlier_addresses": j,
                            "full_row_categorical_law": categorical.tolist(),
                            "conditional_gate_agreement_probability": 1.0,
                            "old_common_forest_equals_original": True,
                        },
                    )
    for d in [1, 2, 4]:
        cap = 2.0
        velocities = np.array([[-0.3] * d, [0.1] * d, [0.4] * d]) * cap / math.sqrt(d)
        mean = velocities.mean(axis=0)
        rotation = -np.eye(d)
        for restitution in [0.0, 0.5, 1.0]:
            prepared = mean + restitution * ((velocities - mean) @ rotation.T)
            W = (1 + 2 * restitution) * cap
            add(
                485,
                "|V_0|",
                f"collision-velocity-envelope-d{d}-a{restitution}",
                float(np.max(np.linalg.norm(prepared, axis=1))),
                W,
                {
                    "d": d,
                    "V": cap,
                    "alpha": restitution,
                    "W": W,
                    "input_velocities": velocities.tolist(),
                    "orthogonal_rotation": rotation.tolist(),
                    "prepared": prepared.tolist(),
                    "operator": "native component readout for an explicitly supplied orthogonal matrix",
                },
            )
        c = 0.05
        sh = 0.3
        Bnorm = 0.2
        s = 0.8
        coefficient = np.column_stack((c * sh * Bnorm * np.eye(d), s * np.eye(d)))
        taux = math.sqrt(c * c * sh * sh * Bnorm * Bnorm + s * s)
        add(
            487,
            "\\|T\\|",
            f"Gaussian-noise-operator-d{d}",
            float(np.linalg.norm(coefficient, ord=2)),
            taux,
            {
                "d": d,
                "T": coefficient.tolist(),
                "c": c,
                "s_h": sh,
                "Bnorm": Bnorm,
                "sigma_x_sqrt_h": s,
                "tau_x": taux,
            },
        )
    add(
        323,
        "q_D^N",
        "finite-QSD-hazard-bracket",
        max(0.01 - (1 - alpha), (1 - alpha) - 1.0),
        0,
        {
            "Q": Q.tolist(),
            "alpha": alpha,
            "N": 2,
            "qD": 0.1,
            "a0": 0.05,
            "p": 0.05,
            "delta_N": min(1, math.exp(-0.05 * 2 / 8) + math.exp(-0.05 * 2 / 16)),
            "operator": "exact finite substochastic fixture with its separately verified survival floor and hazard bracket; not a native QSD",
        },
    )
    for n in [2, 4, 8]:
        z = np.sin(np.arange(n) * 1.7)
        mu = 0.2
        for ell in [1, 2]:
            observed = abs(
                float(
                    np.mean([
                        math.prod(z[i] for i in idx)
                        for idx in itertools.permutations(range(n), ell)
                    ])
                )
                - mu**ell
            )
            econd = abs(float(z.mean()) - mu) / 2
            add(
                631,
                "e_{N,n}",
                f"coordinate-product-transfer-{n}-{ell}",
                observed,
                ell * (ell - 1) / n + 2 * ell * 2 * econd,
                {
                    "N": n,
                    "ell": ell,
                    "test_indices": [1] * ell,
                    "physical_cloud_tests": z.tolist(),
                    "target_test": mu,
                    "metric": "single repeated bounded test family, d=|Lphi-muphi|/2",
                    "e_cond": econd,
                    "E_N_T": econd,
                    "H_N_n": 0.0,
                },
            )

    # Whole alive-product transfers, integrating every distinct-address tuple.
    for n in [2, 4, 8]:
        z = np.sin(np.arange(n) * 1.7)
        for alive_count in sorted({n, n // 2}):
            if alive_count < 2:
                continue
            ell = 2
            alive_fraction = alive_count / n
            mstar = alive_fraction
            a0 = 0.6
            u = 0.1
            U = float(z[:alive_count].sum() / n)
            distinct = float(
                np.mean([
                    math.prod(z[i] for i in indices)
                    for indices in itertools.permutations(range(alive_count), ell)
                ])
            )
            ratio_budget = (abs(U - u) + abs(alive_fraction - a0)) / a0
            add(
                641,
                "\\delta_N",
                f"alive-finite-product-{n}-{alive_count}",
                abs(distinct - (u / a0) ** ell),
                ell * (ell - 1) / (mstar * n) + ell * ratio_budget,
                {
                    "N": n,
                    "M": alive_count,
                    "ell": ell,
                    "m_star": mstar,
                    "a0": a0,
                    "target_submass_test": u,
                    "native_submass_test": U,
                    "native_alive_mass": alive_fraction,
                    "delta_N": 0.0,
                    "law": "uniform permutation of this finite cloud",
                    "alive_test_values": z[:alive_count].tolist(),
                },
            )
            add(
                676,
                "\\delta_N",
                f"stationary-alive-sampling-{n}-{alive_count}",
                abs(distinct - float(z[:alive_count].mean()) ** ell),
                ell * (ell - 1) / (mstar * n),
                {
                    "N": n,
                    "M": alive_count,
                    "ell": ell,
                    "m_star": mstar,
                    "delta_N": 0.0,
                    "alive_test_values": z[:alive_count].tolist(),
                    "known_QSD_model": "uniform permutation-refresh kernel with a constant independent killing probability; no native stationary-law claim",
                },
            )
        difference = abs(float(z.mean()) - 0.2)
        metric = difference / 2
        for j in [1, 2, 4]:
            add(
                653,
                "2^{j+1}",
                f"coordinate-metric-domination-{n}-{j}",
                difference,
                2 ** (j + 1) * metric,
                {
                    "N": n,
                    "j": j,
                    "test_family": "the same bounded test repeated at every index",
                    "metric": metric,
                    "test_difference": difference,
                },
            )
        add(
            655,
            "\\le2",
            f"bounded-input-discrepancy-{n}",
            difference,
            2.0,
            {"N": n, "bounded_test_cloud": z.tolist(), "target_test": 0.2},
        )
    for hazard in [0.001, 0.2, 0.8]:
        add(
            443,
            "1/\\bar b",
            f"stopped-lifetime-{hazard}",
            1 / hazard,
            1 / hazard,
            {"bar_b": hazard, "law": "geometric first bad-count time", "E_zeta": 1 / hazard},
            equal=True,
        )
    for center in [0.0, 0.3, 1.0, 4.0]:
        width = 2.0
        sigma = 0.7

        def density(x):
            return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)

        derivative = (
            density((width / 2 + center) / sigma) - density((width / 2 - center) / sigma)
        ) / sigma
        add(
            354,
            "f'(t)",
            f"Gaussian-box-center-monotonicity-{center}",
            derivative,
            0,
            {
                "width": width,
                "s": sigma,
                "t": center,
                "density_left": density((width / 2 + center) / sigma),
                "density_right": density((width / 2 - center) / sigma),
            },
        )
    # Probability-count generating function: every categorical draw is summed
    # exactly, and the joint Laplace transform is compared with its product law.
    for n in [2, 4]:
        categories = [0, 1, 2]
        p = np.array([0.7, 0.1, 0.2])
        g = np.array([0.0, 0.3, 0.8])
        exact = sum(
            math.prod(p[i] for i in row) * math.exp(-sum(g[i] for i in row))
            for row in itertools.product(categories, repeat=n)
        )
        product = (1 + p[1] * (math.exp(-g[1]) - 1) + p[2] * (math.exp(-g[2]) - 1)) ** n
        add(
            168,
            "\\prod_j",
            f"categorical-Laplace-product-{n}",
            exact,
            product,
            {"independent_rows": n, "full_row_probabilities": p.tolist(), "g": g.tolist()},
            equal=True,
        )
    # A supplied conditional changed-edge law with no common accepted edges.
    # Union of the endpoints is its exact conservative physical support, and
    # all subset outcomes are enumerated. This tests the composition estimate
    # separately from the native coupled replacement dataset.
    for n in [2, 4, 6]:
        probability = 0.1
        meanD = 0.0
        for mask in itertools.product([False, True], repeat=n):
            likelihood = probability ** sum(mask) * (1 - probability) ** (n - sum(mask))
            support = {i for i, b in enumerate(mask) if b} | {
                (i + 1) % n for i, b in enumerate(mask) if b
            }
            meanD += likelihood * len(support)
        C = 0.8
        LT = 1.0
        T = probability
        add(
            256,
            "3M_1(2C)",
            f"conditional-changed-edge-support-{n}",
            meanD / n,
            3 * math.exp(4 * C) * LT * T,
            {
                "N": n,
                "C": C,
                "L_T": LT,
                "T": T,
                "conditional_row_disagreement_probability": probability,
                "common_accepted_edges": [],
                "new_changed_edge_targets": "(i+1) mod N",
                "E_D": meanD,
                "scope": "explicit finite changed-edge conditional input satisfying the source probability hypothesis, not a native fitted influence",
            },
        )
    for y0, y1 in itertools.product(range(4), repeat=2):
        # Possible overlap only reduces the union of the two affected stars.
        root = {0}
        star0 = set(range(1, 3 + y0))
        star1 = set(range(2, 4 + y1))
        actual = len(root | star0 | star1)
        add(
            302,
            "D_i\\le",
            f"ordered-star-support-{y0}-{y1}",
            actual,
            1 + (2 + y0) + (2 + y1),
            {
                "Y_j0": y0,
                "Y_j1": y1,
                "affected_root": sorted(root),
                "old_star": sorted(star0),
                "new_star": sorted(star1),
                "operator": "separate priority/star influence support, not component Haar",
            },
        )

    # Complete quantitative ordered-star special case: constant positive
    # fitness makes every clone gate zero, hence there are no collision edges.
    # The actual quadratic BAOAB position law is then an independent Gaussian
    # for every row; its population reference is integrated exactly.
    h = 0.04
    c = h / 2
    gamma = 1.0
    a = math.exp(-gamma * h)
    k = 1 - c * c
    q2 = -math.expm1(-2 * gamma * h) / (2 * gamma)
    s2 = 0.1 * 0.1 * h
    tau2 = c * c * q2 + s2
    kernel_floor = math.exp(-32 / (2 * 8**2))
    C = 2 / kernel_floor
    DD = C
    B = C
    A = 1 + C + DD
    AD = 9 * (1 + 4 * C) * math.exp(8 * C) * ((1 + B) ** 2 + B) + max(1, 4 * B * B)
    Kord = 27 + 30 * C + 6 * C * C
    Aord = 2 * (AD + Kord + 1)
    N0 = math.ceil((8 * A) ** (6 / 5))
    Bord = 128 * A * A + 16 * (1 + 6 * C + 3 * C * C) * math.exp(8 * C) + math.sqrt(N0)
    for n in [2, 8, 32, 128]:
        positions = np.linspace(-0.3, 0.3, n)
        means = (k - c * c * a) * positions
        first = np.exp(-tau2 / 2) * np.sin(means)
        second = (1 - np.exp(-2 * tau2) * np.cos(2 * means)) / 2
        rowvar = second - first * first
        variance = float(rowvar.sum() / n**2)
        reference_mean = float(first.mean())
        inp = {
            "N": n,
            "d": 1,
            "h": h,
            "gamma": gamma,
            "cap": 2.0,
            "entering_positions": positions.tolist(),
            "entering_velocities": [0.0] * n,
            "reward_exponent": 0.0,
            "diversity_exponent": 0.0,
            "constant_fitness": 1.0,
            "all_gate_probabilities": 0.0,
            "accepted_components": "all singleton",
            "kernel_width": 8.0,
            "kernel_floor": kernel_floor,
            "C": C,
            "D_D": DD,
            "H_s": 0.0,
            "L_T": 0.0,
            "B": B,
            "A": A,
            "A_D": AD,
            "K_ord": Kord,
            "A_ord_unit": Aord,
            "N0": N0,
            "B_ord": Bord,
            "OU_variance": q2,
            "final_position_variance": s2,
            "total_position_variance": tau2,
            "row_position_means": means.tolist(),
            "row_test_means": first.tolist(),
            "row_test_variances": rowvar.tolist(),
            "native_kernel_mean": reference_mean,
            "exact_population_map_mean": reference_mean,
            "scope": "exact full quadratic Gaussian ordered-star zero-selection law; no active ordered-star collision claim",
        }
        # All measurement/clone/collision families have zero physical
        # influence in this declared constant-fitness regime. Each kinetic
        # Gaussian row changes exactly its own physical output: one OU row
        # innovation and one final position row innovation, almost surely.
        add(
            238,
            "\\sum_j\\mathbb E[D_j^2",
            f"zero-gate-all-family-influence-{n}",
            2 * n,
            n * (AD + 10 * (1 + 2 * C) * math.exp(4 * C) + 1),
            {
                **inp,
                "inactive_measurement_clone_and_collision_families": True,
                "kinetic_row_innovation_families": 2,
                "physical_support_size_per_kinetic_row": 1,
                "sum_E_D_j_squared": 2 * n,
            },
            scope=SCOPE
            + " Complete independent Gaussian zero-gate law; the displayed source sums physical replacement support counts over every innovation family, not fitted squared scalar output differences.",
        )
        if n >= N0:
            K = math.floor(n ** (1 / 6))
            M3 = (1 + 6 * C + 3 * C * C) * math.exp(8 * C)
            truncated = 64 * A * A * K**3 / n + 2 * M3 / K**3
            inputs = {
                **inp,
                "K": K,
                "K_upper_N_over_8A": n / (8 * A),
                "component_size": 1,
                "component_tail_probability": 0.0,
                "M3": M3,
                "truncated_bias_budget": truncated,
            }
            if K > n / (8 * A):
                message = "Declared truncation fixture violates K domain"
                raise ValueError(message)
            add(
                266,
                "\\leq2",
                f"zero-gate-truncated-full-bias-{n}",
                0.0,
                2 * truncated,
                inputs,
                scope=SCOPE
                + " Complete canonical zero-gate Gaussian population reference, exact expectation bias zero; N and K satisfy the source truncation domain.",
            )
            add(
                271,
                "64A^2",
                f"truncation-budget-to-root-N-{n}",
                truncated,
                (64 * A * A + 16 * M3) / math.sqrt(n),
                inputs,
            )
            for number in [265, 305]:
                add(
                    number,
                    "2M_3(C)/K^3",
                    f"zero-gate-two-component-tail-{n}",
                    0.0,
                    2 * M3 / K**3,
                    inputs,
                )
        add(
            293,
            "A_{\\rm ord,\\varphi}",
            f"ordered-full-Gaussian-variance-{n}",
            variance,
            Aord / n,
            inp,
        )
        add(
            294,
            "A_{\\rm ord,\\varphi}+4",
            f"ordered-full-Gaussian-bias-MSE-{n}",
            variance,
            (Aord + 4 * Bord * Bord) / n,
            {
                **inp,
                "exact_bias": 0.0,
                "bias_upper": 2 * Bord / math.sqrt(n),
                "exact_mean_square_error": variance,
            },
        )

    # Actual full-map zero-gate regime: couple the complete Gaussian BAOAB,
    # smooth cap and terminal status using the same innovations after maximally
    # coupling root types. There is no accepted incoming component when the
    # fitness exponents vanish. The coupling bounds full continuous-law TV;
    # it does not estimate TV from discrete Gaussian samples.
    rho = np.array([0.1, 0.2, 0.3, 0.4])
    eta = np.array([0.4, 0.3, 0.2, 0.1])
    atoms = np.array([-0.3, -0.1, 0.1, 0.3])
    common = np.minimum(rho, eta)
    left = rho - common
    right = eta - common
    joint = np.diag(common) + np.outer(left, right) / (1 - common.sum())
    h = 0.04
    c = h / 2
    gamma = 1.0
    a = math.exp(-gamma * h)
    k = 1 - c * c
    s = 0.1 * math.sqrt(h)
    q = math.sqrt(-math.expm1(-2 * gamma * h) / (2 * gamma))
    xi = 0.7
    zeta = -0.3
    cap = 2.0
    physical = []
    for X in atoms:
        v1 = -c * X
        x1 = X + c * v1
        v2 = a * v1 + q * xi
        x2 = x1 + c * v2
        v3 = v2 - c * x2
        physical.append([
            x2 + s * zeta,
            v3 / (1 + abs(v3) / cap),
            float(abs(x2 + s * zeta) <= 0.6),
        ])
    physical = np.array(physical)
    mismatch = sum(
        joint[i, j]
        for i in range(4)
        for j in range(4)
        if not np.array_equal(physical[i], physical[j])
    )
    Delta = float(abs(rho - eta).sum())
    kappa = math.exp(-32 / (2 * 8**2))
    LM = 2 + 2 / kappa
    Cmap = 1 / kappa
    Lbeta = 2 * Cmap * Cmap
    Bmap = 2 * Cmap * LM + Lbeta
    Lstep = 2 * (LM + 2 * math.exp(2 * Cmap) * Bmap)
    inp = {
        "rho": rho.tolist(),
        "eta": eta.tolist(),
        "physical_input_atoms": atoms.tolist(),
        "maximal_root_coupling": joint.tolist(),
        "coupling_row_margin_error": float(abs(joint.sum(axis=1) - rho).max()),
        "coupling_column_margin_error": float(abs(joint.sum(axis=0) - eta).max()),
        "full_variation_input": Delta,
        "exact_coupling_mismatch_probability": float(mismatch),
        "physical_output_for_shared_innovations": physical.tolist(),
        "h": h,
        "gamma": gamma,
        "sigma_v": 1.0,
        "sigma_x": 0.1,
        "cap": cap,
        "box_halfwidth": 0.6,
        "reward_exponent": 0.0,
        "diversity_exponent": 0.0,
        "fitness_constant": 1.0,
        "all_acceptances": 0.0,
        "incoming_component_intensity": 0.0,
        "m_star": 1.0,
        "kappa_D": kappa,
        "kappa_C": kappa,
        "L_M": LM,
        "C": Cmap,
        "L_F": 0.0,
        "L_beta": Lbeta,
        "B": Bmap,
        "L_step": Lstep,
        "source_operand_upper_certificate": "Full output variation is at most twice this explicit complete-kernel coupling mismatch probability. This certificate is computed exactly from finite input coupling weights; it is not an empirical full-TV estimator.",
    }
    add(
        789,
        "\\mathbb P",
        "complete-zero-gate-coupling",
        float(mismatch),
        (LM + 2 * math.exp(2 * Cmap) * Bmap) * Delta,
        inp,
        scope="Exact full-kernel zero-selection regime with fixed positive companion floors. All matched roots share cloning/collision (none), Gaussian innovations, smooth cap and terminal status; unmatched roots have distinct physical outputs for every shared innovation because harmonic BAOAB is injective in these root atoms. Active component coupling remains covered analytically, not measured by this special case.",
    )
    add(
        776,
        "L_{\\mathrm{step}}",
        "complete-zero-gate-variation-certificate",
        2 * float(mismatch),
        Lstep * Delta,
        inp,
        scope="Conservative complete continuous-law variation certificate in the exactly configured zero-selection native operator. Observed operand is an independently computed upper certificate for the left-hand full variation, not the unknown exact TV integral. It proves this source inequality for the recorded pair of laws; no uniform empirical certificate for active collision maps is inferred.",
    )
    checks[-1]["validation_kind"] = "conservative_full_law_certificate"

    if resonance:
        report = json.loads((resonance / "report.json").read_text())
        index = json.loads((resonance / "archive-index.json").read_text())
        entries = {e["path"]: e for e in index["entries"]}
        for case in report["cases"]:
            file = resonance / case["raw_operands"]
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            if digest != entries[case["raw_operands"]]["sha256"]:
                message = "Native resonance operands SHA mismatch"
                raise ValueError(message)
            rows = json.load(gzip.open(file, "rt"))["rows"]
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["replica"], row["storage_row"]].append(row)
            maxima = {798: 0.0, 803: 0.0, 805: 0.0, 806: 0.0}
            for (replica, address), items in grouped.items():
                items.sort(key=operator.itemgetter("coordinate"))
                prepared = np.array([x["prepared_velocity"] for x in items])
                uncapped = np.array([x["uncapped_output"] for x in items])
                physical = np.array([x["physical_capped_output"] for x in items])
                expected = -prepared / (1 + np.linalg.norm(prepared) / 2)
                inverse = 2 * physical / (2 - np.linalg.norm(physical))
                inp = {
                    "case": case["id"],
                    "N": case["N"],
                    "d": case["d"],
                    "replica": replica,
                    "storage_address": address,
                    "prepared_velocity": prepared.tolist(),
                    "uncapped_output": uncapped.tolist(),
                    "physical_output": physical.tolist(),
                    "cap": 2.0,
                    "h": 2.0,
                    "k": 0.0,
                    "raw_operands": case["raw_operands"],
                    "raw_sha256": digest,
                }
                scope = "Actual canonical Rust update, h=2 and harmonic coefficient 1, hence k=0. Addresses join saved arrays for observation only; the compared vector identity and norm are permutation equivariant. Prepared velocities include actual cloning and whole-component Haar dynamics."
                for number, value in [
                    (803, np.max(abs(uncapped + prepared))),
                    (805, np.max(abs(physical - expected))),
                    (798, np.max(abs(inverse - uncapped))),
                    (806, np.max(abs(inverse + prepared))),
                ]:
                    maxima[number] = max(maxima[number], float(value))
            for number, needle in [
                (803, "v_3="),
                (805, "C_{V_{\\max}}"),
                (798, "C_{V_{\\max}}^{-1}"),
                (806, "C_{V_{\\max}}^{-1}"),
            ]:
                add(
                    number,
                    needle,
                    case["id"],
                    maxima[number],
                    0,
                    {
                        "case": case,
                        "all_saved_native_vectors_checked": len(grouped),
                        "raw_operands": case["raw_operands"],
                        "raw_sha256": digest,
                        "maximum_over_all_replicas_rows_coordinates": True,
                    },
                    scope=scope,
                )
    return checks
