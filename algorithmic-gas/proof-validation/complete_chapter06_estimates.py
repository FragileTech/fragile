"""Statement ledger and numerical algebra for Chapter 6's conditional estimates.

Finite matrix and reference-law checks test the displayed mathematical estimates;
they never supply a missing full-native drift, QSD, joint LSI or global supremum.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/source/2_fractal_gas/convergence_program/06_convergence.md"
SCOPE = (
    "Numerical algebra on explicitly supplied analytic models. This checks the whole displayed "
    "expression and its own assumptions; it does not identify its coefficients with the actual "
    "native kernel without the separate recorded or analytic applicability certificate."
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    path = Path(__file__).with_name("generate_inventory.py")
    spec = importlib.util.spec_from_file_location("chapter06_inventory_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build(6, "06_convergence.md")


def algebra_checks(items):
    checks = []

    def add(label, name, observed, bound=0.0, relation="equal", tolerance=1e-9, inputs=None):
        observed, bound = float(observed), float(bound)
        passed = (
            abs(observed - bound) <= tolerance
            if relation == "equal"
            else observed <= bound + tolerance
        )
        item = items[label]
        checks.append({
            "id": name,
            "source_labels": [label],
            "source_statement": item["statement"],
            "observed": observed,
            "bound": bound,
            "relation": relation,
            "tolerance": tolerance,
            "signed_residual": observed - bound,
            "passed": passed,
            "hypotheses": inputs or {},
            "scope": SCOPE,
        })

    ac = np.array([[0.9, 0.03], [0.01, 0.85]])
    ak = np.array([[0.7, 0.05], [0.02, 0.8]])
    bc, bk = np.array([0.02, 0.03]), np.array([0.01, 0.04])
    matrix, source = ak @ ac, ak @ bc + bk
    w = np.linalg.solve((np.eye(2) - matrix).T, np.ones(2))
    coefficient = float(np.max((w @ matrix) / w))
    b = float(w @ source)
    values = np.array([1.0, 2.0])
    initial = float(w @ values)
    for step in range(1, 33):
        values = matrix @ values + source
        weighted = float(w @ values)
        bound = coefficient**step * initial + b * (1 - coefficient**step) / (1 - coefficient)
        add(
            "lem-convergence-drift-iteration",
            f"conservative_moment_iteration_{step}",
            weighted,
            bound,
            "upper",
            inputs={"r": coefficient, "b": b, "step": step},
        )
    add(
        "thm-foster-lyapunov-main",
        "composed_matrix_and_source",
        np.max(
            np.abs(
                ak @ (ac @ np.array([2.0, 3.0]) + bc)
                + bk
                - (matrix @ np.array([2.0, 3.0]) + source)
            )
        ),
        inputs={"A_C": ac.tolist(), "A_K": ak.tolist(), "b_C": bc.tolist(), "b_K": bk.tolist()},
    )
    add(
        "thm-synergistic-rate-derivation",
        "neumann_positive_weights",
        np.max(np.abs(w @ matrix - (w - 1))),
        inputs={"weights": w.tolist()},
    )
    add(
        "thm-synergistic-rate-derivation",
        "weighted_spectral_bound",
        max(abs(np.linalg.eigvals(matrix))),
        coefficient,
        "upper",
    )
    dx, dv, off_a, off_b = 0.3, 0.4, 0.05, 0.02
    weight = math.sqrt((off_a / dv) * (dx / off_b))
    effective = min(dx - weight * off_b, dv - off_a / weight)
    r = max(1 - dx + weight * off_b, 1 - dv + off_a / weight)
    add("thm-total-rate-explicit", "two_component_weighted_decrement", 1 - r, effective)
    add("thm-total-rate-explicit", "physical_time_rate", math.exp(math.log(r) / 0.04 * 0.04), r)
    invariant = np.linalg.solve(np.eye(2) - matrix, source)
    add(
        "thm-equilibrium-variance-bounds",
        "conservative_vector_fixed_point",
        np.max(np.abs(invariant - (matrix @ invariant + source))),
    )
    alpha = 0.95
    qsd_moment = np.linalg.solve(alpha * np.eye(2) - matrix, source)
    add(
        "thm-equilibrium-variance-bounds",
        "qsd_vector_drift_algebra",
        np.max(np.abs(alpha * qsd_moment - matrix @ qsd_moment - source)),
        inputs={
            "alpha": alpha,
            "scope": "Algebra only; this alpha is not a native QSD eigenvalue.",
        },
    )
    radius = 4 * b / (1 - coefficient)
    epsilon = 0.2
    beta = epsilon / (radius * (1 + coefficient) + 2 * b)
    rho1 = max(coefficient, (2 + beta * (coefficient * radius + 2 * b)) / (2 + beta * radius))
    rho = max(rho1, 1 - epsilon / 2)
    for s in np.linspace(radius, 10 * radius, 21):
        add(
            "thm-convergence-conservative-harris",
            f"harris_large_pair_{s:.8g}",
            2 + beta * (coefficient * s + 2 * b),
            rho * (2 + beta * s),
            "upper",
            inputs={
                "minorization": "Supplied epsilon=.2; not a native certificate.",
                "r": coefficient,
                "b": b,
                "R": radius,
                "beta": beta,
                "rho_H": rho,
            },
        )

    # Differentiate the matrix composition, its source, and fixed-weight rates.
    def maps(p):
        x, y = p
        c = np.array([[0.95 - 0.03 * x, 0.01 * y], [0.005 * x, 0.9 - 0.02 * y]])
        k = np.array([[0.8 - 0.02 * y, 0.01 * x], [0.003 * y, 0.85 - 0.01 * x]])
        sc, sk = np.array([0.01 * x, 0.02 * y]), np.array([0.03 * y, 0.01 * x])
        return c, k, sc, sk

    p = np.array([1.2, 0.8])
    c, k, sc, sk = maps(p)
    dc = [np.array([[-0.03, 0], [0.005, 0]]), np.array([[0, 0.01], [0, -0.02]])]
    dk = [np.array([[0, 0.01], [0, -0.01]]), np.array([[-0.02, 0], [0.003, 0]])]
    dsc, dsk = (
        [np.array([0.01, 0]), np.array([0, 0.02])],
        [np.array([0, 0.01]), np.array([0.03, 0])],
    )
    rates = 1 - (w @ (k @ c)) / w
    for j in range(2):
        eps = 1e-5
        offset = np.zeros(2)
        offset[j] = eps
        cp, kp, bp, dp = maps(p + offset)
        cm, km, bm, dm = maps(p - offset)
        finite_matrix = (kp @ cp - km @ cm) / (2 * eps)
        analytic = dk[j] @ c + k @ dc[j]
        add(
            "thm-explicit-rate-sensitivity",
            f"matrix_product_derivative_{j}",
            np.max(np.abs(finite_matrix - analytic)),
        )
        finite_source = (kp @ bp + dp - km @ bm - dm) / (2 * eps)
        analytic_source = dk[j] @ sc + k @ dsc[j] + dsk[j]
        add(
            "thm-explicit-rate-sensitivity",
            f"source_derivative_{j}",
            np.max(np.abs(finite_source - analytic_source)),
        )
        finite_rate = (
            (np.log(1 - (w @ (kp @ cp)) / w) - np.log(1 - (w @ (km @ cm)) / w)) / (2 * eps) * p[j]
        )
        analytic_rate = -p[j] / rates / w * (w @ analytic)
        add(
            "thm-explicit-rate-sensitivity",
            f"log_rate_sensitivity_{j}",
            np.max(np.abs(finite_rate - analytic_rate)),
            inputs={"fixed_weights": w.tolist()},
        )
        source0 = k @ sc + sk
        finite_bound = (
            np.log((kp @ bp + dp) / (1 - (w @ (kp @ cp)) / w))
            - np.log((km @ bm + dm) / (1 - (w @ (km @ cm)) / w))
        ) / (2 * eps)
        analytic_bound = analytic_source / source0 + (w @ analytic) / w / rates
        add(
            "def-equilibrium-sensitivity-matrix",
            f"moment_bound_sensitivity_{j}",
            np.max(np.abs(finite_bound - analytic_bound)),
        )
    sensitivity = np.array([[1.0, 2.0, 0.5], [-0.2, 0.7, 1.2]])
    u, singular, vt = np.linalg.svd(sensitivity, full_matrices=True)
    for i in range(2):
        add(
            "thm-svd-rate-matrix",
            f"singular_direction_{i}",
            np.linalg.norm(sensitivity @ vt[i] - singular[i] * u[:, i]),
        )
    add("thm-svd-rate-matrix", "null_direction", np.linalg.norm(sensitivity @ vt[2]))
    add(
        "thm-svd-rate-matrix",
        "kernel_dimension",
        sensitivity.shape[1] - np.linalg.matrix_rank(sensitivity),
        1,
    )
    v = 0.3 * vt[0] - 0.4 * vt[1]
    add(
        "prop-condition-number-rate",
        "restricted_upper",
        np.linalg.norm(sensitivity @ v),
        singular[0] * np.linalg.norm(v),
        "upper",
    )
    add(
        "prop-condition-number-rate",
        "restricted_lower",
        singular[-1] * np.linalg.norm(v),
        np.linalg.norm(sensitivity @ v),
        "upper",
    )
    move = np.array([0.03, -0.02, 0.01])
    change = sensitivity @ move
    add(
        "thm-error-propagation",
        "finite_log_rate_lipschitz",
        np.linalg.norm(change),
        singular[0] * np.linalg.norm(move),
        "upper",
    )
    add(
        "thm-error-propagation",
        "affine_taylor_remainder",
        np.linalg.norm(change - sensitivity @ move),
    )
    baseline = np.array([-1.0, -0.8])
    add(
        "thm-error-propagation",
        "minimum_log_rate_perturbation",
        abs(min(baseline + change) - min(baseline)),
        max(abs(change)),
        "upper",
    )

    for direction in [-2.0, -0.3, 0.7, 3.0]:
        predicted = min(direction, -2 * direction)
        observed = min(1e-7 * direction, -2e-7 * direction) / 1e-7
        add("thm-subgradient-min", f"minimum_direction_{direction}", observed, predicted)
    for a, b, budget in [(0.7, 2.1, 3.0), (3.0, 0.2, 1.0), (1.0, 1.0, 5.0)]:
        optimal = b * budget / (a + b)
        result = minimize_scalar(
            lambda x: -min(a * x, b * (budget - x)),
            bounds=(0, budget),
            method="bounded",
            options={"xatol": 1e-13},
        )
        add(
            "thm-closed-form-optimum",
            f"two_rate_allocation_{a}_{b}_{budget}",
            result.x,
            optimal,
            tolerance=1e-7,
        )
        add(
            "thm-balanced-optimality",
            f"balanced_active_rates_{a}_{b}_{budget}",
            a * optimal,
            b * (budget - optimal),
        )
    velocities = np.array([[0.3, -0.8], [1.1, 0.5], [-0.4, 0.2]])
    mean = np.mean(velocities, axis=0)
    for restitution in [0.0, 0.5, 1.0]:
        after = mean + restitution * (velocities - mean)
        predicted = len(velocities) * np.dot(mean, mean) + restitution**2 * np.sum(
            (velocities - mean) ** 2
        )
        add(
            "prop-restitution-friction-coupling",
            f"relative_collision_energy_{restitution}",
            np.sum(after**2),
            predicted,
        )
    c, selection, b0, b1, jitter = 0.2, 0.6, 0.03, 0.4, 0.1
    invariant = (b0 + b1 * jitter**2) / (c * selection)
    add(
        "prop-jitter-cloning-coupling",
        "scalar_jitter_fixed_point",
        (1 - c * selection) * invariant + b0 + b1 * jitter**2,
        invariant,
    )
    dx, dv, coupling, bandwidth = 0.7, -0.4, 1.3, 0.8
    log_ratio = -(dx + coupling * dv) / (2 * bandwidth**2)
    add(
        "prop-phase-space-pairing", "companion_log_ratio", math.log(math.exp(log_ratio)), log_ratio
    )
    eps = 1e-5

    def f(lam, width):
        return -(dx + lam * dv) / (2 * width**2)

    add(
        "prop-phase-space-pairing",
        "companion_metric_derivative",
        (f(coupling + eps, bandwidth) - f(coupling - eps, bandwidth)) / (2 * eps),
        -dv / (2 * bandwidth**2),
    )
    add(
        "prop-phase-space-pairing",
        "companion_log_bandwidth_derivative",
        (f(coupling, bandwidth * math.exp(eps)) - f(coupling, bandwidth * math.exp(-eps)))
        / (2 * eps),
        (dx + coupling * dv) / bandwidth**2,
    )
    # Exact Gaussian reference-law entropy/TV, not a full-native stationary law.
    for shift in [0.0, 0.1, 0.5, 1.0, 2.0]:
        entropy = shift**2 / 2
        tv = 2 * ndtr(abs(shift) / 2) - 1
        add(
            "thm-convergence-entropy-to-tv",
            f"gaussian_reference_pinsker_{shift}",
            tv,
            math.sqrt(entropy / 2),
            "upper",
            inputs={
                "reference": "N(0,1)",
                "compared_law": f"N({shift},1)",
                "scope": "Exact reference-law Pinsker only; no native entropy identity.",
            },
        )
    # A declared finite killed kernel verifies normalization and the complete
    # two-sided block theorem numerically. It is not substituted for the gas.
    killed = np.array([[0.3, 0.2], [0.2, 0.4]])
    eigvals, eigvecs = np.linalg.eig(killed.T)
    index = int(np.argmax(eigvals))
    alpha = float(eigvals[index])
    nu = eigvecs[:, index] / sum(eigvecs[:, index])
    lower, upper = 0.4, 0.8
    rho = 1 - lower / upper
    fixture = {
        "Q": killed.tolist(),
        "eta": [0.5, 0.5],
        "c": lower,
        "C": upper,
        "alpha": alpha,
        "nu": nu.tolist(),
        "scope": "Exact supplied two-state killed-kernel algebra; not the native swarm law.",
    }
    add(
        "def-qsd",
        "finite_kernel_eigenmeasure",
        np.max(np.abs(nu @ killed - alpha * nu)),
        inputs=fixture,
    )
    add(
        "thm-main-convergence",
        "finite_kernel_eigenvalue_lower",
        lower,
        alpha,
        "upper",
        inputs=fixture,
    )
    add(
        "thm-main-convergence",
        "finite_kernel_eigenvalue_upper",
        alpha,
        upper,
        "upper",
        inputs=fixture,
    )
    for time in range(1, 17):
        power = np.linalg.matrix_power(killed, time)
        conditioned = []
        for initial_law in [np.array([1.0, 0.0]), np.array([0.0, 1.0])]:
            raw = initial_law @ power
            conditional = raw / sum(raw)
            conditioned.append(conditional)
            add(
                "def-qsd",
                f"conditional_own_denominator_{time}_{initial_law[0]}",
                sum(conditional),
                1,
                inputs=fixture,
            )
            add(
                "thm-main-convergence",
                f"conditioned_block_convergence_{time}_{initial_law[0]}",
                sum(abs(conditional - nu)) / 2,
                rho**time,
                "upper",
                inputs=fixture,
            )
        add(
            "prop-qsd-properties",
            f"geometric_qsd_survival_{time}",
            sum(nu @ power),
            alpha**time,
            inputs=fixture,
        )
        add(
            "rem-extinction-inevitable",
            f"uniform_hazard_survival_{time}",
            max(sum(power.T)),
            0.6**time,
            "upper",
            inputs=fixture,
        )
    expected_lifetime = np.linalg.solve(np.eye(2) - killed, np.ones(2))
    add(
        "rem-extinction-inevitable",
        "uniform_hazard_lifetime",
        max(expected_lifetime),
        1 / 0.4,
        "upper",
        inputs=fixture,
    )
    add(
        "prop-qsd-properties",
        "qsd_lifetime",
        nu @ expected_lifetime,
        1 / (1 - alpha),
        inputs=fixture,
    )
    moment = np.array([1.0, 2.0])
    drift_r = 0.4
    drift_b = max(killed @ moment - drift_r * moment)
    add(
        "thm-equilibrium-variance-bounds",
        "finite_kernel_qsd_moment",
        nu @ moment,
        drift_b / (alpha - drift_r),
        "upper",
        inputs=fixture,
    )
    for count in [2, 4, 16, 64]:
        p = 0.3
        exact_fewer_two = p**count + count * (1 - p) * p ** (count - 1)
        add(
            "prop-convergence-survival-bound",
            f"independent_identified_deaths_{count}",
            exact_fewer_two,
            count * p ** (count - 1),
            "upper",
            inputs={
                "identified_count": count,
                "conditional_failure_probability": p,
                "scope": "Exact independent final-death model under its stated conditional premise.",
            },
        )
        # Product Gaussian LSI rho=1/sigma² with bounded F=clip(mean(X),-1,1).
        # This is the conditional candidate Gaussian law, not its killed readout.
        sigma = 0.2
        for multiple in [1.0, 2.0, 3.0]:
            threshold = multiple * sigma / math.sqrt(count)
            exact_tail = 2 * ndtr(-multiple)
            lsi_bound = 2 * math.exp(-count * threshold**2 / (2 * sigma**2))
            add(
                "thm-convergence-lsi-concentration",
                f"conditional_gaussian_lsi_N{count}_c{multiple}",
                exact_tail,
                lsi_bound,
                "upper",
                inputs={
                    "rho": 1 / sigma**2,
                    "L": 1,
                    "N": count,
                    "observable": "clip(mean(X),-1,1), conditional centered final Gaussian coordinates",
                    "scope": "Analytic Gaussian reference/candidate law only; not joint stationary/QSD LSI across alive strata.",
                },
            )
    epsilon, prefactor, contraction = 0.01, 2.0, 0.8
    mixing_step = math.ceil(math.log(prefactor / epsilon) / (-math.log(contraction)))
    add(
        "prop-mixing-time-explicit",
        "mixing_time_integer_ceiling",
        prefactor * contraction**mixing_step,
        epsilon,
        "upper",
        inputs={
            "prefactor": prefactor,
            "rho": contraction,
            "scope": "Supplied contraction estimate; no native rho identification.",
        },
    )
    return checks


def missing_premises(label):
    if label in {
        "def-qsd",
        "prop-qsd-properties",
        "thm-main-convergence",
        "thm-equilibrium-variance-bounds",
    }:
        return "Native QSD/eigenvalue and the same-kernel two-sided surviving-block domination are not identified by finite survivor samples. Algebra below does not provide those premises."
    if label == "thm-convergence-lsi-concentration":
        return "No certified joint invariant/QSD swarm LSI across alive-status strata. Final conditional Gaussian LSI is a different law and can only validate its own smooth candidate observable."
    if label == "thm-convergence-entropy-to-tv":
        return "Native stationary/conditioned reference and its complete entropy dissipation identity, including own survival normalization, are not supplied. Gaussian-reference Pinsker is tested separately."
    if label in {
        "thm-convergence-conservative-harris",
        "thm-phi-irreducibility",
        "thm-aperiodicity",
        "prop-mixing-time-explicit",
    }:
        return "Native drift/minorization/actual mixing coefficient must share the same law and stated class. Positive finite-N minorization or empirical endpoint decline alone does not close default mixing."
    if label in {
        "prop-complete-drift-summary",
        "prop-position-rate-explicit",
        "prop-velocity-rate-explicit",
        "prop-boundary-rate-explicit",
        "thm-foster-lyapunov-main",
        "thm-synergistic-rate-derivation",
        "thm-total-rate-explicit",
    }:
        return "Constant component matrices and sources require full-native analytic estimates on the same entering/intermediate/output class; selected-source and slow-zone absorption must close before assigning a global rate."
    return "Validate only with the complete displayed hypotheses; no finite sampling claim replaces a global infimum, supremum, stationary identity, or continuous parameter neighbourhood."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--native-completion", type=Path, action="append", default=[])
    parser.add_argument("--native-evidence", type=Path, action="append", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inv = inventory()
    items = {item["id"]: item for item in inv["formal_items"]}
    checks = algebra_checks(items)
    native = []
    native_by_statement = {}
    for path in [*args.native_completion, *args.native_evidence]:
        value = json.loads(path.read_text())
        comparisons = value.get("comparisons", [])
        if "uniform_box_hazards" in value:
            labels = ["thm-foster-lyapunov-main", "lem-convergence-drift-iteration"]
            kind = "complete native conditional p4/p8 source-position moment"
        elif "profiles" in value and "maximum_acceptance_upper_residual" in value:
            labels = ["thm-foster-lyapunov-main", "lem-convergence-drift-iteration"]
            kind = "global weak-selection native source and full-update moment drift"
        elif "maximum_incoming_load_residual" in value:
            labels = ["thm-foster-lyapunov-main", "lem-convergence-drift-iteration"]
            kind = "exact native donor/acceptance conditional source integration"
        elif "checkpoint" in value and "independent_law_groups" in value.get("summary", {}):
            labels = ["thm-convergence-lsi-concentration"]
            kind = "conditional native final-Gaussian candidate concentration"
        else:
            labels = sorted({label for c in comparisons for label in c.get("source_labels", [])})
            kind = "retained report under its explicit analytic applicability"
        numeric = [
            c
            for c in comparisons
            if isinstance(c.get("observed"), (int, float))
            and isinstance(c.get("bound"), (int, float))
            and math.isfinite(c["observed"])
            and math.isfinite(c["bound"])
        ]
        residuals = [c["observed"] - c["bound"] for c in numeric]
        owned_subestimates = {}
        for c in numeric:
            expressions = (
                [c["source_formula"]] if c.get("source_formula") else c.get("source_quotes", [])
            )
            for label in c.get("source_labels", []):
                if label not in items:
                    raise ValueError(f"Unknown native source label {label}")
                owned = [
                    expression
                    for expression in expressions
                    if expression in items[label]["statement"]
                ]
                if expressions and not owned:
                    raise ValueError(
                        f"Native subestimate quote does not belong to {label}: {c['id']}"
                    )
                if owned:
                    own = owned_subestimates.setdefault(
                        label,
                        {
                            "comparisons": 0,
                            "failures": 0,
                            "exact_source_subexpressions": [],
                            "scopes": [],
                        },
                    )
                    own["comparisons"] += 1
                    own["failures"] += not c["passed"]
                    for expression in owned:
                        if expression not in own["exact_source_subexpressions"]:
                            own["exact_source_subexpressions"].append(expression)
                    if c.get("scope") not in own["scopes"]:
                        own["scopes"].append(c.get("scope"))
        evidence = {
            "path": str(path.resolve()),
            "sha256": sha(path),
            "summary": value.get("summary", {}),
            "kind": kind,
            "comparisons": comparisons,
            "finite_comparisons": len(numeric),
            "exact_owned_native_subestimates": owned_subestimates,
            "maximum_signed_residual": max(residuals, default=None),
            "analytic_hazard_profiles": value.get("uniform_box_hazards", []),
            "native_drift_profiles": value.get("profiles", []),
            "gaps": value.get("gaps", []),
            "scope": value.get("scope", "Scope must be read from each retained comparison."),
            "source_statement_credit": "Related explicit estimate only; no whole-statement native theorem credit is inferred.",
        }
        native.append(evidence)
        for label in labels:
            native_by_statement.setdefault(label, []).append({
                key: evidence[key]
                for key in (
                    "path",
                    "sha256",
                    "kind",
                    "finite_comparisons",
                    "maximum_signed_residual",
                    "scope",
                    "exact_owned_native_subestimates",
                )
            })
        if value.get("uniform_box_hazards"):
            for label in ("prop-convergence-survival-bound", "rem-extinction-inevitable"):
                native_by_statement.setdefault(label, []).append({
                    "path": str(path.resolve()),
                    "sha256": sha(path),
                    "kind": "Unrestricted-mean final Gaussian box escape and fixed-N k0/k<2 hazard certificates",
                    "profiles": value["uniform_box_hazards"],
                    "scope": "Analytic log-scale bounds; no finite-run identification of an extremely rare probability or N-uniform whole-swarm hazard.",
                })
    ledger = []
    for label, item in items.items():
        applicable = [check for check in checks if label in check["source_labels"]]
        ledger.append({
            "id": label,
            "kind": item["kind"],
            "title": item["title"],
            "source_statement": item["statement"],
            "formula_expression_ids": item["formula_expression_ids"],
            "source_line": item["source_line"],
            "algebra_comparisons": len(applicable),
            "algebra_comparisons_failed": sum(not c["passed"] for c in applicable),
            "status": "whole-statement applicability retained; algebra checked"
            if applicable
            else "hypothesis audit; see native evidence and obligations",
            "missing_native_premises": missing_premises(label),
            "native_related_estimates": native_by_statement.get(label, []),
            "exact_source_owned_native_subestimate_comparisons": sum(
                e.get("exact_owned_native_subestimates", {}).get(label, {}).get("comparisons", 0)
                for e in native_by_statement.get(label, [])
            ),
            "complete_native_statement_certificate": False,
            "global_native_certificate": False,
        })
    report = {
        "chapter": 6,
        "source_path": str(SOURCE),
        "source_sha256": sha(SOURCE),
        "script_sha256": sha(Path(__file__)),
        "comparisons": checks,
        "statement_ledger": ledger,
        "native_completion_reports": native,
        "summary": {
            "formal_items": len(ledger),
            "current_inventory_counts": inv["counts"],
            "native_evidence_reports": len(native),
            "exact_source_owned_native_subestimate_comparisons": sum(
                row["exact_source_owned_native_subestimate_comparisons"] for row in ledger
            ),
            "algebra_comparisons": len(checks),
            "algebra_comparisons_failed": sum(not c["passed"] for c in checks),
            "new_native_steps": 0,
        },
        "scope": SCOPE,
    }
    (args.output / "statement-estimates.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "source-snapshot.md").write_bytes(SOURCE.read_bytes())
    (args.output / "executed-helper.py").write_bytes(Path(__file__).read_bytes())
    lines = [
        "# Chapter 6 statement estimate ledger",
        "",
        SCOPE,
        "",
        "| Statement | Algebra checks | Failures | Exact-source native subestimate checks | Native estimate evidence | Required native hypothesis |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in ledger:
        lines.append(
            f"| `{row['id']}` | {row['algebra_comparisons']} | {row['algebra_comparisons_failed']} | {row['exact_source_owned_native_subestimate_comparisons']:,} | "
            + "; ".join(
                f"{e['kind']} ([report]({e['path']}))" for e in row["native_related_estimates"]
            )
            + f" | {row['missing_native_premises']} |"
        )
    (args.output / "statement-ledger.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report["summary"]))
    if report["summary"]["algebra_comparisons_failed"]:
        message = "Chapter 6 algebra discrepancy"
        raise SystemExit(message)


if __name__ == "__main__":
    main()
