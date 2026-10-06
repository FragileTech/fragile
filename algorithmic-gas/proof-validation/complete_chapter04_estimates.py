"""Numerical statement ledger and exact small-population Chapter 4 pressure audit.

Reads immutable native Rust operator frames. Never runs a replacement algorithm,
fits a convergence constant, or treats a chosen coupling as a transition-law metric.
"""

import argparse
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_cloud(points):
    return [math.fsum(p[a] for p in points) / len(points) for a in range(len(points[0]))]


def norm2(x):
    return math.fsum(z * z for z in x)


def distance2(x, y):
    return math.fsum((a - b) ** 2 for a, b in zip(x, y, strict=True))


def centered(points):
    m = mean_cloud(points)
    return [[x - a for x, a in zip(p, m, strict=True)] for p in points]


def variance(points):
    return math.fsum(norm2(p) for p in centered(points)) / len(points)


def logistic(z, profile):
    require(profile["kind"] == "logistic", "Only archived native logistic maps supported")
    s = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
    return profile["floor"] + profile["amplitude"] * s


def standardize(values, regularizer):
    m = math.fsum(values) / len(values)
    scale = math.sqrt(math.fsum((x - m) ** 2 for x in values) / len(values) + regularizer**2)
    return [(x - m) / scale for x in values]


def features(points, velocities, distance):
    require(distance["kind"] == "squashed_phase_space", "Unsupported feature geometry")
    output = []
    for x, v in zip(points, velocities, strict=True):
        sx = 1 + math.sqrt(norm2(x)) / distance["position_radius"]
        sv = 1 + math.sqrt(norm2(v)) / distance["velocity_radius"]
        output.append([a / sx for a in x] + [math.sqrt(distance["lambda"]) * a / sv for a in v])
    return output


def donor_rows(feature, provider):
    require(provider["law"] == "independent", "A product measurement law requires independence")
    require(provider["kernel"]["kind"] == "gaussian", "Unsupported donor kernel")
    require(not provider["allow_self"], "Self-donor convention differs from theorem")
    require(provider["count"] == 1 and provider["history_window"] == 0, "Unsupported donor plan")
    width = provider["kernel"]["width"]
    rows = []
    for i, x in enumerate(feature):
        w = [
            0.0 if i == j else math.exp(-distance2(x, y) / (2 * width**2))
            for j, y in enumerate(feature)
        ]
        total = math.fsum(w)
        require(total > 0, "All-alive donor law needs at least two rows")
        rows.append([a / total for a in w])
    return rows


def fitness_for_pattern(feature, reward, pattern, config):
    fitness = config["fitness"]
    require(
        fitness["reward_standardizer"]["kind"] == "global", "Shared reward normalizer required"
    )
    require(
        fitness["diversity_standardizer"]["kind"] == "global",
        "Shared diversity normalizer required",
    )
    separation = [
        math.sqrt(distance2(x, feature[j]) + fitness["distance_floor"] ** 2)
        for x, j in zip(feature, pattern, strict=True)
    ]
    rz = standardize(reward, fitness["reward_standardizer"]["sigma_min"])
    dz = standardize(separation, fitness["diversity_standardizer"]["sigma_min"])
    scores = [
        logistic(a, fitness["reward_map"]) ** fitness["reward_exponent"]
        * logistic(b, fitness["diversity_map"]) ** fitness["diversity_exponent"]
        for a, b in zip(rz, dz, strict=True)
    ]
    return scores, separation, rz, dz


def conditional_copying(points, scores, rows, config):
    """Exact independent recipient donor/Bernoulli/jitter positional moments."""
    decision = config["clone_decision"]
    require(decision["every"] == 1, "This integration audits an active cloning update")
    epsilon, saturation = decision["epsilon"], decision["saturation"]
    n, d = len(points), len(points[0])
    sigma = config["clone_transform"]["jitter_amplitude"]
    require(
        config["clone_transform"]["jitter"]["innovation"] == "gaussian", "Gaussian jitter required"
    )
    probabilities, means, covariances = [], [], []
    for i, x in enumerate(points):
        edge = [
            w * min(1, max(0, (f - scores[i]) / (scores[i] + epsilon) / saturation))
            for w, f in zip(rows[i], scores, strict=True)
        ]
        p = math.fsum(edge)
        shift = [
            math.fsum(a * (y[b] - x[b]) for a, y in zip(edge, points, strict=True))
            for b in range(d)
        ]
        means.append([a + b for a, b in zip(x, shift, strict=True)])
        covariances.append(
            math.fsum(a * distance2(x, y) for a, y in zip(edge, points, strict=True))
            - norm2(shift)
            + d * sigma**2 * p
        )
        probabilities.append(p)
    expected = variance(means) + (1 - 1 / n) * math.fsum(covariances) / n
    return probabilities, expected


def integrate_measurement(points, velocities, reward, config):
    """Enumerate the complete product measurement law, not sampled outcomes."""
    n = len(points)
    require(2 <= n <= 4, "Exact finite enumeration is restricted to 2..4 rows")
    feature = features(points, velocities, config["distance_donors"]["distance"])
    measurement = donor_rows(feature, config["distance_donors"])
    cloning = donor_rows(
        features(points, velocities, config["cloning_donors"]["distance"]),
        config["cloning_donors"],
    )
    probabilities = [0.0] * n
    expected_variance = 0.0
    total = 0.0
    ledger = []
    for pattern in itertools.product(*[[j for j in range(n) if j != i] for i in range(n)]):
        mass = math.prod(measurement[i][j] for i, j in enumerate(pattern))
        scores, separation, _, _ = fitness_for_pattern(feature, reward, pattern, config)
        p, var = conditional_copying(points, scores, cloning, config)
        total += mass
        expected_variance += mass * var
        probabilities = [a + mass * b for a, b in zip(probabilities, p, strict=True)]
        ledger.append({
            "measurement_companions": pattern,
            "probability": mass,
            "fitness": scores,
            "separation": separation,
            "row_pressure": p,
            "expected_variance": var,
        })
    require(abs(total - 1) < 2e-12, "Incomplete measurement probability integration")
    return {
        "probability_mass": total,
        "mean_row_pressure": probabilities,
        "expected_variance": expected_variance,
        "expected_variance_drift": expected_variance - variance(points),
        "patterns": ledger,
    }


def reward_profile(benchmark, d):
    """Analytic bounds on the declared native box [-2,2]^d, never sampled maxima."""
    if benchmark == "constant":
        return 0.0, 0.0
    k = 1.0 if benchmark == "quadratic" else 2.0
    if benchmark != "rastrigin":
        return 2 * k * math.sqrt(d), 2 * k * d
    a, w, radius = 10.0, math.tau, 2.0

    def g(x):
        return abs(k * x + a * w * math.sin(w * x))

    candidates = [-radius, radius]
    angle = math.acos(-k / (a * w**2))
    for sign in (-1, 1):
        start = math.ceil((-radius * w - sign * angle) / math.tau)
        end = math.floor((radius * w - sign * angle) / math.tau)
        candidates.extend((sign * angle + math.tau * j) / w for j in range(start, end + 1))
    return math.sqrt(d) * max(map(g, candidates)), 24 * d


def keystone_constants(config, benchmark, d, threshold=0.01):
    """Complete source assembly, including a rigorous powered-logistic specialization."""
    distance = config["distance_donors"]["distance"]
    require(
        distance == config["cloning_donors"]["distance"],
        "This specialization needs the same two feature maps",
    )
    rx, rv, lam = distance["position_radius"], distance["velocity_radius"], distance["lambda"]
    cap = config["kinetic"]["velocity_cap"]
    require(
        config["boundary"]["kind"] == "absorbing_box",
        "An analytic entering box certificate is required",
    )
    lower, upper = config["boundary"]["domain"]["lower"], config["boundary"]["domain"]["upper"]
    require(
        lower == [-2.0] * d and upper == [2.0] * d,
        "Reward certificate is for the declared radius-two box",
    )
    bx = 2 * math.sqrt(d)
    mx = rx**2 / (rx + bx) ** 2
    mz = min(mx, math.sqrt(lam) * rv**2 / (rv + cap) ** 2)
    d02 = 4 * (rx**2 + lam * rv**2)
    bf = max(rx, math.sqrt(lam) * rv)
    fit = config["fitness"]
    rp, dp = fit["reward_exponent"], fit["diversity_exponent"]
    require(rp >= 0 and dp > 0, "Positive diversity exponent is an actual pressure hypothesis")
    rm, dm = fit["reward_map"], fit["diversity_map"]
    require(rm["kind"] == dm["kind"] == "logistic", "Unsupported powered map")
    amin, amax = rm["floor"] ** rp, (rm["floor"] + rm["amplitude"]) ** rp
    fplus = (dm["floor"] + dm["amplitude"]) ** dp
    fstar = amax * fplus
    lr, osc = reward_profile(benchmark, d)
    lh = (
        0.0
        if rp == 0
        else rp
        * rm["amplitude"]
        / 4
        * max(rm["floor"] ** (rp - 1), (rm["floor"] + rm["amplitude"]) ** (rp - 1))
    )
    la = lh * lr / (fit["reward_standardizer"]["sigma_min"] * mz)
    floor = fit["distance_floor"]
    span = math.sqrt(d02 + floor**2) - floor
    ss = math.sqrt(span**2 / 4 + fit["diversity_standardizer"]["sigma_min"] ** 2)
    zs = span / fit["diversity_standardizer"]["sigma_min"]
    emax = 64 * d
    v0 = mx**2 * threshold / 4
    hf = math.sqrt(v0 / 2)
    rhof = (v0 / 2) / (d02 - v0 / 2)
    delta = 3 * hf**2 / 4 / (math.sqrt(hf**2 + floor**2) + math.sqrt(hf**2 / 4 + floor**2))
    tf = delta / ss
    if dp == 1:
        logomega = (
            math.log(dm["amplitude"])
            - zs
            + math.log(math.expm1(tf))
            - math.log1p(math.exp(-zs))
            - math.log1p(math.exp(-zs + tf))
        )
        omega_scope = "Exact endpoint minimum of unpowered logistic increment"
    else:
        # f'=p A H^(p-1) sigmoid'(u), with each factor bounded from below
        # on [-Z,Z]. Integrating this positive floor over an interval of length t
        # gives an admissible lower omega, not an assertion of the exact minimum.
        powered_floor = min(dm["floor"] ** (dp - 1), (dm["floor"] + dm["amplitude"]) ** (dp - 1))
        logomega = (
            math.log(tf * dp * dm["amplitude"] * powered_floor)
            - zs
            - 2 * math.log1p(math.exp(-zs))
        )
        omega_scope = (
            "Rigorous derivative-floor specialization; omega_lower <= the defined minimum"
        )
    logr = (
        math.log(hf / 2)
        if la == 0
        else min(math.log(hf / 2), math.log(amin) + logomega - math.log(2 * fplus * la))
    )
    loggamma = math.log(amin / 2) + logomega
    loga = min(
        0.0,
        loggamma
        - math.log(
            config["clone_decision"]["saturation"] * (fstar + config["clone_decision"]["epsilon"])
        ),
    )
    logkd = -d02 / (2 * config["distance_donors"]["kernel"]["width"] ** 2)
    logkc = -d02 / (2 * config["cloning_donors"]["kernel"]["width"] ** 2)
    logc = logkc + 2 * logkd + math.log(rhof) + loga
    logside = math.log(2 * bf * math.sqrt(2 * d)) - logr
    logmr = 2 * d * math.log(math.ceil(math.exp(logside)))
    logchi = logc + 2 * math.log(threshold) - math.log(2) - 2 * math.log(emax) - 2 * logmr
    return {
        "D_0": math.sqrt(d02),
        "B_f": bf,
        "B_x": bx,
        "m_x": mx,
        "m_z": mz,
        "D_m": span,
        "s_star": ss,
        "Z_star": zs,
        "Z_R": osc / fit["reward_standardizer"]["sigma_min"],
        "L_R": lr,
        "L_H_upper": lh,
        "L_A_upper": la,
        "A_minus": amin,
        "f_plus": fplus,
        "F_star": fstar,
        "E_max": emax,
        "W_0": threshold,
        "v_0": v0,
        "h_f": hf,
        "rho_f": rhof,
        "Delta_f": delta,
        "t_f": tf,
        "log_kappa_D": logkd,
        "log_kappa_C": logkc,
        "log_omega_lower": logomega,
        "log_r": logr,
        "log_gamma_0": loggamma,
        "log_a_0": loga,
        "log_C_0": logc,
        "log_M_r": logmr,
        "log_chi_star": logchi,
        "omega_scope": omega_scope,
        "N_independent": True,
        "hypotheses": "Entering all-alive actual box/cap certificate only; independent current Gaussian providers, shared global regularizers, actual powered positive maps; no propagated unbounded moment claim",
    }


def comparison(identifier, labels, observed, bound, relation="upper", **extra):
    require(math.isfinite(observed) and math.isfinite(bound), f"Nonfinite comparison {identifier}")
    residual = (
        abs(observed - bound)
        if relation == "equal"
        else observed - bound
        if relation == "upper"
        else bound - observed
    )
    tolerance = 2e-11 * (1 + abs(observed) + abs(bound))
    return {
        "id": identifier,
        "source_labels": labels,
        "observed": observed,
        "bound": bound,
        "relation": relation,
        "residual": residual,
        "tolerance": tolerance,
        "passed": residual <= tolerance,
        "standard_error": 0.0,
        "scope": "Exact finite-input binary64 audit; numerical tolerance is not an interval proof",
        **extra,
    }


def tv(p, q):
    return math.fsum(abs(a - b) for a, b in zip(p, q, strict=True)) / 2


def reweight(p, fitness):
    w = [a * b for a, b in zip(p, fitness, strict=True)]
    total = math.fsum(w)
    return [a / total for a in w]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"Refusing to overwrite immutable derived file {path}")
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(dataset, output, repository):
    require(not output.exists(), "Use a fresh derived output directory")
    output.mkdir(parents=True)
    report_path = dataset / "chapter04-report.json"
    raw = json.loads(report_path.read_text())
    inventory_path = repository / "proof-validation/chapter04_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    source_path = repository / inventory["source_path"]
    require(
        sha(source_path) == inventory["source_sha256"],
        "Refresh source inventory before this audit",
    )
    entries = {
        e["path"]: e
        for e in (
            json.loads(line)
            for line in (dataset / "archive-journal.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    comparisons, cases, exclusions, verified = [], [], [], []
    ledger_paths = []
    for case in raw["cases"]:
        ident = f"case{case['case']}"
        if not all(case["entering_alive"]):
            exclusions.append({
                "case": case["case"],
                "profile": case["profile"],
                "reason": "Complete all-population Keystone uses all-alive input; mandatory revival is checked in existing native/tightened reset ledgers, never imported into this pressure theorem",
            })
            continue
        n, d = case["N"], case["d"]
        config = case["config"]
        constants = keystone_constants(config, case["benchmark"], d)
        points, velocities = case["initial_positions"], case["initial_velocities"]
        require(
            all(abs(z) <= 2 for cloud in points for x in cloud for z in x),
            "Entering box hypothesis failed",
        )
        require(
            all(
                norm2(v) <= config["kinetic"]["velocity_cap"] ** 2
                for cloud in velocities
                for v in cloud
            ),
            "Entering cap hypothesis failed",
        )
        cx, cy = map(centered, points)
        vx, vy = map(centered, velocities)
        errors = list(itertools.starmap(distance2, zip(cx, cy, strict=True)))
        w = math.fsum(errors) / n
        dv = math.fsum(itertools.starmap(distance2, zip(vx, vy, strict=True))) / n
        q_fixed = (
            math.fsum(
                distance2(a, b)
                + distance2(v, z)
                + 0.5
                * math.fsum(
                    (aa - bb) * (vv - zz) for aa, bb, vv, zz in zip(a, b, v, z, strict=True)
                )
                for a, b, v, z in zip(cx, cy, vx, vy, strict=True)
            )
            / n
        )
        comparisons.extend([
            comparison(
                f"{ident}:all-mass-center-bound",
                ["prop-w2-averaged-keystone-constants"],
                w,
                2 * sum(map(variance, points)),
            ),
            comparison(
                f"{ident}:centering-velocity-projection",
                ["prop-w2-averaged-keystone-constants"],
                dv,
                math.fsum(itertools.starmap(distance2, zip(*velocities, strict=True))) / n,
            ),
            comparison(
                f"{ident}:velocity-cap-remainder",
                ["prop-w2-averaged-keystone-constants"],
                dv,
                4 * config["kinetic"]["velocity_cap"] ** 2,
            ),
        ])
        for eta in (0.25, 1.0, 4.0):
            comparisons.append(
                comparison(
                    f"{ident}:Young-{eta}",
                    ["lem-quantitative-keystone-w2"],
                    q_fixed,
                    (1 + eta) * w + (1 + 0.25 / (4 * eta)) * dv,
                    eta=eta,
                    hypotheses="Fixed admissible equal-mass comparison plan, lambda_v=1,b=.5; optimal structural cost is bounded by this plan, no permanent labels",
                )
            )
        c0, chi = math.exp(constants["log_C_0"]), math.exp(constants["log_chi_star"])
        lower = chi * (w - constants["W_0"]) - c0 * constants["E_max"] / n**2
        frame_ledger, pressure_samples, native_rewards = [], [], None
        feature = [
            features(x, v, config["distance_donors"]["distance"])
            for x, v in zip(points, velocities, strict=True)
        ]
        rows = [donor_rows(z, config["cloning_donors"]) for z in feature]
        side_exact = None
        for relative in case["operator_frame_archives"]:
            path = dataset / relative
            entry = entries[relative]
            require(sha(path) == entry["sha256"], f"Source checksum mismatch {path}")
            data = json.loads(gzip.decompress(path.read_bytes()))
            verified.append({
                "path": str(path),
                "sha256": entry["sha256"],
                "frames": len(data["frames"]),
            })
            for frame in data["frames"]:
                require(
                    frame["before"]["positions"] == points
                    and frame["before"]["velocities"] == velocities,
                    "Fixed-state restore assumption failed",
                )
                measured_p, maximum_fitness_error, maximum_pressure_error = [], 0.0, 0.0
                moment_residuals, tv_checks = [], []
                for side in (0, 1):
                    sampled = frame["sampled_fitness"][side]
                    pattern = frame["distance_companions"][side]["indices"]
                    scores, separation, rz, dz = fitness_for_pattern(
                        feature[side], sampled["oriented_reward"], pattern, config
                    )
                    maximum_fitness_error = max(
                        maximum_fitness_error,
                        *(
                            abs(a - b)
                            for name, computed in (
                                ("fitness", scores),
                                ("separation", separation),
                                ("reward_z", rz),
                                ("diversity_z", dz),
                            )
                            for a, b in zip(sampled[name], computed, strict=True)
                        ),
                    )
                    pressure, expected_variance = conditional_copying(
                        points[side], scores, rows[side], config
                    )
                    saved = frame["conditional_moments"][side]
                    maximum_pressure_error = max(
                        maximum_pressure_error,
                        *(
                            abs(a - b)
                            for a, b in zip(
                                pressure, saved["acceptance_probabilities"], strict=True
                            )
                        ),
                    )
                    moment_residuals.append(
                        abs(expected_variance - saved["expected_position_variance"])
                    )
                    measured_p.append(pressure)
                    a, b = rows[side][0], rows[side][1]
                    tv_checks.append((
                        tv(reweight(a, scores), reweight(b, scores)),
                        max(scores) / min(scores) * tv(a, b),
                    ))
                if native_rewards is None:
                    native_rewards = [s["oriented_reward"] for s in frame["sampled_fitness"]]
                    if n <= 4:
                        side_exact = [
                            integrate_measurement(x, v, reward, config)
                            for x, v, reward in zip(
                                points, velocities, native_rewards, strict=True
                            )
                        ]
                q = math.fsum((a + b) * e for a, b, e in zip(*measured_p, errors, strict=True)) / n
                pressure_samples.append(q)
                frame_ledger.append({
                    "repetition": frame["repetition"],
                    "source_archive": relative,
                    "Q": q,
                    "fitness_identity_absolute_residual": maximum_fitness_error,
                    "pressure_identity_absolute_residual": maximum_pressure_error,
                    "conditional_variance_absolute_residuals": moment_residuals,
                    "bounded_reweighting_TV": tv_checks,
                })
        require(len(frame_ledger) == case["samples"], "Missing fixed-state replica")
        # Every raw residual is checked; SEM never hides a primitive identity violation.
        for key in ("fitness_identity_absolute_residual", "pressure_identity_absolute_residual"):
            comparisons.append(
                comparison(
                    f"{ident}:{key}",
                    ["def-w2-averaged-keystone-constants"],
                    max(f[key] for f in frame_ledger),
                    0,
                    "equal",
                    individual_checks=len(frame_ledger),
                )
            )
        comparisons.append(
            comparison(
                f"{ident}:conditional-variance-identity",
                ["lem-w2-normalized-occupation-reset"],
                max(max(f["conditional_variance_absolute_residuals"]) for f in frame_ledger),
                0,
                "equal",
                individual_checks=2 * len(frame_ledger),
            )
        )
        comparisons.append(
            comparison(
                f"{ident}:bounded-reweighting-TV",
                ["lem-w2-bounded-reweighting-tv"],
                max(a - b for f in frame_ledger for a, b in f["bounded_reweighting_TV"]),
                0,
                individual_checks=2 * len(frame_ledger),
                hypotheses="Same strictly positive realized fitness function reweights two probability donor laws over the same physical atoms; a=min fitness,b=max fitness. Not QSD eigenfunction reweighting.",
            )
        )
        eq = (
            math.fsum(
                (a + b) * e
                for a, b, e in zip(
                    side_exact[0]["mean_row_pressure"],
                    side_exact[1]["mean_row_pressure"],
                    errors,
                    strict=True,
                )
            )
            / n
            if side_exact
            else statistics.fmean(pressure_samples)
        )
        se = statistics.stdev(pressure_samples) / math.sqrt(len(pressure_samples))
        pressure_check = comparison(
            f"{ident}:complete-averaged-pressure",
            ["prop-w2-averaged-keystone-constants"],
            eq,
            lower,
            "lower",
            exact_product_integration=bool(side_exact),
            standard_error=0 if side_exact else se,
            sampling_diagnostic_standard_error=se,
            positive_lower_bound=lower > 0,
            hypotheses=constants["hypotheses"],
            scope="Exact complete measurement-law integration and analytic donor/Bernoulli expectation at N4"
            if side_exact
            else "Retained independent fixed-state measurement means; six SEM diagnostic, not finite-sample proof of expectation",
        )
        if side_exact is None:
            pressure_check["passed"] = lower <= eq + 6 * se
        comparisons.append(pressure_check)
        for eta in (0.25, 1.0, 4.0):
            structural_lower = (
                chi / (1 + eta) * q_fixed
                - chi * (constants["W_0"] + (1 + 0.25 / (4 * eta)) * dv / (1 + eta))
                - c0 * constants["E_max"] / n**2
            )
            comparisons.append(
                comparison(
                    f"{ident}:complete-structural-pressure-{eta}",
                    ["lem-quantitative-keystone-w2", "prop-w2-averaged-keystone-constants"],
                    eq,
                    structural_lower,
                    "lower",
                    eta=eta,
                    standard_error=0 if side_exact else se,
                    hypotheses="Actual normalized positional/velocity error and fixed-plan q retained; replacing optimal Vstruct by larger q_fixed strengthens the checked lower bound; same conditioning as complete-averaged-pressure",
                )
            )
        # Check the covering-bound algebra and both low-/high-error branches,
        # without mistaking their frequently zero/negative values for measured decay.
        invcover = math.exp(-constants["log_M_r"])
        coverage = c0 * w * max(0, (n * w / constants["E_max"] * invcover - 1) / (n - 1)) ** 2
        simpler = c0 * w * max(0, w / constants["E_max"] * invcover - 1 / n) ** 2
        cubic = c0 * w**3 * invcover**2 / (2 * constants["E_max"] ** 2) - c0 * w / n**2
        comparisons.extend([
            comparison(
                f"{ident}:coverage-first-reduction",
                ["prop-w2-averaged-keystone-constants"],
                coverage,
                simpler,
                "lower",
            ),
            comparison(
                f"{ident}:coverage-cubic-reduction",
                ["prop-w2-averaged-keystone-constants"],
                simpler,
                cubic,
                "lower",
            ),
        ])
        ledger_path = output / f"{ident}-pressure-ledger.json.gz"
        content = {
            "case": case["case"],
            "N": n,
            "d": d,
            "landscape": case["benchmark"],
            "config": config,
            "origin_law": case["origin_law"],
            "seeds": case["seeds"],
            "constants": constants,
            "W": w,
            "D_v": dv,
            "fixed_plan_structural_cost": q_fixed,
            "pressure_prediction": lower,
            "measured_pressure": eq,
            "exact_side_measurement_integrations": side_exact,
            "frames": frame_ledger,
            "source_archive_paths": case["raw_archives"],
        }
        ledger_path.write_bytes(
            gzip.compress(json.dumps(content, allow_nan=False).encode(), mtime=0)
        )
        ledger_paths.append({
            "path": str(ledger_path),
            "sha256": sha(ledger_path),
            "decoded_bytes": len(gzip.decompress(ledger_path.read_bytes())),
        })
        cases.append({
            "case": case["case"],
            "N": n,
            "d": d,
            "landscape": case["benchmark"],
            "profile": case["profile"],
            "W": w,
            "D_v": dv,
            "E_Q": eq,
            "standard_error": 0 if side_exact else se,
            "lower_bound": lower,
            "lower_bound_positive": lower > 0,
            "log_chi_star": constants["log_chi_star"],
            "exact_measurement_patterns": 2 * len(side_exact[0]["patterns"]) if side_exact else 0,
            "exact_expected_cloning_variance_drifts": [
                s["expected_variance_drift"] for s in side_exact
            ]
            if side_exact
            else None,
            "pressure_ledger": str(ledger_path),
        })
        print(
            f"Chapter 4 completed pressure case {case['case']}: N={n},d={d}, Q={eq:.6g}",
            flush=True,
        )
    # Existing numerical evidence is linked, not relabelled as stronger proof.
    tightening_path = (
        repository
        / "outputs/convergence/chapters04-06-experiments/tightening-20261004/chapter04/report.json"
    )
    tightening = json.loads(tightening_path.read_text())
    evidence = [
        (str(report_path), raw["comparisons"]),
        (str(tightening_path), tightening["comparisons"]),
        (str(output / "report.json"), comparisons),
    ]
    items = []
    remaining = {
        "def-w2-finite-population-qsd-regime": "Native reference primitive/force predicates checked; full one-update lower density, two-update upper density, target minorization and eigenfunction-floor assembly need same-kernel evaluator; dead physical coordinates remain unbounded.",
        "thm-w2-finite-n-conditioned-convergence": "A stationary QSD/eigenfunction is not supplied by pair-error traces. Full primitive coefficient and survivor-law observable comparison required; global existence/uniqueness remains analytic.",
        "cor-w2-reference-fitness-degeneracy": "Primitive reference is valid; rate-dependent exponent/prefactor require full same-kernel minorization/eigenfunction assembly. No finite-N rate is promoted to N-uniform.",
        "cor-w2-force-profile-band": "Native analytic force profile and coercivity predicates can be evaluated. Uniform density/eigenfunction floor across force family needs complete constant assembly.",
        "rem-w2-closed-drift-vxstruct": "Closed positional drift needs signed fitness-pressure/variance/kinetic composition; exact conditional cloning drifts are measured but not a uniform default-feedback certificate.",
        "rem-w2-completed-conservative-alive-law": "Separate bounded-center, weak-selection, weak-viscosity and large-box imported regimes have distinct moment/provider hypotheses. Default nonlinear nu=.3 own-provider attraction remains unresolved; no observed finite trajectory proves it.",
        "rem-structural-dominance": "Illustrative coefficient dominance is conditional on fitted-free verified drift constants; default global drift constants not closed.",
    }
    for item in inventory["formal_items"]:
        label = item["source_label"]
        linked = [
            {
                "report": p,
                "id": c["id"],
                "observed": c["observed"],
                "bound": c["bound"],
                "relation": c.get("relation"),
                "passed": c.get("passed"),
                "scope": c.get("scope"),
                "hypotheses": c.get("hypotheses"),
                "kind": c.get("kind"),
            }
            for p, rows in evidence
            for c in rows
            if label in c.get("source_labels", [])
            and isinstance(c.get("observed"), (float, int))
            and isinstance(c.get("bound"), (float, int))
        ]
        status = (
            "numerical_subestimates_checked"
            if linked
            else "analytic_statement_or_missing_numeric_evaluator"
        )
        items.append({
            "source_label": label,
            "title": item["title"],
            "source_line": item["source_line"],
            "statement": item["statement"],
            "status": status,
            "numeric_evidence": linked,
            "global_claim_empirically_proved": False,
            "remaining_obligation": remaining.get(
                label,
                "Finite numerical checks validate only their recorded applicability. Universal quantifiers, theorem existence/uniqueness and imported hypotheses remain analytic; no source expression alone is credited.",
            ),
        })
    result = {
        "schema_version": 1,
        "chapter": 4,
        "source_path": str(source_path),
        "source_sha256": sha(source_path),
        "inventory_sha256": sha(inventory_path),
        "helper_sha256": sha(Path(__file__)),
        "new_native_steps": 0,
        "source_report": str(report_path),
        "source_report_sha256": sha(report_path),
        "cases": cases,
        "exclusions": exclusions,
        "comparisons": comparisons,
        "statement_ledger": items,
        "source_archive_verification": verified,
        "derived_archive_verification": ledger_paths,
        "summary": {
            "all_alive_cases": len(cases),
            "excluded_revival_cases": len(exclusions),
            "comparisons": len(comparisons),
            "failed": sum(not c["passed"] for c in comparisons),
            "exact_measurement_patterns": sum(c["exact_measurement_patterns"] for c in cases),
            "positive_complete_pressure_bounds": sum(c["lower_bound_positive"] for c in cases),
            "formal_statements": len(items),
            "statements_with_numeric_subestimate_evidence": sum(
                bool(i["numeric_evidence"]) for i in items
            ),
        },
        "scope": "Complete statement-level ledger of current numerical subestimates and explicit remaining obligations; not a claim that every global theorem is empirically proved. N-normalized physical errors, full probability integration for N4, retained Gaussian samples and revived states are preserved in source archives.",
    }
    write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = run(args.dataset, args.output, args.repository)
    print(json.dumps(result["summary"], indent=2))
    raise SystemExit(bool(result["summary"]["failed"]))


if __name__ == "__main__":
    main()
