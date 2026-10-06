"""Independent distance and marked-position variation checks on retained Rust inputs.

The finite native companion law excludes self. The separate atomic population
reference includes self, as the chapter's integral kernel does. Neither experiment
identifies a finite-particle law with a QSD or certifies a global analytic theorem.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def normed(weights: np.ndarray) -> np.ndarray:
    return weights / weights.sum()


def kernel(weights: np.ndarray, law: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    numerator = weights * law[None, :]
    z = numerator.sum(axis=1)
    if np.min(z) <= 0:
        msg = "Strictly positive companion denominators required"
        raise ValueError(msg)
    return numerator / z[:, None], z


def moments(joint: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    mean = float(np.sum(joint * values))
    variance = float(np.sum(joint * (values - mean) ** 2))
    return mean, variance


def constants(params: dict, high_precision: bool = False) -> dict:
    """Two separately evaluated dependency chains, including zero exponents."""
    if high_precision:
        with localcontext() as ctx:
            ctx.prec = 70
            p = {k: Decimal(str(v)) for k, v in params.items()}
            two, four = Decimal(2), Decimal(4)

            def exp(v):
                return v.exp()

            def log(v):
                return v.ln()

            return _constants(p, two, four, exp, log)
    return _constants(params, 2.0, 4.0, np.exp, np.log)


def _constants(p, two, four, exp, log):
    gr, gd = p["etaR"] + p["AR"], p["etaD"] + p["AD"]
    tr = (
        p["AR"] * p["pR"] / four * max(p["etaR"] ** (p["pR"] - 1), gr ** (p["pR"] - 1))
        if p["pR"]
        else p["pR"]
    )
    td = (
        p["AD"] * p["pD"] / four * max(p["etaD"] ** (p["pD"] - 1), gd ** (p["pD"] - 1))
        if p["pD"]
        else p["pD"]
    )
    lm = two + two / p["kD"]
    qr = p["R"] / p["sigmaR"] + 3 * p["R"] ** 3 / (two * p["sigmaR"] ** 3)
    qd = lm * (p["D"] / p["sigmaD"] + 3 * p["D"] ** 3 / (two * p["sigmaD"] ** 3))
    lf = tr * gd ** p["pD"] * qr + td * gr ** p["pR"] * qd
    low = p["etaR"] ** p["pR"] * p["etaD"] ** p["pD"]
    high = gr ** p["pR"] * gd ** p["pD"]
    la = max(
        1 / (p["sa"] * (low + p["epsa"])), (high + p["epsa"]) / (p["sa"] * (low + p["epsa"]) ** 2)
    )
    c = 1 / (p["kC"] * p["mstar"])
    lb = two * c**2 + two * c * la * lf
    b = two * c * lm + lb
    # Stable logarithm retains the full value even when the exponential overflows.
    log_step = log(two) + two * c + log(two * b + lm * exp(-two * c))
    lp = two * (lm * (1 + two / p["kC"]) + two * la * lf)
    return {
        "GR": gr,
        "GD": gd,
        "TR": tr,
        "TD": td,
        "LM": lm,
        "QR": qr,
        "QD": qd,
        "LF": lf,
        "Fstar": low,
        "Fmax": high,
        "La": la,
        "C": c,
        "Lbeta": lb,
        "B": b,
        "Lstep": exp(log_step),
        "logLstep": log_step,
        "Lpos": lp,
    }


def fitness(
    reward: np.ndarray, distance: np.ndarray, alive: np.ndarray, kd: np.ndarray, p: dict
) -> np.ndarray:
    mean_r = float(alive @ reward)
    var_r = float(alive @ (reward - mean_r) ** 2)
    pair = alive[:, None] * kd
    mean_d, var_d = moments(pair, distance)
    rr = (reward - mean_r) / np.sqrt(var_r + p["sigmaR"] ** 2)
    dd = (distance - mean_d) / np.sqrt(var_d + p["sigmaD"] ** 2)
    gr = p["etaR"] + p["AR"] / (1 + np.exp(-rr))
    gd = p["etaD"] + p["AD"] / (1 + np.exp(-dd))
    return gr[:, None] ** p["pR"] * gd ** p["pD"]


def position_gate_law(
    mu: np.ndarray,
    alive: np.ndarray,
    marks: np.ndarray,
    kd: np.ndarray,
    kc: np.ndarray,
    f: np.ndarray,
    p: dict,
) -> np.ndarray:
    """Exactly integrate recipient/donor sampled fitness, then the clipped gate.

    X is the unchanged frozen source on rejection and the donor source on an
    accepted cloning/revival. J is exactly that recipient-jitter gate. Incoming
    collision vertices cannot alter this frozen source position.
    """
    n = len(mu)
    out = np.zeros((n, 2))
    for i in range(n):
        for j in range(n):
            if alive[j] == 0:
                continue
            if marks[i]:
                acceptance = np.clip(
                    (f[j][None, :] - f[i][:, None]) / (p["sa"] * (f[i][:, None] + p["epsa"])), 0, 1
                )
                gate = float(np.sum(kd[i][:, None] * kd[j][None, :] * acceptance))
            else:
                gate = 1.0
            mass = mu[i] * kc[i, j]
            out[i, 0] += mass * (1 - gate)
            out[j, 1] += mass * gate
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    inventory_path = root / "proof-validation/chapter09_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    source = root / inventory["source_path"]
    if sha(source) != inventory["source_sha256"]:
        msg = "The expression inventory must match the current source exactly"
        raise ValueError(msg)
    expressions = {e["id"]: e for e in inventory["quantitative_expressions"]}
    index = json.loads((args.native / "archive-index.json").read_text())
    checks, cases, archives = [], [], []

    def check(name, ids, lhs, rhs, scope, operands, equality=False):
        lhs, rhs = float(lhs), float(rhs)
        tolerance = 2e-11 * max(1, abs(lhs), abs(rhs))
        error = abs(lhs - rhs) if equality else lhs - rhs
        checks.append({
            "id": name,
            "source_expressions": [f"chapter09-expression-{i:04d}" for i in ids],
            "lhs": lhs,
            "rhs": rhs,
            "allowance": tolerance,
            "relation": "equality" if equality else "upper_bound",
            "passed": bool(error <= tolerance),
            "scope": scope,
            "operands": operands,
        })

    scope_distance = "Exact weighted distance laws on retained native Rust frozen physical coordinates; perturbation of probabilities on fixed physical support."
    scope_position = "Exact atomic full marked input law and recipient/donor measurement integration of the population frozen-source/jitter pair; collision-component/full kinetic population law remains analytic."
    for entry in index["entries"]:
        if not entry["tag"].endswith("native-operands"):
            continue
        path = args.native / entry["path"]
        if sha(path) != entry["sha256"]:
            raise ValueError(f"Archive SHA mismatch: {path}")
        data = json.loads(gzip.decompress(path.read_bytes()))
        frozen = data["frozen_input"]
        cfg = data["configuration"]["gas"]
        fields = frozen["observations"]["fields"]
        n = len(frozen["validity"])
        d = fields["positions"]["item_shape"][0]
        x = np.array(fields["positions"]["values"]).reshape(n, d)
        v = np.array(fields["velocities"]["values"]).reshape(n, d)
        alive = np.array([not any(s.values()) for s in frozen["validity"]])
        alive_indices = np.flatnonzero(alive)
        k = len(alive_indices)
        ds = cfg["distance_donors"]["distance"]
        width = cfg["distance_donors"]["kernel"]["width"]
        sx = x / (1 + np.linalg.norm(x, axis=1) / ds["position_radius"])[:, None]
        sv = v / (1 + np.linalg.norm(v, axis=1) / ds["velocity_radius"])[:, None]
        squared = np.sum((sx[:, None, :] - sx[None, :, :]) ** 2, axis=2)
        squared += ds["lambda"] * np.sum((sv[:, None, :] - sv[None, :, :]) ** 2, axis=2)
        distance = np.sqrt(squared + cfg["fitness"]["distance_floor"] ** 2)
        weights = np.exp(-squared / (2 * width**2))
        rho = alive.astype(float) / k
        rewards = np.array(frozen["rewards"]["raw"])
        tilt = np.exp(
            0.2 * (rewards - rewards.min()) / max(1, np.ptp(rewards))
            + 0.1 * np.linalg.norm(x, axis=1) / max(1, np.max(np.linalg.norm(x, axis=1)))
        )
        target = normed(tilt * alive)
        case = {
            "tag": entry["tag"],
            "archive": str(path.resolve()),
            "archive_sha256": entry["sha256"],
            "native_config": cfg,
            "physical_x": x.tolist(),
            "physical_v": v.tolist(),
            "marks": alive.tolist(),
            "native_raw_reward": rewards.tolist(),
            "squared_algorithmic_distance": squared.tolist(),
            "diversity_distance": distance.tolist(),
            "actual_gaussian_weights": weights.tolist(),
            "native_input_archive_verified": True,
            "probability_perturbations": [],
            "position_cases": [],
        }
        tag = entry["tag"].removesuffix("-native-operands")
        # Confirm reconstruction against every exact native finite normalizer retained by Rust.
        original = {c["id"]: c for c in data["checks"] if c["id"].startswith("self-exclusion-")}
        for i in alive_indices:
            z = float(weights[i, alive].sum() - weights[i, i])
            check(
                f"{tag}/native-normalizer-{i}",
                [106],
                z / k,
                original[f"self-exclusion-{i}"]["operands"]["nonself_normalizer"] / k,
                "Actual native Rust finite self-excluded companion normalizer reconstruction; prerequisite, not an independent source theorem.",
                {"alive_count": k, "recipient": int(i), "original_check": f"self-exclusion-{i}"},
                True,
            )
        for excluded in (False, True):
            w = weights.copy()
            if excluded:
                np.fill_diagonal(w, 0)
            kr, zr = kernel(w, rho)
            jr = rho[:, None] * kr
            mr, vr = moments(jr, distance)
            for eps in (0.001, 0.01, 0.1, 0.5, 1):
                eta = (1 - eps) * rho + eps * target
                ke, ze = kernel(w, eta)
                je = eta[:, None] * ke
                me, ve = moments(je, distance)
                delta = float(np.sum(abs(rho - eta)))
                a = float(min(np.min(zr), np.min(ze)))
                md = float(np.max(distance))
                row = {
                    "self_excluded": excluded,
                    "perturbation": eps,
                    "rho": rho.tolist(),
                    "eta": eta.tolist(),
                    "denominator_floor": a,
                    "distance_bound": md,
                    "norm_difference": delta,
                    "mean_rho": mr,
                    "mean_eta": me,
                    "variance_rho": vr,
                    "variance_eta": ve,
                    "mean_constant": md * (1 + 2 / a),
                    "variance_constant": 3 * md**2 * (1 + 2 / a),
                }
                case["probability_perturbations"].append(row)
                prefix = f"{tag}/exclude{excluded}/eps{eps}"
                check(
                    prefix + "/mean",
                    [115],
                    abs(mr - me),
                    row["mean_constant"] * delta,
                    scope_distance,
                    row,
                )
                check(
                    prefix + "/variance",
                    [114],
                    abs(vr - ve),
                    row["variance_constant"] * delta,
                    scope_distance,
                    row,
                )
                check(
                    prefix + "/joint-probability-rho",
                    [114, 115],
                    jr.sum(),
                    1,
                    scope_distance,
                    row,
                    True,
                )
                check(
                    prefix + "/joint-probability-eta",
                    [114, 115],
                    je.sum(),
                    1,
                    scope_distance,
                    row,
                    True,
                )
                check(
                    prefix + "/joint-variation",
                    [113],
                    np.sum(abs(jr - je)),
                    (1 + 2 / a) * delta,
                    scope_distance,
                    row,
                )
                check(
                    prefix + "/kernel-variation",
                    [113],
                    np.max(np.sum(abs(kr - ke), axis=1)),
                    (2 / a) * delta,
                    scope_distance,
                    row,
                )
                # Simultaneously permute every physical field and both laws: no labeled-swarm observable.
                perm = np.arange(n)[::-1]
                jp = eta[perm, None] * kernel(w[np.ix_(perm, perm)], eta[perm])[0]
                mp, vp = moments(jp, distance[np.ix_(perm, perm)])
                check(prefix + "/mean-permutation", [115], mp, me, scope_distance, row, True)
                check(prefix + "/variance-permutation", [114], vp, ve, scope_distance, row, True)
                if not excluded:
                    rep = np.repeat(np.arange(n), 2)
                    doubled = np.repeat(eta / 2, 2)
                    jd = doubled[:, None] * kernel(w[np.ix_(rep, rep)], doubled)[0]
                    mm, vv = moments(jd, distance[np.ix_(rep, rep)])
                    check(
                        prefix + "/same-population-2N-mean",
                        [115],
                        mm,
                        me,
                        scope_distance,
                        row,
                        True,
                    )
                    check(
                        prefix + "/same-population-2N-variance",
                        [114],
                        vv,
                        ve,
                        scope_distance,
                        row,
                        True,
                    )
        # Exact frozen-position law needs four sampled type indices; run on each d/profile N8 input.
        if n == 8:
            dead = ~alive
            if not dead.any():
                # Explicitly declared reference marking on the same original physical coordinates.
                dead = np.arange(n) % 3 == 0
                alive = ~dead
            aidx = np.flatnonzero(alive)
            rho_a = normed(alive.astype(float))
            rho_d = normed(dead.astype(float))
            zeta_a, zeta_d = normed(tilt * alive), normed((2 - tilt) * dead)
            # Reward is shifted/directed as in standardization; shift has no effect on fitness.
            reward = -rewards if cfg["fitness"]["direction"] == "minimize" else rewards.copy()
            reward -= reward.min()
            maxsq = 4 * ds["position_radius"] ** 2 + 4 * ds["lambda"] * ds["velocity_radius"] ** 2
            lower_d = float(np.exp(-maxsq / (2 * width**2)))
            width_c = cfg["cloning_donors"]["kernel"]["width"]
            wc = np.exp(-squared / (2 * width_c**2))
            lower_c = float(np.exp(-maxsq / (2 * width_c**2)))
            # Conservative reward oscillation on a box containing all original alive coordinates.
            box_radius = max(1, float(np.max(abs(x[aidx]))))
            rbound = d * box_radius**2 + 20 * d if "rastrigin" in tag else d * box_radius**2 / 2
            fc = cfg["fitness"]
            basep = {
                "etaR": fc["reward_map"]["floor"],
                "AR": fc["reward_map"]["amplitude"],
                "etaD": fc["diversity_map"]["floor"],
                "AD": fc["diversity_map"]["amplitude"],
                "pR": fc["reward_exponent"],
                "pD": fc["diversity_exponent"],
                "sigmaR": fc["reward_standardizer"]["sigma_min"],
                "sigmaD": fc["diversity_standardizer"]["sigma_min"],
                "kD": lower_d,
                "kC": lower_c,
                "R": rbound,
                "D": float(np.sqrt(maxsq + fc["distance_floor"] ** 2)),
                "sa": cfg["clone_decision"]["saturation"],
                "epsa": cfg["clone_decision"]["epsilon"],
                "mstar": 0.5,
            }
            for exponents in ((0, 0), (0.5, 2), (1, 1), (2, 0.5)):
                p = dict(basep, pR=exponents[0], pD=exponents[1])
                cv = constants(p)
                precise = constants(p, True)
                for name, value in cv.items():
                    ids = (
                        [770]
                        if name
                        in {"GR", "GD", "TR", "TD", "LM", "QR", "QD", "LF", "Fstar", "Fmax"}
                        else [775]
                    )
                    if name == "Lpos":
                        ids = [777]
                    check(
                        f"{tag}/p{exponents}/constant-{name}",
                        ids,
                        value,
                        float(precise[name]),
                        "Binary64 explicit source parameter chain versus independently evaluated 70-digit Decimal chain; Lstep represented by its logarithm.",
                        {"parameters": p, "high_precision": str(precise[name])},
                        True,
                    )
                for suffix, eta0, amplitude, power, upper in (
                    ("R", p["etaR"], p["AR"], p["pR"], cv["TR"]),
                    ("D", p["etaD"], p["AD"], p["pD"], cv["TD"]),
                ):
                    sigmoid = 1 / (1 + np.exp(-np.linspace(-12, 12, 257)))
                    derivative = (
                        power
                        * amplitude
                        * sigmoid
                        * (1 - sigmoid)
                        * (eta0 + amplitude * sigmoid) ** (power - 1)
                        if power
                        else np.zeros_like(sigmoid)
                    )
                    check(
                        f"{tag}/p{exponents}/logistic-derivative-{suffix}",
                        [770],
                        np.max(abs(derivative)),
                        upper,
                        scope_position,
                        {
                            "eta": eta0,
                            "amplitude": amplitude,
                            "power": power,
                            "standardized_grid": [-12, 12, 257],
                        },
                    )
                for m, mass_n in ((0.6, 0.7), (0.95, 0.96), (1.0, 1.0)):
                    mu = m * rho_a + (1 - m) * rho_d
                    eta = mass_n * zeta_a + (1 - mass_n) * zeta_d
                    delta = (
                        abs(m - mass_n) + np.sum(abs(rho_a - zeta_a)) + np.sum(abs(rho_d - zeta_d))
                    )
                    kda = kernel(weights, rho_a)[0]
                    kdb = kernel(weights, zeta_a)[0]
                    kca = kernel(wc, rho_a)[0]
                    kcb = kernel(wc, zeta_a)[0]
                    fa = fitness(reward, distance, rho_a, kda, p)
                    fb = fitness(reward, distance, zeta_a, kdb, p)
                    law_a = position_gate_law(mu, rho_a, alive, kda, kca, fa, p)
                    law_b = position_gate_law(eta, zeta_a, alive, kdb, kcb, fb, p)
                    difference = float(np.sum(abs(law_a - law_b)))
                    row = {
                        "parameters": p,
                        "constants": {key: float(val) for key, val in cv.items()},
                        "constants_70digit": {key: str(val) for key, val in precise.items()},
                        "masses": [m, mass_n],
                        "alive_mark_reference": alive.tolist(),
                        "conditional_alive_rho": rho_a.tolist(),
                        "conditional_alive_eta": zeta_a.tolist(),
                        "conditional_dead_rho": rho_d.tolist(),
                        "conditional_dead_eta": zeta_d.tolist(),
                        "delta": float(delta),
                        "shifted_directed_reward": reward.tolist(),
                        "box_radius": box_radius,
                        "fitness_rho": fa.tolist(),
                        "fitness_eta": fb.tolist(),
                        "frozen_position_jitter_law_rho": law_a.tolist(),
                        "frozen_position_jitter_law_eta": law_b.tolist(),
                        "exact_pair_variation": difference,
                        "upper": cv["Lpos"] * delta,
                        "native_exponents": exponents == (1, 1),
                        "reference_self_inclusion": True,
                        "full_population_collision_stability": "analytic exploration obligation; only constants evaluated here",
                    }
                    case["position_cases"].append(row)
                    prefix = f"{tag}/p{exponents}/m{m}-{mass_n}"
                    check(
                        prefix + "/position-gate",
                        [777],
                        difference,
                        row["upper"],
                        scope_position,
                        row,
                    )
                    check(
                        prefix + "/probability-a", [777], law_a.sum(), 1, scope_position, {}, True
                    )
                    check(
                        prefix + "/probability-b", [777], law_b.sum(), 1, scope_position, {}, True
                    )
                    check(
                        prefix + "/marked-input",
                        [779],
                        np.sum(abs(mu - eta)),
                        2 * delta,
                        scope_position,
                        row,
                    )
                    check(
                        prefix + "/fitness",
                        [770],
                        np.max(abs(fa[aidx] - fb[aidx])),
                        cv["LF"] * delta,
                        scope_position,
                        row,
                    )
                    check(
                        prefix + "/fitness-low",
                        [770],
                        cv["Fstar"],
                        min(fa.min(), fb.min()),
                        scope_position,
                        {},
                    )
                    check(
                        prefix + "/fitness-high",
                        [770],
                        max(fa.max(), fb.max()),
                        cv["Fmax"],
                        scope_position,
                        {},
                    )
                    check(
                        prefix + "/reward-oscillation",
                        [770],
                        np.ptp(reward[aidx]),
                        p["R"],
                        scope_position,
                        {},
                    )
                    for i in aidx:
                        for j in aidx:
                            gate_a = np.clip(
                                (fa[j][None, :] - fa[i][:, None])
                                / (p["sa"] * (fa[i][:, None] + p["epsa"])),
                                0,
                                1,
                            )
                            gate_b = np.clip(
                                (fb[j][None, :] - fb[i][:, None])
                                / (p["sa"] * (fb[i][:, None] + p["epsa"])),
                                0,
                                1,
                            )
                            check(
                                prefix + f"/acceptance-{i}-{j}",
                                [775],
                                np.max(abs(gate_a - gate_b)),
                                2 * cv["La"] * cv["LF"] * delta,
                                scope_position,
                                {
                                    "parameters": p,
                                    "delta": float(delta),
                                    "recipient": int(i),
                                    "donor": int(j),
                                    "La": cv["La"],
                                    "LF": cv["LF"],
                                },
                            )
                    perm = np.arange(n)[::-1]
                    law_p = position_gate_law(
                        eta[perm],
                        zeta_a[perm],
                        alive[perm],
                        kdb[np.ix_(perm, perm)],
                        kcb[np.ix_(perm, perm)],
                        fb[np.ix_(perm, perm)],
                        p,
                    )
                    check(
                        prefix + "/position-permutation",
                        [777],
                        np.max(abs(law_p - law_b[perm])),
                        0,
                        scope_position,
                        {},
                        True,
                    )
                    if exponents == (1, 1) and m < 0.7:
                        rep = np.repeat(np.arange(n), 2)
                        law_2n = position_gate_law(
                            np.repeat(eta / 2, 2),
                            np.repeat(zeta_a / 2, 2),
                            alive[rep],
                            kdb[np.ix_(rep, rep)] / 2,
                            kcb[np.ix_(rep, rep)] / 2,
                            fb[np.ix_(rep, rep)],
                            p,
                        )
                        aggregated = law_2n.reshape(n, 2, 2).sum(axis=1)
                        check(
                            prefix + "/identical-atomic-population-2N",
                            [777],
                            np.max(abs(aggregated - law_b)),
                            0,
                            scope_position,
                            {},
                            True,
                        )
        archive = args.output / f"{tag}.json.gz"
        encoded = json.dumps(case, separators=(",", ":"), allow_nan=False).encode()
        archive.write_bytes(gzip.compress(encoded, mtime=0))
        if json.loads(gzip.decompress(archive.read_bytes())) != case:
            msg = "Lossless read-back mismatch"
            raise ValueError(msg)
        archives.append({
            "path": archive.name,
            "sha256": sha(archive),
            "compressed_bytes": archive.stat().st_size,
            "decoded_bytes": len(encoded),
            "deep_verified": True,
        })
        cases.append({
            "tag": tag,
            "N": n,
            "d": d,
            "distance_perturbations": len(case["probability_perturbations"]),
            "marked_position_laws": len(case["position_cases"]),
            "native_source_sha256": entry["sha256"],
        })
    for name, path in (
        ("execution-helper.py", Path(__file__)),
        ("chapter09-source.md", source),
        ("chapter09-inventory.json", inventory_path),
    ):
        original = path.read_bytes()
        snapshot = args.output / (name + ".gz")
        snapshot.write_bytes(gzip.compress(original, mtime=0))
        if gzip.decompress(snapshot.read_bytes()) != original:
            msg_0 = "Source snapshot byte-for-byte verification failed"
            raise ValueError(msg_0)
        archives.append({
            "path": snapshot.name,
            "sha256": sha(snapshot),
            "compressed_bytes": snapshot.stat().st_size,
            "decoded_bytes": len(original),
            "deep_verified": True,
            "source_sha256": hashlib.sha256(original).hexdigest(),
        })
    ids = sorted({i for c in checks for i in c["source_expressions"]})
    ledger = [
        {
            "source_expression_id": i,
            "exact_formula": expressions[i]["formula"],
            "source_line": expressions[i]["source_line"],
            "source_sha256": sha(source),
            "check_ids": [c["id"] for c in checks if i in c["source_expressions"]],
            "status": "finite_operator_denominator_contract"
            if i.endswith("0106")
            else "finite_conditional_distance_or_position_law_comparison",
            "scope": scope_position if int(i[-4:]) >= 770 else scope_distance,
        }
        for i in ids
    ]
    report = {
        "schema_version": 1,
        "source_sha256": sha(source),
        "helper_sha256": sha(Path(__file__)),
        "inventory_sha256": sha(inventory_path),
        "native_archive_index_sha256": sha(args.native / "archive-index.json"),
        "checks": checks,
        "check_count": len(checks),
        "failures": [c["id"] for c in checks if not c["passed"]],
        "cases": cases,
        "expression_ledger": ledger,
        "archive_manifest": archives,
        "new_native_rust_updates": 0,
        "retained_native_frozen_inputs": len(cases),
        "analytic_obligations": [
            "Full marked collision-component population-map stability and global law hypotheses remain analytic; only explicit constants and frozen source/jitter pair are evaluated.",
            "Atomic parameter perturbations are deterministic probability experiments, not additional native Rust simulations or QSD draws.",
        ],
    }
    dump(args.output / "report.json", report)
    dump(args.output / "expression-ledger.json", ledger)
    dump(args.output / "archive-index.json", {"status": "complete", "entries": archives})
    (args.output / "helper.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps({
            "checks": len(checks),
            "failures": report["failures"],
            "inputs": len(cases),
            "expressions": ids,
            "output": str(args.output.resolve()),
        })
    )
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
