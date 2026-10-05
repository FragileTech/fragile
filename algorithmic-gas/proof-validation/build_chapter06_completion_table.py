"""Build a source-bound table of retained Chapter 6 numerical estimates."""

import argparse
import hashlib
import json
import operator
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root, output):
    output.mkdir(parents=True, exist_ok=False)
    paths = [
        p
        for p in sorted(root.glob("*/report.json"))
        if p.parent.name not in {"native-displacement-v1", "weak-native-moment-closure"}
    ]
    reports = [(path, json.loads(path.read_text())) for path in paths]
    bindings = [
        {"path": str(path.resolve()), "sha256": digest(path), "summary": value.get("summary", {})}
        for path, value in reports
    ]
    lines = [
        "# Chapter 6 completed numerical estimates",
        "",
        (
            "All swarm sums use probability normalization and retain each law's own seed denominator. "
            "The conditional Gaussian, conservative moment, killed-chain and survivor-law estimates concern distinct laws."
        ),
        "",
        "| Dataset / estimate | Native updates re-used | Checks | Failures | Scope |",
        "|---|---:|---:|---:|---|",
    ]
    for path, value in reports:
        s = value.get("summary", {})
        steps = s.get(
            "retained_complete_updates",
            s.get(
                "complete_native_updates",
                s.get("native_frames", s.get("retained_native_stages", 0)),
            ),
        )
        checks = s.get("exact_conditional_comparisons", s.get("comparisons", 0))
        scope = value.get(
            "scope", value.get("arithmetic_scope", "See each comparison hypothesis.")
        )
        lines.append(
            f"| [{path.parent.name}]({path.resolve()}) | {steps:,} | {checks:,} | {s.get('comparisons_failed', 0)} | {scope} |"
        )
    weak = next(
        ((p, v) for p, v in reports if p.parent.name == "weak-native-moment-tightened"), None
    )
    if weak:
        path, report = weak
        rows = {}
        for profile in report["profiles"]:
            key = (profile["d"], profile["moment_order"])
            closure = profile["closure"]
            refined = profile["refined_closure"]
            root_closure = profile["root_closure"]
            row = {
                "d": key[0],
                "p": key[1],
                "young_rate": closure["moment"]["coefficient"],
                "root_rate": root_closure["root_moment_coefficient"],
                "young_floor": closure["moment"]["invariant_moment_upper"],
                "refined_young_floor": refined["refined_moment"]["invariant_moment_upper"],
                "root_floor": root_closure["invariant_moment_upper"],
                "refined_Cp": root_closure["additive_root_budget"],
                "a_star": closure["source"]["acceptance_upper"],
                "kappa": closure["source"]["kernel_lower_float"],
                "source_coefficient": closure["source"]["current_frame_source_moment_coefficient"],
            }
            if key in rows and row != rows[key]:
                message = (
                    "Claimed N-free constants vary across native population/zone/law profiles."
                )
                raise ValueError(message)
            rows[key] = row
        lines += [
            "",
            (
                "The weak native profile uses h=0.04, gamma=1, cap=2, alpha=0.5, jitter=0.1, "
                "position diffusion=0.1, viscosity=0, both positive exponents=1e-5, and cloning Gaussian width=3. "
                "Actual channel amplitude/floor are 2/0.1; kappa=exp(-32/18)=0.1690133154060661, "
                "a_star=6.08922417204e-5, S=1.0003602807363084. These constants are independent of N "
                "and satisfy the same full-source force/noise hypotheses."
            ),
            "",
            "| d | p | Young Mp coefficient | Root u coefficient | Cp refined | Original Mp floor | Energy/gated-jitter floor | Root-Minkowski Mp floor |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in sorted(rows.values(), key=operator.itemgetter("d", "p")):
            lines.append(
                f"| {row['d']} | {row['p']} | {row['young_rate']:.12g} | {row['root_rate']:.12g} | {row['refined_Cp']:.8g} | {row['young_floor']:.8g} | {row['refined_young_floor']:.8g} | {row['root_floor']:.8g} |"
            )
        lines += [
            "",
            (
                "The root recurrence is u=(E Mp)^(1/p), u_next<=r*u+Cp. Its unrooted moment bound "
                "is (r^n*u0+Cp*(1-r^n)/(1-r))^p; r^p is not asserted as a linear excess rate. "
                "The floors remain conservative bounds and are not measured stationary moments. "
                "The sampled weak run contains zero accepted clones; exact native donor/gate integration "
                "checks positive probabilities separately."
            ),
            "",
            "| Law / zone | p | Final mean completed Mp | Old complete bound | Refined complete bound | Root complete bound | Independent seeds |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        keyed = {
            (c["law_group"], c["moment_order"], c["step"], c["kind"]): c
            for c in report["comparisons"]
        }
        for law, p in sorted({(c["law_group"], c["moment_order"]) for c in report["comparisons"]}):
            step = max(
                c["step"]
                for c in report["comparisons"]
                if c["law_group"] == law and c["moment_order"] == p
            )
            old, ref, sharp = (
                keyed[law, p, step, kind]
                for kind in ("complete", "complete_refined", "complete_root")
            )
            lines.append(
                f"| {law} | {p} | {sharp['observed']:.8g} | {old['bound']:.8g} | {ref['bound']:.8g} | {sharp['bound']:.8g} | {sharp['independent_seed_units']} |"
            )
        (output / "dimension-moment-constants.json").write_text(
            json.dumps(
                {
                    "source": str(path.resolve()),
                    "source_sha256": digest(path),
                    "rows": list(rows.values()),
                },
                indent=2,
            )
            + "\n"
        )
    killed = next(((p, v) for p, v in reports if p.parent.name == "original-killed-moments"), None)
    if killed:
        path, report = killed
        grouped = {}
        for c in report["comparisons"]:
            grouped.setdefault((c["law_group"], c["moment_order"]), []).append(c)
        lines += [
            "",
            "The original native killed-chain completed positional moments decrease as follows. These are moments of the actual alive rows normalized by original N, with each initial law's own 256-seed denominator; they are not transport errors between swarm laws or survival-conditioned moments.",
            "",
            "| Initial law | p | Last retained comparison time | Initial completed moment | Last completed moment | Last / initial |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for (law, order), rows in sorted(grouped.items()):
            first, last = (
                min(rows, key=operator.itemgetter("step")),
                max(rows, key=operator.itemgetter("step")),
            )
            initial = first["mean_before_alive_N_moment"]
            ratio = last["observed"] / initial if initial else None
            lines.append(
                f"| {law} | {order} | {last['step']} | {initial:.9g} | {last['observed']:.9g} | {ratio:.9g} |"
            )
        lines += [
            "",
            "N2 hazard probes terminate early after absorption, so their final zero moment concerns the killed observable; it is not evidence of mixing among surviving laws. The N4/16/64 d1/2 rows compare time0 with completed time128.",
        ]
    displacement = next(
        ((p, v) for p, v in reports if p.parent.name == "native-displacement-v2"), None
    )
    if displacement:
        path, report = displacement
        by_dimension = {}
        for row in report["operands"]:
            by_dimension.setdefault((row["d"], row["N"]), []).append(row)
        lines += [
            "",
            "The standard native position budget retains its actual prepared first force/viscosity kick: Dh²=c_h² mean(||v1||²)+d tau². The terminal velocity cap does not bound this transient kick. The centered budget removes common displacement and has noise source d tau²(1−1/N).",
            "",
            "| d | N | Prepared stages | Mean raw Dh² | Mean centered Dh² | Largest raw Dh² |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for (d, n), rows in sorted(by_dimension.items()):
            count = len(rows)
            raw = sum(r["raw_displacement_budget"] for r in rows) / count
            centered = sum(r["centered_displacement_budget"] for r in rows) / count
            largest = max(r["raw_displacement_budget"] for r in rows)
            lines.append(f"| {d} | {n} | {count:,} | {raw:.9g} | {centered:.9g} | {largest:.9g} |")
        lines += [
            "",
            "These prepared-state means are diagnostics; they are not uniform affine displacement constants. A positional contraction coefficient requires a proved P_C Dh²≤a_D Vx+b_D and includes the full coefficient (1+theta)r_C+(1+1/theta)a_D. Squared realized displacement was not reconstructed from missing cross moments.",
        ]
    tails = next(
        ((p, v) for p, v in reports if p.parent.name == "native-tail-subestimates-v1"), None
    )
    if tails:
        path, report = tails
        lines += [
            "",
            "| Native barrier / tail subestimate | Numerical checks | Largest signed residual | Scope |",
            "|---|---:|---:|---|",
        ]
        labels = sorted({label for c in report["comparisons"] for label in c["source_labels"]})
        for label in labels:
            rows = [c for c in report["comparisons"] if label in c["source_labels"]]
            scopes = sorted({c["scope"] for c in rows})
            lines.append(
                f"| `{label}` | {len(rows):,} | {max(c['signed_residual'] for c in rows):.9g} | "
                + " ".join(scopes)
                + " |"
            )
        lines += [
            "",
            "The barrier uses own original independent seed denominators, with absorbed trajectories zero. Gaussian tail probabilities use the actual prepared conditional means and covariance; finite-row moment tails validate their finite marginal law only. None supplies a global stationary/QSD moment from samples.",
        ]
    lines += [
        "",
        (
            "Finite ensemble moment comparisons use family-adjusted Cantelli uncertainty from normalized 2p bounds; "
            "Gaussian concentration uses fixed checkpoints and independent seeds. Neither test supplies joint stationary/QSD "
            "LSI, a uniform QSD two-sided block, eigenfunction infimum, or default strong-selection moment absorption. "
            "The default multiwell slow-zone endpoints can increase; regional hypotheses and additive defects are retained."
        ),
        "",
        "The full formal statement ledger retains those obligations and every exact displayed formula under its own hypotheses.",
    ]
    (output / "chapter06-estimates-table.md").write_text("\n".join(lines) + "\n")
    (output / "provenance.json").write_text(
        json.dumps(
            {"reports": bindings, "executed_helper_sha256": digest(Path(__file__))}, indent=2
        )
        + "\n"
    )
    (output / "executed-helper.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps({
            "reports": len(bindings),
            "table": str(output / "chapter06-estimates-table.md"),
        })
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.root, args.output)
