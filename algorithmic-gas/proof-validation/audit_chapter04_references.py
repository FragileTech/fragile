"""Check every saved dense-reference force residual and rebuild its constants."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def primitive(config, n, d):
    kinetic = config["kinetic"]
    integrator = kinetic["integrator"]
    assert integrator["kind"] == "baoab"
    assert kinetic["noise"]["innovation"] == "gaussian"
    assert kinetic["noise"]["geometry"]["scale"]["values"] == [1.0]
    boundary = config["boundary"]["domain"]
    assert boundary["upper"] == [2.0] * d
    assert boundary["lower"] == [-2.0] * d
    viscosity = config["qft"]["viscosity"]
    clone = config["clone_transform"]
    assert clone["jitter"]["innovation"] == "gaussian"
    assert clone["jitter"]["geometry"]["scale"]["values"] == [1.0]
    h = integrator["dt"]
    t = h / 2
    gamma = integrator["friction"]
    decay = math.exp(-gamma * h)
    q2 = -math.expm1(-2 * gamma * h) / (2 * gamma) if gamma else h
    s = kinetic["position_diffusion"] * math.sqrt(h)
    sj = clone["jitter_amplitude"]
    vmax = kinetic["velocity_cap"]
    vc = (1 + 2 * abs(clone["restitution"])) * vmax
    rd = 2 * math.sqrt(d)
    nu = viscosity["coefficient"]
    rho = viscosity["bandwidth"]
    row = viscosity["row_normalized"]
    b1 = (1 + 2 * t * nu) * vc + t * (rd + sj)
    a = 2 + sj + t * (1 + decay) * b1
    sigma = math.sqrt(t * t * q2 + s * s)
    z = (a - 2) / sigma + 1 / 13
    # This reference has d=3. It uses a Gaussian density lower bound on the
    # unit ball and a coordinate interval, avoiding tiny CDF subtraction.
    assert n == 200 and d == 3
    log_p0 = math.log(4 * math.pi / 3) - 1.5 * math.log(math.tau) - 0.5
    log_a = log_p0 + d * (-0.5 * z * z - 0.5 * math.log(math.tau) - math.log(13))
    radius = math.sqrt(2 * d * (math.log(12 * d * n) - 2 * log_a))
    jitter_radius = sj * radius
    cx = (
        16 * nu * vc * (rd + jitter_radius) / (rho * rho)
        if row
        else 4 * nu * vc * math.exp(-0.5) / rho
    )
    return {
        "h": h,
        "t": t,
        "gamma": gamma,
        "c": decay,
        "q_squared": q2,
        "s": s,
        "sigma_J": sj,
        "V": vmax,
        "V_c": vc,
        "nu": nu,
        "rho": rho,
        "L_F": 1.0,
        "B_F": 0.0,
        "R_D": rd,
        "B_1": b1,
        "A": a,
        "sigma_h": sigma,
        "log_survival_floor_lower": log_a,
        "r_star_from_survival_lower": radius,
        "J": jitter_radius,
        "C_x": cx,
        "kappa_F": 1 - t * t - (2 if row else 1) * t * nu,
        "beta_F": t * t * (1 + cx),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--presentation-output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.dataset / "chapter04-report.json").read_text())
    journal = [
        json.loads(s)
        for s in (args.dataset / "archive-journal.jsonl").read_text().splitlines()
        if s.strip()
    ]
    by_path = {e["path"]: e for e in journal}
    comparisons = []
    primitive_rows = []
    provenance = []
    for reference in report["references"]:
        path = reference["force_ledger"]
        data = (args.dataset / path).read_bytes()
        entry = by_path[path]
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        provenance.append(entry)
        ledger = json.loads(gzip.decompress(data))
        row = reference["row_normalized"]
        mode = "row" if row else "count"
        prefix = f"reference-{str(row).lower()}"
        config = reference["native_config"]
        rebuilt = primitive(config, reference["N"], reference["d"])
        primitive_rows.append({"mode": mode, "recomputed": rebuilt})
        values = {}

        def append(name, observed, bound, relation="equal"):
            if not (math.isfinite(observed) and math.isfinite(bound)):
                raise ValueError(f"Nonfinite {name}")
            values.setdefault((name, relation), []).append((observed, bound))

        assert len(ledger["force_frames"]) == reference["steps"]
        assert [f["step"] for f in ledger["force_frames"]] == list(range(reference["steps"]))
        for frame in ledger["force_frames"]:
            assert [c["stage"] for c in frame["force_checks"]] == ["B1", "B2"]
            for check in frame["force_checks"]:
                append("actual_dense_force", check["viscous_formula_residual"], 0.0)
                append("actual_harmonic_force", check["harmonic_force_residual"], 0.0)
                append(
                    "actual_dense_edges",
                    check["influence_edges"],
                    check["expected_dense_influence_edges"],
                )
                append(
                    "all_alive_dense_edge_count",
                    check["expected_dense_influence_edges"],
                    reference["N"] * (reference["N"] - 1),
                )
                if not row:
                    append("count_momentum_conservation", check["summed_viscous_force_norm"], 0.0)
        for key, expected in rebuilt.items():
            append(f"primitive:{key}", ledger["primitive"][key], expected)
        append("B2_coercivity", rebuilt["kappa_F"], 0.9876 if row else 0.9936)
        append("beta_F_margin", rebuilt["beta_F"], 0.071 if row else 0.002, "upper")
        for (name, relation), pairs in values.items():

            def residual(pair):
                return abs(pair[0] - pair[1]) if relation == "equal" else pair[0] - pair[1]

            witness = max(pairs, key=residual)
            tolerance = 2e-10 * (1 + max(max(abs(a), abs(b)) for a, b in pairs))
            comparisons.append({
                "id": f"{prefix}:{name}",
                "mode": mode,
                "observed": witness[0],
                "bound": witness[1],
                "relation": relation,
                "samples": len(pairs),
                "standard_error": 0.0,
                "maximum_residual": residual(witness),
                "passed": all(residual(p) <= tolerance for p in pairs),
                "kind": "every_retained_sample_exact",
                "hypotheses": [
                    "Actual native reference parameters and Gaussian draws retained",
                    "Continuous reference trajectory; no statistical independence assumption",
                    "Every retained B1/B2 residual and edge count is tested separately",
                ],
                "scope": "Audit of SHA256-verified retained force-provider reconstruction residuals and independently rebuilt analytic primitive constants. No QSD-law rate conclusion.",
            })
        for stream in reference["raw_archives"]:
            provenance.append(by_path[stream["archive"]])
    correction = {c["id"]: c for c in comparisons}
    presentation_rows = report["comparisons"] + [
        c for reference in report["references"] for c in reference["comparisons"]
    ]
    for comparison in presentation_rows:
        if comparison["id"] in correction:
            fixed = correction[comparison["id"]]
            comparison.update({
                k: fixed[k]
                for k in ["observed", "bound", "standard_error", "passed", "kind", "hypotheses"]
            })
            comparison["status"] = "not_rejected" if fixed["passed"] else "violated"
    report["summary"]["comparisons_failed"] = sum(
        c["passed"] is False for c in report["comparisons"]
    )
    # The source coverage ledger is preserved verbatim; no new quote credits.
    report["derived_applicability_correction"] = {
        "raw_dataset": str(args.dataset),
        "force_audit": str(args.output),
        "new_engine_steps": 0,
        "scope": "Reference comparisons corrected to continuous-trajectory provenance and individual exact checks; raw report remains immutable.",
    }
    result = {
        "dataset": str(args.dataset),
        "analysis_implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "new_engine_steps": 0,
        "source_provenance": [
            e for e in journal if e["tag"] in {"chapter04-source", "provenance"}
        ],
        "raw_provenance": provenance,
        "comparisons": comparisons,
        "recomputed_primitive_constants": primitive_rows,
        "summary": {
            "comparisons": len(comparisons),
            "individual_residuals_checked": sum(c["samples"] for c in comparisons),
            "failed": sum(not c["passed"] for c in comparisons),
        },
        "scope": "Finite native reference kernel identities and rigorous analytic primitive bounds; source coverage credits unchanged.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.presentation_output.parent.mkdir(parents=True, exist_ok=True)
    args.presentation_output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
