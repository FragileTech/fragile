"""Publish dimension-aware Chapter 6 bounds from separate retained/new datasets."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    directory = Path(sys.argv[1]).resolve()
    retained_path = directory / "retained-bounds.json"
    probe_path = directory / "native-probe-bounds.json"
    retained = json.loads(retained_path.read_text())
    probes = json.loads(probe_path.read_text())
    rows = retained["dimension_rows"]
    text = """# Chapter 6 dimension-aware bound tightening

The final position Gaussian is conditioned on the complete native preparation; its mean is unrestricted. Coordinate factorization removes the unused full-cube density factor. A closed-form layer-cake integral tightens the one-dimensional barrier further. Every swarm observable remains averaged over N.

For L=2 and s=0.02, a=2L/(sqrt(2pi)s), and the improved bound is d Lambda(a) min(1,a)^(d-1). These algebraic dimension rows use identical box/noise parameters; native observations are listed separately with their actual populations, horizons and seed counts.

| d | Archived full-box density bound | Coordinate-linear bound | Sharp layer-cake bound | Old/sharp |
|---:|---:|---:|---:|---:|
"""
    for row in rows:
        text += f"| {row['d']} | {row['old_bound']:.10g} | {row['linear_bound']:.10g} | {row['sharp_bound']:.10g} | {row['old_bound'] / row['sharp_bound']:.7g} |\n"
    text += """
The old values and their original audit remain immutable. For a<1, retaining the other-coordinate survival factors recovers the former small-box bound; dropping them alone could loosen it. No compact-support approximation or numerical quadrature is used. The analytic formulas are real inequalities evaluated in f64, with logarithmic bounds and positive underflow floors; this is not machine interval arithmetic.

| Data | N | d | Independent seeds per initial law | Terminal step | Observed alive barrier, left/right (unconditional) | Sharp uniform bound | Prepared-state mean bound, left/right | Cantelli allowance, left/right | Survivor counts, left/right |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    terminal_rows = []
    for name, report in [("Retained", retained), ("Fresh", probes)]:
        for case in report["cases"]:
            final = max(frame["step"] for frame in case["frames"])
            frames = [next(frame for frame in case["frames"] if frame["step"] == final and frame["cemetery_convention"] == "native_zero_alive" and frame["distribution"] == distribution) for distribution in [0, 1]]
            terminal_rows.append({"dataset": name, "case": case["case"], "N": case["N"], "d": case["d"], "frames": frames})
            text += f"| {name} | {case['N']} | {case['d']} | {frames[0]['requested_independent_seeds']} | {final} | {frames[0]['observed_unconditional_barrier']:.8g} / {frames[1]['observed_unconditional_barrier']:.8g} | {case['bounds']['sharp_bound']:.9g} | {frames[0]['prepared_state_mean_upper']:.8g} / {frames[1]['prepared_state_mean_upper']:.8g} | {frames[0]['independent_seed_Cantelli_allowance']:.7g} / {frames[1]['independent_seed_Cantelli_allowance']:.7g} | {frames[0]['survivors']} / {frames[1]['survivors']} |\n"
    text += """
Observed unconditional averages include absorbed zero trajectories and divide by all requested seeds. Survivor averages are recorded separately using each initial law and cemetery convention's own survivor count; they are unavailable at zero survivors. The N=2 hazard row is zero at its terminal horizon because all trajectories have absorbed; it is not a zero-error conditional law.

Each prepared-state bound uses analytic interior/tail Gaussian density envelopes and exact logarithmic barrier integrals. Its second-moment envelope gives a one-sided Cantelli allowance across independent trajectory seeds. A declared 1% family budget covers the comparisons; walkers and serial updates never increase the independent sample count. Fresh d=4/8 probes have 32 seeds per initial law and are kept distinct from the original 256-seed d=1/2 runs.

"""
    text += f"Retained-data checks: {retained['summary']['comparisons']} comparisons, {retained['summary']['comparisons_failed']} discrepancies, zero new simulation steps. Fresh-data checks: {probes['summary']['comparisons']} comparisons, {probes['summary']['comparisons_failed']} discrepancies. Passing these finite-sample checks does not supply a QSD eigenvalue, stationary reference or uniform mixing rate.\n"
    endpoint_path = directory / "exact-native-dimension-endpoints-final.json"
    endpoints = json.loads(endpoint_path.read_text())
    text += """
| Fresh N=16 probes | Own seeds per initial law | Whole empirical-law cost, initial → final | Whole-law final/initial | Uniform-alive cost, initial → final | Alive final/initial | Zero-event extinction upper (95%, per initial law) |
|---|---:|---:|---:|---:|---:|---:|
"""
    for case in endpoints["cases"]:
        first, last = case["endpoints"]
        seeds = case["independent_seeds_per_initial_law"][0]
        whole = "whole_swarm_exact_empirical_law_cost"
        alive = "uniform_alive_exact_empirical_law_cost"
        extinction_upper = -math.expm1(math.log(0.05) / seeds)
        text += (
            f"| d={case['d']}, 64 updates | {seeds} | "
            f"{first[whole]:.8g} → {last[whole]:.8g} | "
            f"{case['terminal_to_initial_ratios'][whole]:.8g} | "
            f"{first[alive]:.8g} → {last[alive]:.8g} | "
            f"{case['terminal_to_initial_ratios'][alive]:.8g} | "
            f"{extinction_upper:.8g} |\n"
        )
    text += "\nThese are exact minima for the finite empirical endpoint laws, with minimum-permutation whole-swarm state costs and each law's own survival denominator. They give observed decline with 32 independent seeds per law, not a certified QSD rate. All seeds survive through 64 updates; the zero-event upper bounds expose the limited precision for rare extinction.\n"
    text += "\nComplete formal proofs are in Chapter 6, labels `lem-convergence-dimension-log-barrier` and `lem-convergence-state-log-barrier`. The reports record each consumed manifest SHA256; the separate provenance records proof/API snapshots and the executed-binary identity.\n"
    text += "\nThe executed binary and its embedded source snapshot are bound in `execution-provenance.json`. Its linked analysis library predates presentation metadata additions; numerical formulas are unchanged. Final proof/API snapshots are retained separately, so they are not substituted for the recorded execution identity.\n"
    table = directory / "dimension-bounds-table.md"
    if table.exists():
        raise FileExistsError("Existing tightening results are immutable; select a new directory.")
    table.write_text(text)
    (directory / "terminal-comparisons.json").write_text(json.dumps(terminal_rows, indent=2))
    proof_path = Path("docs/source/2_fractal_gas/convergence_program/06_convergence.md")
    snapshots = {}
    for name, source in [
        ("proof-snapshot.md", proof_path),
        ("barrier-api-snapshot.rs", Path("crates/benchmarks/src/convergence_barrier_dimension.rs")),
        ("structural-tails-snapshot.rs", Path("crates/benchmarks/src/convergence_structural_tails.rs")),
    ]:
        target = directory / name
        target.write_bytes(source.read_bytes())
        snapshots[name] = {"source": str(source.resolve()), "path": str(target), "sha256": digest(target)}
    provenance = {"retained_report": {"path": str(retained_path), "sha256": digest(retained_path)}, "new_probe_report": {"path": str(probe_path), "sha256": digest(probe_path)}, "table": {"path": str(table), "sha256": digest(table)}, "helper": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__).resolve())}, "proof": {"path": str(proof_path.resolve()), "sha256": digest(proof_path)}, "original_data_mutations": 0, "dimension_rows": rows, "scope": "Dimension-explicit analytic bounds and independent native empirical comparisons; no unsupported uniform-QSD-rate claim."}
    provenance["source_snapshots"] = snapshots
    execution_path = directory / "execution-provenance.json"
    provenance["execution_identity"] = {
        "path": str(execution_path),
        "sha256": digest(execution_path),
        "details": json.loads(execution_path.read_text()),
    }
    provenance["exact_empirical_endpoints"] = {
        "path": str(endpoint_path),
        "sha256": digest(endpoint_path),
    }
    verification = directory / "native-probe-deep-verification.json"
    provenance["native_dataset_deep_verification"] = {
        "path": str(verification),
        "sha256": digest(verification),
        "details": json.loads(verification.read_text()),
    }
    (directory / "tightening-provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps({"table": str(table), "retained": retained["summary"], "fresh": probes["summary"]}))


if __name__ == "__main__":
    main()
