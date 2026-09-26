"""Population sweep with target stopping and verified numerical precision.

Each result is durable and resumable. CMA uses its own population schedule and is
run once per function/seed, rather than repeated for each fractal population.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import ctypes
import fcntl
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import random
import statistics


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "long_budget", Path(__file__).with_name("benchmark-long-budget.py")
)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)
METHODS = {
    "gaussian": ("Gaussian", {"perturbation": "gaussian"}),
    "local": ("Local covariance", {"perturbation": "local_covariance"}),
    "adaptive": ("Bounded adaptive", {"perturbation": "adaptive_fractal"}),
    "cloning": ("Clone-score covariance", {"perturbation": "cloning_guided", "boundary": "none"}),
    "cloning_cma": ("Clone-score covariance + boundary mapping", {"perturbation": "cloning_guided", "boundary": "cma"}),
    "cma": ("BIPOP-active CMA-ES", {"algorithm": "cmaes_bipop"}),
}
BENCH.VARIANTS.update(dict(METHODS.values()))
PROBLEMS = ["quadratic", "rastrigin", "rosenbrock"] + [f"bbob_{i}" for i in range(1, 25)]


def precision_info():
    lib = ctypes.CDLL(str(BENCH.LIBRARY))
    lib.fgo_precision.argtypes = []
    lib.fgo_precision.restype = ctypes.c_char_p
    return json.loads(lib.fgo_precision())


def run(case):
    problem, method, population, seed, budget, tolerance = case
    row = BENCH.run_case(
        (problem, 20, METHODS[method][0], seed, budget),
        {"walkers": population or 8, "max_walkers": population or 8,
         "controller_enabled": False},
        target_error=tolerance,
    )
    row["requested_population"] = population
    row["method"] = method
    row["precision"] = precision_info()
    row["boundary"] = row["initial_status"]["effective_settings"]["boundary"]
    return row


def row_key(row):
    return row["problem"], row["method"], row["requested_population"], row["seed"]


def report(output, rows, manifest):
    if not rows:
        return
    columns = [k for k in rows[0] if k not in
               ("config", "initial_status", "final_status", "precision")]
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=row_key))
    lines = [
        "# 20D population benchmark", "",
        f"{len(rows)}/{manifest['expected_runs']} runs completed. Up to "
        f"{manifest['budget']:,} evaluations each; absolute objective target "
        f"{manifest['tolerance']:g}. {manifest['seeds']} seeds per case.", "",
        "Fractal methods use Wave, five elites, nonperiodic bounds and no automatic "
        "restarts. Gaussian/local covariance scale is 0.2; bounded strategies use "
        "[0.0001, 0.2]. CMA uses its own population and restart schedule. "
        "BBOB instance is 1. No settings are tuned against these results.", "",
        "Clone-score covariance uses no boundary repair; Clone-score covariance + boundary mapping uses "
        "the same cloning-guided movement with libcmaes repair of outside proposals. "
        "Both use the current engine's post-movement elite restoration. "
        "These are fractal Wave variants, not runs of the CMA-ES optimizer.", "",
        f"Measured storage precision: `{json.dumps(manifest['precision'], sort_keys=True)}`. "
        "A double-precision snapshot does not imply double-precision search coordinates.", "",
        "Stopping is checked after initialization and complete steps. Unused budget "
        "is retained when another complete operation cannot fit. Failed runs retain "
        "their last best objective and count as failures. CPU timings were measured "
        "during concurrent execution, not isolated latency.", "",
        f"Actual evaluations: {sum(r['evaluations'] for r in rows):,}. "
        f"Execution errors: {sum(bool(r['error']) for r in rows)}.", "",
        "| Function | Method | Walkers | Runs | Target reached | Median error | Median evaluations | Median CPU seconds | Errors |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups = {}
    for row in rows:
        groups.setdefault(row_key(row)[:3], []).append(row)
    for (problem, method, population), group in sorted(groups.items()):
        lines.append(
            f"| {problem} | {METHODS[method][0]} | {population or 'automatic'} | "
            f"{len(group)} | {sum(r['target_reached'] for r in group)}/{len(group)} | "
            f"{statistics.median(r['regret'] for r in group):.6g} | "
            f"{statistics.median(r['evaluations'] for r in group):g} | "
            f"{statistics.median(r['cpu_seconds'] for r in group):.3f} | "
            f"{sum(bool(r['error']) for r in group)} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budget", type=int, default=1_000_000)
    parser.add_argument("--tolerance", type=float, default=10e-6)
    parser.add_argument("--populations", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512, 1000])
    parser.add_argument("--methods", choices=list(METHODS), nargs="+", default=list(METHODS))
    parser.add_argument("--problems", choices=PROBLEMS, nargs="+", default=PROBLEMS)
    parser.add_argument("--precision", choices=["fp64", "mixed"], default="fp64")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.seeds, args.budget) < 1 or min(args.populations) < 8:
        parser.error("Workers, seeds and budget must be positive; populations must be at least eight")
    if not BENCH.math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("Tolerance must be finite and nonnegative")
    for values in (args.methods, args.problems, args.populations):
        if len(values) != len(set(values)):
            parser.error("Duplicate matrix entries are not allowed")
    precision = precision_info()
    fractal = any(method != "cma" for method in args.methods)
    required = ["objective_bits", "cma_coordinates_bits"]
    if fractal:
        required += ["swarm_coordinates_bits", "swarm_fitness_bits"]
    if args.precision == "fp64" and any(precision[key] != 64 for key in required):
        parser.error(f"Requested FP64 is unsupported by this engine: {precision}")
    cases = [(problem, method, population, seed, args.budget, args.tolerance)
             for problem in args.problems for method in args.methods
             for population in ([0] if method == "cma" else args.populations)
             for seed in range(args.seeds)]
    files = [BENCH.LIBRARY, Path(__file__), Path(BENCH.__file__),
             *sorted((ROOT / "src").rglob("*.cpp")), *sorted((ROOT / "src").rglob("*.hpp")),
             *sorted((ROOT / "src").rglob("*.h"))]
    manifest = {
        "dimensions": 20, "elites": 5, "budget": args.budget, "tolerance": args.tolerance,
        "seeds": args.seeds, "populations": args.populations, "methods": args.methods,
        "problems": args.problems, "precision": precision,
        "requested_precision": args.precision, "expected_runs": len(cases),
        "fingerprints": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in files},
    }
    if args.dry_run:
        print(json.dumps({k: v for k, v in manifest.items() if k != "fingerprints"}, indent=2))
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = []
    if args.resume:
        if json.loads((output / "manifest.json").read_text()) != manifest:
            raise RuntimeError("Cannot resume: configuration, precision or implementation changed")
        rows = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
    elif any((output / name).exists() for name in ("runs.jsonl", "manifest.json")):
        raise RuntimeError("Existing results are not overwritten; choose a new output or --resume")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    done = {row_key(row) for row in rows}
    expected = {case[:4] for case in cases}
    if len(done) != len(rows) or not done <= expected:
        raise RuntimeError("Duplicate or unexpected saved results")
    cases = [case for case in cases if case[:4] not in done]
    random.Random(20260926).shuffle(cases)
    with (output / "runs.jsonl").open("a") as stream, ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for future in as_completed([pool.submit(run, case) for case in cases]):
            row = future.result()
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(f"{len(rows)}/{manifest['expected_runs']} {row['problem']} {row['method']} "
                  f"N={row['requested_population']} seed={row['seed']} "
                  f"error={row['regret']:.6g} evals={row['evaluations']} "
                  f"target={row['target_reached']} {row['error']}", flush=True)
            if len(rows) % 10 == 0:
                report(output, rows, manifest)
    for name, digest in manifest["fingerprints"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Implementation changed during benchmark: {name}")
    assert len(rows) == manifest["expected_runs"]
    report(output, rows, manifest)


if __name__ == "__main__":
    main()
