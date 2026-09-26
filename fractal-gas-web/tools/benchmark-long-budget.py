"""Reproducible long-budget comparison of preserved and adaptive perturbations.

Uses separate processes because the native C API is serialized, not thread-safe.
Writes each completed run immediately; never overwrites the earlier benchmark reports.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import ctypes
import fcntl
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "build-optimization-native/optimization/libfg_optimization.so"
PROBLEMS = {
    "quadratic": "Quadratic bowl",
    "bbob_10": "Rotated ill-conditioned ellipsoid",
    "rastrigin": "Rastrigin",
    "bbob_15": "Rotated Rastrigin",
    "rosenbrock": "Rosenbrock",
    "bbob_5": "Boundary optimum (linear slope)",
}
VARIANTS = {
    "Gaussian": {"perturbation": "gaussian"},
    "Local covariance": {"perturbation": "local_covariance"},
    "Bounded adaptive": {"perturbation": "adaptive_fractal"},
    "Bounded + basin restarts": {
        "perturbation": "adaptive_fractal", "controller_enabled": True,
    },
    "BIPOP-active CMA-ES": {"algorithm": "cmaes_bipop"},
}


def run_case(case, overrides=None, target_error=None):
    if target_error is not None and (not math.isfinite(target_error) or target_error < 0):
        raise ValueError("Target error must be finite and nonnegative")
    problem, dimensions, variant, seed, budget = case
    lib = ctypes.CDLL(str(LIBRARY))
    for name in ("fgo_create", "fgo_status", "fgo_snapshot", "fgo_step", "fgo_destroy"):
        getattr(lib, name).argtypes = [
            ctypes.c_char_p if name == "fgo_create" else ctypes.c_uint32
        ]
    lib.fgo_create.restype = ctypes.c_uint32
    lib.fgo_status.restype = ctypes.c_char_p
    lib.fgo_snapshot.restype = ctypes.POINTER(ctypes.c_double)
    lib.fgo_step.restype = ctypes.c_int
    lib.fgo_error.restype = ctypes.c_char_p
    config = dict(
        algorithm="wave", benchmark=problem, dimensions=dimensions,
        walkers=8, max_walkers=64, perturbation_std=0.2, periodic=False,
        potential_force=False, gas_local_search=False,
        adaptive_min_scale=0.0001, adaptive_max_scale=0.2,
        max_evaluations=budget, seed=seed,
    )
    config.update(VARIANTS[variant])
    if overrides:
        config.update(overrides)
    if config["algorithm"] == "wave":
        config["elites"] = 5
    started, cpu_started = time.perf_counter(), time.process_time()
    handle = lib.fgo_create(json.dumps(config).encode())
    if not handle:
        raise RuntimeError(f"{case}: {lib.fgo_error().decode()}")
    try:
        initial = json.loads(lib.fgo_status(handle))
        error = ""
        target_reached = False
        minimum = initial["effective_settings"].get("reference_minimum", 0.0)
        previous_evaluations = -1
        previous_best = float("inf")
        minimum_population = maximum_population = int(lib.fgo_snapshot(handle)[1])
        while True:
            frame = lib.fgo_snapshot(handle)
            evaluations, best = int(frame[5]), float(frame[9])
            minimum_population = min(minimum_population, int(frame[1]))
            maximum_population = max(maximum_population, int(frame[1]))
            if evaluations < previous_evaluations or evaluations > budget:
                raise AssertionError(f"Nonmonotonic or over-budget accounting: {case}")
            if best > previous_best:
                raise AssertionError(f"Best-so-far regressed: {case}")
            previous_evaluations, previous_best = evaluations, best
            # Stop between complete operations, including immediately after
            # initialization. Never truncate a generation or pad a budget.
            if target_error is not None and abs(best - minimum) <= target_error:
                target_reached = True
                status = json.loads(lib.fgo_status(handle))
                break
            # Session::step performs complete-operation admission internally. Query
            # full status only at termination, avoiding archive JSON on every step.
            if lib.fgo_step(handle) < 0:
                message = lib.fgo_error().decode()
                status = json.loads(lib.fgo_status(handle))
                if not status["finished"] and not status["budget_exhausted"]:
                    error = message
                break
        seconds, cpu_seconds = time.perf_counter() - started, time.process_time() - cpu_started
        frame = lib.fgo_snapshot(handle)
        if config["algorithm"] == "wave" and status["effective_settings"]["elites"] != 5:
            raise AssertionError("Fractal benchmark must retain five elites")
        controller = status.get("controller", {})
        minimum = initial["effective_settings"].get("reference_minimum", 0.0)
        regret = float(frame[9]) - minimum
        if regret < -1e-7:
            raise AssertionError(f"Result below reference optimum: {case}: {regret}")
        return dict(
            problem=problem, dimensions=dimensions, variant=variant, seed=seed,
            budget=budget, evaluations=int(frame[5]), best=float(frame[9]),
            iterations=int(frame[4]), minimum_population=minimum_population,
            maximum_population=maximum_population,
            reference_minimum=minimum, regret=max(0.0, regret),
            seconds=seconds, cpu_seconds=cpu_seconds,
            finished=status["finished"], budget_exhausted=status["budget_exhausted"],
            stop_reason="target_reached" if target_reached else status.get("stop_reason", ""),
            target_error=target_error, target_reached=target_reached,
            archived_basins=len(controller.get("basins", [])),
            error=error, config=config, initial_status=initial, final_status=status,
        )
    finally:
        lib.fgo_destroy(handle)


def write_report(output, rows, manifest):
    columns = [key for key in rows[0] if key not in ("config", "initial_status", "final_status")]
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Long-budget optimization comparison", "",
        f"{manifest['budget']:,} evaluations allowed per run; {manifest['seeds']} seeds "
        f"(0–{manifest['seeds'] - 1}); dimensions {manifest['dimensions']}. "
        f"{len(rows)} completed runs, {sum(bool(r['error']) for r in rows)} reported errors.", "",
        "All fractal variants use Wave, 8 initial walkers, maximum population 64, five protected elites, "
        "nonperiodic bounds, and unchanged default selection settings. Gaussian and "
        "local covariance are the preserved pre-existing perturbations in the current "
        "engine, not separately rebuilt historical revisions. Their standard deviation "
        "is 0.2; bounded adaptive uses [0.0001, 0.2]. The restart variant enables the "
        "existing controller and basin archive with defaults. No tuning was done for "
        "these problems. CMA uses its own selection rule, default initial population, initial sigma of "
        "20% of domain width, and 9 large-population runs. Equal evaluation allowance "
        "does not imply equal movement scales or populations. COCO instance is 1; "
        "seeds vary optimizer randomness, not the problem instance.", "",
        "Every actual initialization, trial, paired alternative, and restart evaluation "
        "is charged by the engine. Only complete operations are admitted. Runs may "
        "stop below the allowance if the next operation does not fit or the optimizer "
        "terminates. No artificial evaluations are added. Monotonic best-so-far and "
        "evaluation counts, and the hard evaluation ceiling, are checked after every step.", "",
        "Regret means best objective minus the known optimum; lower is better. "
        "BBOB reference minima come from the realized COCO problem. Success means "
        "regret ≤ 1e-6, a common reporting threshold, not an engine stopping rule. "
        "IQR is the interpolated 25th–75th percentile range. These seeds are descriptive "
        "evidence, not a universal performance claim. CMA uses double-coordinate "
        "evaluation; fractal walkers use float coordinates, limiting comparisons "
        "near numerical precision.", "",
        "Failed runs remain in every quality summary using their last best-so-far; "
        "they are not silently removed or restarted. A dagger (†) marks any group "
        "containing a reported error. Inspect evaluation use and the termination "
        "section before interpreting those groups as a full-budget comparison.", "",
        "Process CPU time includes the native engine, snapshot checks, and initial/final "
        "status serialization. Wall times also include contention between benchmark "
        "workers; they are not isolated latency measurements. Full configurations, "
        "effective settings, final diagnostics, best coordinates, and archive state "
        "are retained in runs.jsonl. Build and source fingerprints are in manifest.json.", "",
        "## Median final regret", "",
        "| Problem | Dimensions | Gaussian | Local covariance | Bounded adaptive | Bounded + basins | BIPOP-active CMA-ES |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    groups = {}
    for problem in PROBLEMS:
        for dimensions in manifest["dimensions"]:
            values = []
            for variant in VARIANTS:
                group = [r for r in rows if (r["problem"], r["dimensions"], r["variant"]) == (problem, dimensions, variant)]
                groups[problem, dimensions, variant] = group
                suffix = "†" if any(r["error"] for r in group) else ""
                values.append(f"{statistics.median(r['regret'] for r in group):.6g}{suffix}")
            lines.append(f"| {PROBLEMS[problem]} | {dimensions} | " + " | ".join(values) + " |")
    lines += ["", "## Variability, success, evaluation use, and time", "",
              "| Problem | d | Variant | Median | IQR | Best–worst | Successes | Errors | Evaluations min–max | Median CPU seconds |",
              "|---|---:|---|---:|---|---|---:|---:|---|---:|"]
    for (problem, dimensions, variant), group in groups.items():
        regrets = [r["regret"] for r in group]
        q1, _, q3 = statistics.quantiles(regrets, n=4, method="inclusive") if len(regrets) > 1 else [regrets[0]] * 3
        counts = [r["evaluations"] for r in group]
        lines.append(
            f"| {PROBLEMS[problem]} | {dimensions} | {variant} | {statistics.median(regrets):.6g} "
            f"| {q1:.4g}–{q3:.4g} | {min(regrets):.4g}–{max(regrets):.4g} "
            f"| {sum(value <= 1e-6 for value in regrets)}/{len(group)} "
            f"| {sum(bool(r['error']) for r in group)} "
            f"| {min(counts)}–{max(counts)} | {statistics.median(r['cpu_seconds'] for r in group):.3f} |"
        )
    lines += ["", "## Matched-seed comparisons against previous local covariance", "",
              "Wins/ties/losses compare final regrets with tolerance 1e-8 × max(1, |a|, |b|). "
              "Ratios compare medians; below 1 is better. These are descriptive comparisons without significance tests.", "",
              "| Problem | d | Candidate | Median regret ratio | Wins / ties / losses |",
              "|---|---:|---|---:|---|"]
    for (problem, dimensions, variant), group in groups.items():
        if variant not in ("Bounded adaptive", "Bounded + basin restarts", "BIPOP-active CMA-ES"):
            continue
        baseline = groups[problem, dimensions, "Local covariance"]
        base_by_seed = {r["seed"]: r["regret"] for r in baseline}
        wins = ties = losses = 0
        for row in group:
            a, b = row["regret"], base_by_seed[row["seed"]]
            tolerance = 1e-8 * max(1, abs(a), abs(b))
            if a < b - tolerance:
                wins += 1
            elif a > b + tolerance:
                losses += 1
            else:
                ties += 1
        base_median = statistics.median(base_by_seed.values())
        ratio = f"{statistics.median(r['regret'] for r in group) / base_median:.4g}" if base_median else "n/a (baseline zero)"
        lines.append(f"| {PROBLEMS[problem]} | {dimensions} | {variant} | {ratio} | {wins} / {ties} / {losses} |")
    previous_file = ROOT / "tests/optimization/reports/bounded-scales.csv"
    if previous_file.exists() and manifest.get("fractal_elites", 0) == 0:
        with previous_file.open() as stream:
            previous_rows = list(csv.DictReader(stream))
        lines += ["", "## Larger-budget effect on the original five seeds", "",
                  "This section uses only matching seeds 0–4, in two dimensions, "
                  "from bounded-scales.csv and this run. These medians can differ "
                  "from the ten-seed summaries above. Both budgets use the same "
                  "fractal configuration; failed runs retain their last best-so-far.", "",
                  "| Problem | Variant | 5,000-evaluation median | 100,000-evaluation median |",
                  "|---|---|---:|---:|"]
        for problem in ("quadratic", "bbob_10", "rastrigin", "bbob_5"):
            for variant in VARIANTS:
                earlier = [r for r in previous_rows if r["problem"] == problem and r["variant"] == variant]
                later = [r for r in rows if r["problem"] == problem and r["variant"] == variant
                         and r["dimensions"] == 2 and r["seed"] < 5]
                if len(earlier) == len(later) == 5:
                    lines.append(f"| {PROBLEMS[problem]} | {variant} "
                                 f"| {statistics.median(float(r['regret']) for r in earlier):.6g} "
                                 f"| {statistics.median(r['regret'] for r in later):.6g} |")
    lines += ["", "## Termination", ""]
    for row in rows:
        if row["error"] or row["finished"]:
            lines.append(f"- {row['problem']}, d={row['dimensions']}, {row['variant']}, seed {row['seed']}: "
                         f"{row['evaluations']} evaluations; {row['error'] or row['stop_reason'] or 'optimizer finished'}.")
    lines += ["", "Reproduce after building the native engine:", "", "```sh",
              "python3 fractal-gas-web/tools/benchmark-long-budget.py "
              f"--budget {manifest['budget']} --seeds {manifest['seeds']} "
              f"--dimensions {' '.join(map(str, manifest['dimensions']))} --workers {manifest['workers']} "
              f"--output {output.relative_to(ROOT)}", "```", "",
              "Use `--report-only` with the same output directory to regenerate this report without running optimization."]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=100000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[20])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default="tests/optimization/reports/long-budget-100k-20d-5elites")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.budget < 32 or args.seeds < 1 or args.workers < 1:
        parser.error("Budget must be at least 32; seeds and workers must be positive")
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    run_lock = (output / ".run.lock").open("a")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("This output directory already has an active benchmark process")
    if args.report_only:
        rows = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
        manifest = json.loads((output / "manifest.json").read_text())
    else:
        if not args.resume and (output / "runs.jsonl").exists() and (output / "runs.jsonl").stat().st_size:
            parser.error("Output already contains results; choose a new directory or --report-only")
        sources = sorted((ROOT / "src/optimization").glob("*")) + sorted((ROOT / "src/fractal").glob("*.hpp"))
        sources += [ROOT / "src/fractal_gas.hpp", ROOT / "src/fractal_tree.hpp", Path(__file__)]
        manifest = dict(
            budget=args.budget, seeds=args.seeds, dimensions=args.dimensions, fractal_elites=5,
            workers=args.workers, platform=platform.platform(), python=platform.python_version(),
            git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            library_sha256=hashlib.sha256(LIBRARY.read_bytes()).hexdigest(),
            sources={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sources if path.is_file()},
        )
        rows = []
        if args.resume:
            previous = json.loads((output / "manifest.json").read_text())
            for key in ("budget", "seeds", "dimensions", "fractal_elites", "library_sha256"):
                if previous[key] != manifest[key]:
                    parser.error(f"Resume mismatch: {key}")
            for path, fingerprint in previous["sources"].items():
                if path.startswith("src/") and manifest["sources"].get(path) != fingerprint:
                    parser.error(f"Engine source changed: {path}")
            manifest["previous_executions"] = previous.get("previous_executions", []) + [
                {"workers": previous["workers"], "runner_sha256": previous["sources"][str(Path(__file__).relative_to(ROOT))]}
            ]
            rows = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        cases = [(problem, dimension, variant, seed, args.budget)
                 for problem in PROBLEMS for dimension in args.dimensions
                 for variant in VARIANTS for seed in range(args.seeds)]
        random.Random(20260925).shuffle(cases)
        total = len(cases)
        completed = {(r["problem"], r["dimensions"], r["variant"], r["seed"], r["budget"]) for r in rows}
        cases = [case for case in cases if case not in completed]
        with (output / "runs.jsonl").open("a") as stream, ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [pool.submit(run_case, case) for case in cases]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(f"{len(rows)}/{total} {row['problem']} d={row['dimensions']} "
                      f"{row['variant']} seed={row['seed']} regret={row['regret']:.5g} "
                      f"evals={row['evaluations']} cpu={row['cpu_seconds']:.2f}s {row['error']}", flush=True)
    rows.sort(key=lambda row: (row["problem"], row["dimensions"], row["variant"], row["seed"]))
    expected = len(PROBLEMS) * len(manifest["dimensions"]) * len(VARIANTS) * manifest["seeds"]
    keys = {(r["problem"], r["dimensions"], r["variant"], r["seed"]) for r in rows}
    if len(rows) != expected or len(keys) != expected:
        raise RuntimeError(f"Incomplete matrix: {len(rows)} / {expected}")
    write_report(output, rows, manifest)


if __name__ == "__main__":
    main()
