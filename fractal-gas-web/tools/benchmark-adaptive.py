"""Equal-budget, multi-seed exploratory comparison; no external Python dependencies."""
import argparse
import ctypes
import csv
import json
from pathlib import Path
import statistics
import time
import math

parser = argparse.ArgumentParser()
parser.add_argument("--budget", type=int, default=2000)
parser.add_argument("--seeds", type=int, default=3)
parser.add_argument("--report-only", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
output = root / "tests/optimization/reports"
output.mkdir(exist_ok=True)
lib = ctypes.CDLL(str(root / "build-optimization-native/optimization/libfg_optimization.so"))
for name in ["fgo_create", "fgo_status", "fgo_snapshot", "fgo_step", "fgo_destroy"]:
    getattr(lib, name).argtypes = [ctypes.c_char_p if name == "fgo_create" else ctypes.c_uint32]
lib.fgo_create.restype = ctypes.c_uint32
lib.fgo_status.restype = ctypes.c_char_p
lib.fgo_snapshot.restype = ctypes.POINTER(ctypes.c_double)
lib.fgo_error.restype = ctypes.c_char_p
variants = {
    "Gaussian": {"perturbation": "gaussian"},
    "Local covariance": {"perturbation": "local_covariance"},
    "Paths / fixed mixture": {"perturbation": "adaptive_fractal", "adaptive_active": False, "adaptive_difference": False, "adaptive_pairs": False, "adaptive_mixture": False},
    "Active / fixed mixture": {"perturbation": "adaptive_fractal", "adaptive_difference": False, "adaptive_pairs": False, "adaptive_mixture": False},
    "Adaptive mixture": {"perturbation": "adaptive_fractal"},
    "Restarts / no memory": {"perturbation": "adaptive_fractal", "controller_enabled": True, "basin_avoidance": False},
    "Restarts / basin memory": {"perturbation": "adaptive_fractal", "controller_enabled": True},
    "BIPOP-active CMA-ES": {"algorithm": "cmaes_bipop"},
}
problems = ["quadratic", "bbob_10", "rastrigin", "bbob_5", "stochastic_gaussian"]
if args.report_only:
    with (output / "adaptive.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    args.budget = int(rows[0]["budget"])
    args.seeds = len({int(row["seed"]) for row in rows})
else:
    rows = []
    builtin_minima = {"quadratic": 0.0, "rastrigin": 0.0}
    for problem in problems:
        for variant, fields in variants.items():
            for seed in range(args.seeds):
                cfg = dict(algorithm="wave", benchmark=problem, dimensions=2, walkers=8, max_walkers=64,
                           perturbation_std=0.2, periodic=False, potential_force=False, gas_local_search=False,
                           max_evaluations=args.budget, seed=seed, **{})
                cfg.update(fields)
                started = time.perf_counter()
                handle = lib.fgo_create(json.dumps(cfg).encode())
                if not handle:
                    raise RuntimeError(lib.fgo_error().decode())
                error = ""
                try:
                    while True:
                        status = json.loads(lib.fgo_status(handle))
                        if status["finished"] or status["budget_exhausted"]:
                            break
                        if lib.fgo_step(handle) < 0:
                            error = lib.fgo_error().decode()
                            break
                    elapsed = time.perf_counter() - started
                    snapshot = lib.fgo_snapshot(handle)
                    controller = status.get("controller", {})
                    minimum = status["effective_settings"].get(
                        "reference_minimum", builtin_minima.get(problem)
                    )
                    rows.append(dict(problem=problem, variant=variant, seed=seed, budget=args.budget,
                                     evaluations=int(snapshot[5]), best=snapshot[9], seconds=elapsed,
                                     reference_minimum=minimum if minimum is not None else "",
                                     regret=max(0.0, snapshot[9] - minimum) if minimum is not None else "",
                                     archived_basins=len(controller.get("basins", [])),
                                     repeated_basin_evaluations=controller.get("repeated_basin_evaluations", 0), error=error))
                finally:
                    lib.fgo_destroy(handle)
            print(problem, variant, flush=True)
    with (output / "adaptive.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

builtin_minima = {"quadratic": 0.0, "rastrigin": 0.0}


def regret(row):
    if row["regret"] != "":
        return float(row["regret"])
    minimum = builtin_minima.get(row["problem"])
    return None if minimum is None else max(0.0, float(row["best"]) - minimum)
lines = ["# Adaptive exploration: measured comparison", "", f"{args.seeds} seeds per case; {args.budget} evaluation allowance; two dimensions; initial swarm population 8 (CMA-ES uses its default population); maximum swarm population 64. CPU native build. Each method admits complete operations, so actual evaluation counts can fall below the common allowance.", "", "The Gaussian and Local covariance rows are the pre-existing perturbation baselines. All fractal variants use Wave with identical initial population, bounds, seed, and evaluation allowance. BIPOP-active CMA-ES is an external reference. Fixed-mixture rows are ablations of the new strategy. Basin matching is heuristic; archived-basin counts are not certified local minima. On the stochastic control the expectation is zero everywhere, so raw observed best measures noise extremes and is excluded from improvement claims.", "", "| Problem | Variant | Median regret | Median raw best | Mean seconds | Mean evaluations | Mean archived basins |", "|---|---|---:|---:|---:|---:|---:|"]
for problem in problems:
    for variant in variants:
        group = [row for row in rows if row["problem"] == problem and row["variant"] == variant]
        regrets = [value for row in group if (value := regret(row)) is not None]
        median_regret = f"{statistics.median(regrets):.6g}" if regrets else "n/a"
        lines.append(f"| {problem} | {variant} | {median_regret} | {statistics.median(float(r['best']) for r in group):.6g} | {statistics.mean(float(r['seconds']) for r in group):.4f} | {statistics.mean(float(r['evaluations']) for r in group):.1f} | {statistics.mean(float(r['archived_basins']) for r in group):.1f} |")

lines += ["", "## Improvement over previous perturbations", "", "Positive percentages and positive log10 values mean lower median regret. Paired wins compare the same seeds; ties use a relative tolerance of 1e-12.", "", "| Problem | New variant | Previous perturbation | Median-regret change | Orders of magnitude | Paired wins / ties / losses | Runtime ratio |", "|---|---|---|---:|---:|---:|---:|"]
for problem in problems:
    if problem == "stochastic_gaussian":
        continue
    for candidate in ["Adaptive mixture", "Restarts / basin memory"]:
        candidate_rows = {int(r["seed"]): r for r in rows if r["problem"] == problem and r["variant"] == candidate}
        for baseline in ["Gaussian", "Local covariance"]:
            baseline_rows = {int(r["seed"]): r for r in rows if r["problem"] == problem and r["variant"] == baseline}
            candidate_regret = statistics.median(regret(r) for r in candidate_rows.values())
            baseline_regret = statistics.median(regret(r) for r in baseline_rows.values())
            if baseline_regret == 0:
                change = 0.0 if candidate_regret == 0 else float("-inf")
            else:
                change = 100.0 * (baseline_regret - candidate_regret) / baseline_regret
            log_gain = math.log10(max(baseline_regret, 1e-300)) - math.log10(max(candidate_regret, 1e-300))
            wins = ties = losses = 0
            for seed in sorted(candidate_rows.keys() & baseline_rows.keys()):
                new = regret(candidate_rows[seed])
                old = regret(baseline_rows[seed])
                tolerance = 1e-12 * max(1.0, abs(new), abs(old))
                if new < old - tolerance:
                    wins += 1
                elif new > old + tolerance:
                    losses += 1
                else:
                    ties += 1
            runtime = statistics.mean(float(r["seconds"]) for r in candidate_rows.values()) / statistics.mean(float(r["seconds"]) for r in baseline_rows.values())
            change_text = f"{change:+.1f}%" if math.isfinite(change) else "worse (baseline zero)"
            lines.append(f"| {problem} | {candidate} | {baseline} | {change_text} | {log_gain:+.2f} | {wins} / {ties} / {losses} | {runtime:.2f}x |")

lines += ["", "## Aggregate deterministic comparison", "", "The regret factor is the geometric mean of the four per-problem median-regret ratios. Values above 1 mean worse objective quality. Runtime uses the corresponding geometric mean.", "", "| New variant | Previous perturbation | Regret factor | Runtime factor | Total paired wins / ties / losses |", "|---|---|---:|---:|---:|"]
deterministic = [problem for problem in problems if problem != "stochastic_gaussian"]
for candidate in ["Adaptive mixture", "Restarts / basin memory"]:
    for baseline in ["Gaussian", "Local covariance"]:
        regret_ratios = []
        runtime_ratios = []
        wins = ties = losses = 0
        for problem in deterministic:
            candidate_rows = {int(r["seed"]): r for r in rows if r["problem"] == problem and r["variant"] == candidate}
            baseline_rows = {int(r["seed"]): r for r in rows if r["problem"] == problem and r["variant"] == baseline}
            candidate_median = statistics.median(regret(r) for r in candidate_rows.values())
            baseline_median = statistics.median(regret(r) for r in baseline_rows.values())
            regret_ratios.append(candidate_median / max(baseline_median, 1e-300))
            runtime_ratios.append(
                statistics.mean(float(r["seconds"]) for r in candidate_rows.values())
                / statistics.mean(float(r["seconds"]) for r in baseline_rows.values())
            )
            for seed in sorted(candidate_rows.keys() & baseline_rows.keys()):
                new = regret(candidate_rows[seed])
                old = regret(baseline_rows[seed])
                tolerance = 1e-12 * max(1.0, abs(new), abs(old))
                if new < old - tolerance:
                    wins += 1
                elif new > old + tolerance:
                    losses += 1
                else:
                    ties += 1
        regret_factor = math.prod(regret_ratios) ** (1 / len(regret_ratios))
        runtime_factor = math.prod(runtime_ratios) ** (1 / len(runtime_ratios))
        lines.append(f"| {candidate} | {baseline} | {regret_factor:.2f}x | {runtime_factor:.2f}x | {wins} / {ties} / {losses} |")

memory_ratios = []
lines += ["", "## Basin-memory contribution", "", "This isolates the archive by comparing otherwise identical restart controllers.", "", "| Problem | Regret change from basin memory |", "|---|---:|"]
for problem in deterministic:
    without_memory = statistics.median(
        regret(r) for r in rows
        if r["problem"] == problem and r["variant"] == "Restarts / no memory"
    )
    with_memory = statistics.median(
        regret(r) for r in rows
        if r["problem"] == problem and r["variant"] == "Restarts / basin memory"
    )
    memory_ratios.append(with_memory / max(without_memory, 1e-300))
    lines.append(f"| {problem} | {100 * (without_memory - with_memory) / max(without_memory, 1e-300):+.1f}% |")
memory_factor = math.prod(memory_ratios) ** (1 / len(memory_ratios))
lines += ["", f"Across the four deterministic problems, basin memory reduced median regret by a geometric factor of {1 / memory_factor:.2f}x relative to the same restart controller without memory."]
errors = [row for row in rows if row["error"]]
lines += ["", f"Runs ending with a reported error: {len(errors)}."]
for row in errors:
    lines.append(f"- {row['problem']}, {row['variant']}, seed {row['seed']}: {row['error']}")
lines += ["", "Exact counts and timings are in adaptive.csv.", "", f"Reproduce with `python3 fractal-gas-web/tools/benchmark-adaptive.py --budget {args.budget} --seeds {args.seeds}` after building the native optimization engine, or regenerate the Markdown from the saved CSV with `--report-only`."]
(output / "adaptive.md").write_text("\n".join(lines) + "\n")
