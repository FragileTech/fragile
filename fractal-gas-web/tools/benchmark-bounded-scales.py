"""Compare bounded adaptive scaling with preserved perturbation baselines."""

import argparse
import csv
import ctypes
import json
import statistics
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=5)
parser.add_argument("--budget", type=int, default=5000)
parser.add_argument("--min-scale", type=float, default=0.0001)
parser.add_argument("--max-scale", type=float, default=0.2)
args = parser.parse_args()
if args.seeds < 1 or args.budget < 32 or not 0 <= args.min_scale <= args.max_scale:
    parser.error("Require positive seeds/budget and ordered, nonnegative scales")
root = Path(__file__).resolve().parents[1]
library = ctypes.CDLL(str(root / "build-optimization-native/optimization/libfg_optimization.so"))
library.fgo_create.argtypes = [ctypes.c_char_p]
library.fgo_create.restype = ctypes.c_uint32
library.fgo_status.argtypes = [ctypes.c_uint32]
library.fgo_status.restype = ctypes.c_char_p
library.fgo_snapshot.argtypes = [ctypes.c_uint32]
library.fgo_snapshot.restype = ctypes.POINTER(ctypes.c_double)
library.fgo_step.argtypes = [ctypes.c_uint32]
library.fgo_step.restype = ctypes.c_int
library.fgo_destroy.argtypes = [ctypes.c_uint32]
library.fgo_error.restype = ctypes.c_char_p

variants = {
    "Gaussian": {"perturbation": "gaussian"},
    "Local covariance": {"perturbation": "local_covariance"},
    "Bounded adaptive": {"perturbation": "adaptive_fractal"},
    "Bounded + basin restarts": {"perturbation": "adaptive_fractal", "controller_enabled": True},
}
problems = ["quadratic", "bbob_10", "rastrigin", "bbob_5"]
rows = []
for problem in problems:
    for variant, fields in variants.items():
        for seed in range(args.seeds):
            config = dict(
                algorithm="wave", benchmark=problem, dimensions=2, walkers=8,
                max_walkers=64, perturbation_std=0.2, periodic=False,
                potential_force=False, gas_local_search=False,
                adaptive_min_scale=args.min_scale, adaptive_max_scale=args.max_scale,
                max_evaluations=args.budget, seed=seed, **fields,
            )
            start = time.perf_counter()
            handle = library.fgo_create(json.dumps(config).encode())
            if not handle:
                raise RuntimeError(library.fgo_error().decode())
            error = ""
            try:
                while True:
                    status = json.loads(library.fgo_status(handle))
                    exploration = status.get("exploration", {})
                    if exploration:
                        assert exploration["scale_min"] >= args.min_scale - 1e-12
                        assert exploration["scale_max"] <= args.max_scale + 1e-12
                    if status["finished"] or status["budget_exhausted"]:
                        break
                    if library.fgo_step(handle) < 0:
                        error = library.fgo_error().decode()
                        break
                frame = library.fgo_snapshot(handle)
                minimum = status["effective_settings"].get("reference_minimum", 0)
                rows.append(dict(
                    problem=problem, variant=variant, seed=seed, budget=args.budget,
                    evaluations=int(frame[5]), best=frame[9], regret=max(0, frame[9] - minimum),
                    seconds=time.perf_counter() - start,
                    min_scale=args.min_scale, max_scale=args.max_scale, error=error,
                ))
            finally:
                library.fgo_destroy(handle)
        print(problem, variant, flush=True)

output = root / "tests/optimization/reports"
output.mkdir(exist_ok=True)
with (output / "bounded-scales.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
lines = [
    "# Bounded adaptive scale comparison (fgopt-7)", "",
    f"Wave, two dimensions, {args.seeds} seeds, {args.budget} evaluations per run; "
    f"8 initial walkers, 64 maximum, nonperiodic bounds. Adaptive scale range "
    f"[{args.min_scale}, {args.max_scale}], Gaussian/local covariance standard deviation 0.2. "
    "Complete operations only; actual counts may be below the shared allowance. "
    "Runtime includes status extraction and controller metadata. This is a small experimental comparison, not a superiority claim.",
    "", "| Problem | Variant | Median regret | Mean evaluations | Mean seconds | Errors |",
    "|---|---|---:|---:|---:|---:|",
]
for problem in problems:
    for variant in variants:
        group = [r for r in rows if r["problem"] == problem and r["variant"] == variant]
        lines.append(
            f"| {problem} | {variant} | {statistics.median(r['regret'] for r in group):.6g} "
            f"| {statistics.mean(r['evaluations'] for r in group):.1f} "
            f"| {statistics.mean(r['seconds'] for r in group):.4f} "
            f"| {sum(bool(r['error']) for r in group)} |"
        )
lines += ["", f"Runs: {len(rows)}. Errors: {sum(bool(r['error']) for r in rows)}.", "",
          f"Reproduce: `python3 fractal-gas-web/tools/benchmark-bounded-scales.py --seeds {args.seeds} "
          f"--budget {args.budget} --min-scale {args.min_scale} --max-scale {args.max_scale}`.", "",
          "The earlier adaptive.md report describes the obsolete path-based scale rule. "
          "The bounded revision changes scalar learning and proposal normalization, so comparisons "
          "with that report measure the combined revision rather than isolating one coefficient."]
(output / "bounded-scales.md").write_text("\n".join(lines) + "\n")
