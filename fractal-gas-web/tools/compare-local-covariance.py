"""Fixed-budget Wave comparison; run after building optimization-native.

Uses the native C API and standard library only. Rastrigin occupancy bins are
nearest integer lattice basins, a descriptive proxy rather than density error.
"""
import ctypes
import json
from pathlib import Path
import statistics
import time
from collections import Counter

root = Path(__file__).resolve().parents[1]
lib = ctypes.CDLL(str(root / "build-optimization-native/optimization/libfg_optimization.so"))
lib.fgo_create.argtypes = [ctypes.c_char_p]
lib.fgo_create.restype = ctypes.c_uint32
lib.fgo_snapshot.restype = ctypes.POINTER(ctypes.c_double)
lib.fgo_error.restype = ctypes.c_char_p
lib.fgo_config.restype = ctypes.c_char_p
rows = []
for benchmark in ["bbob_10", "rosenbrock", "rastrigin"]:
    for strategy in ["gaussian", "local_covariance"]:
        for seed in [7, 17, 27, 37, 47]:
            config = dict(algorithm="wave", benchmark=benchmark, dimensions=2,
                          walkers=64, seed=seed, perturbation=strategy,
                          perturbation_std=.15, max_evaluations=6400,
                          covariance_learning_rate=.1, periodic=False)
            handle = lib.fgo_create(json.dumps(config).encode())
            if not handle:
                raise RuntimeError(lib.fgo_error().decode())
            start = time.perf_counter()
            trace = []
            while True:
                size = lib.fgo_snapshot_size(handle)
                frame = list(lib.fgo_snapshot(handle)[:size])
                trace.append([int(frame[5]), frame[9]])
                if frame[5] >= 6400 or frame[6] == 0:
                    break
                if lib.fgo_step(handle) < 0:
                    raise RuntimeError(lib.fgo_error().decode())
                if lib.fgo_snapshot(handle)[5] == frame[5]:
                    break
            occupancy = Counter()
            for i in range(int(frame[1])):
                offset = 12 + i * 12
                if frame[offset + 6]:
                    # Rastrigin minima lie close to this integer lattice.
                    occupancy[str(tuple(round(v) for v in frame[offset:offset + 2]))] += 1
            rows.append(dict(benchmark=benchmark, strategy=strategy, seed=seed,
                             evaluations=int(frame[5]), best=frame[9],
                             seconds=time.perf_counter() - start,
                             occupancy=dict(occupancy) if benchmark == "rastrigin" else None,
                             convergence=trace))
            lib.fgo_destroy(handle)
output = root.parent / "outputs/optimization/local_covariance"
output.mkdir(parents=True, exist_ok=True)
(output / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
lines = ["# Local covariance comparison", "",
         "Wave, 2 dimensions, 64 walkers, 6,400 evaluations, seeds 7/17/27/37/47, "
         "standard deviation 0.15, learning rate 0.1, bounded domains. "
         "BBOB f10 uses the default instance. Values below are raw best objectives "
         "(BBOB includes its objective offset), with median across seeds.", "",
         "| Benchmark | Perturbation | Median best | Median seconds | Median occupied Rastrigin bins |",
         "|---|---|---:|---:|---:|"]
for benchmark in ["bbob_10", "rosenbrock", "rastrigin"]:
    for strategy in ["gaussian", "local_covariance"]:
        group = [r for r in rows if r["benchmark"] == benchmark and r["strategy"] == strategy]
        bins = statistics.median(len(r["occupancy"]) for r in group) if benchmark == "rastrigin" else "—"
        lines.append(f"| {benchmark} | {strategy} | {statistics.median(r['best'] for r in group):.8g} | "
                     f"{statistics.median(r['seconds'] for r in group):.4f} | {bins} |")
lines += ["", "Occupancy counts and relative mass (count / alive population), plus convergence "
          "traces, are available in comparison.json. Integer-lattice bins approximate "
          "Rastrigin basins; they do not measure fidelity to a specified reward density. "
          "This small comparison does not establish universal improvement.", ""]
(output / "comparison.md").write_text("\n".join(lines))
print("\n".join(lines))
