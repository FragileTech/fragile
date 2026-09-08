"""Reproducible native, fixed-evaluation-cap optimization reference comparison."""
import ctypes
import json
from pathlib import Path
import statistics
import time

root = Path(__file__).resolve().parents[1]
lib = ctypes.CDLL(str(root / "build-optimization-native/optimization/libfg_optimization.so"))
lib.fgo_create.argtypes = [ctypes.c_char_p]
lib.fgo_create.restype = ctypes.c_uint32
lib.fgo_snapshot.restype = ctypes.POINTER(ctypes.c_double)
for name in ["fgo_error", "fgo_config", "fgo_status"]:
    getattr(lib, name).restype = ctypes.c_char_p
rows = []
for benchmark in ["bbob_10", "rosenbrock", "rastrigin"]:
    for algorithm in ["cmaes_active", "cmaes_bipop", "wave", "gas"]:
        for seed in [7, 17, 27, 37, 47]:
            cfg = dict(algorithm=algorithm, benchmark=benchmark, dimensions=2,
                       walkers=64, seed=seed, max_evaluations=6400, periodic=False)
            start = time.perf_counter()
            handle = lib.fgo_create(json.dumps(cfg).encode())
            if not handle:
                raise RuntimeError(lib.fgo_error().decode())
            resolved = json.loads(lib.fgo_config(handle))
            while True:
                status = json.loads(lib.fgo_status(handle))
                if status["finished"] or status["budget_exhausted"]:
                    break
                if lib.fgo_step(handle) < 0:
                    raise RuntimeError(lib.fgo_error().decode())
            frame = list(lib.fgo_snapshot(handle)[:lib.fgo_snapshot_size(handle)])
            rows.append(dict(benchmark=benchmark, algorithm=algorithm, seed=seed,
                             precision=resolved.get("precision", "float32 coordinates"),
                             evaluations=int(frame[5]), best=frame[9],
                             seconds=time.perf_counter()-start, status=status,
                             config=resolved))
            lib.fgo_destroy(handle)
output = root.parent / "outputs/optimization/cma"
output.mkdir(parents=True, exist_ok=True)
(output / "comparison.json").write_text(json.dumps(rows, indent=2)+"\n")
lines = ["# CMA optimization reference comparison", "",
         "Native build; 2 dimensions; seeds 7, 17, 27, 37, 47; 6,400 total evaluation cap including initialization. "
         "Default algorithm parameters (Wave/GAS: 64 walkers). CMA may converge early; whole steps that cannot fit are not evaluated. "
         "Runtime includes initialization. CMA evaluates float64 coordinates; Wave/GAS use float32 coordinates. "
         "BBOB f10 is the rotated ellipsoid, instance 1; its raw objective includes an offset. "
         "These small optimization comparisons do not measure reward-density fidelity.", "",
         "| Benchmark | Algorithm | Median best objective | Median evaluations | Median seconds | Precision |",
         "|---|---|---:|---:|---:|---|"]
for benchmark in ["bbob_10", "rosenbrock", "rastrigin"]:
    for algorithm in ["cmaes_active", "cmaes_bipop", "wave", "gas"]:
        group = [r for r in rows if r["benchmark"] == benchmark and r["algorithm"] == algorithm]
        lines.append(f"| {benchmark} | {algorithm} | {statistics.median(r['best'] for r in group):.10g} | "
                     f"{statistics.median(r['evaluations'] for r in group):g} | "
                     f"{statistics.median(r['seconds'] for r in group):.5f} | {group[0]['precision']} |")
lines += ["", "Per-seed results, resolved configurations, and termination status are in comparison.json. "
          "Runtime is machine/build dependent; this is a smoke comparison, not a general ranking.", ""]
(output / "comparison.md").write_text("\n".join(lines))
print("\n".join(lines))
