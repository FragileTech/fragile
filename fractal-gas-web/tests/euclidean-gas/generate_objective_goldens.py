"""Generate objective reference values for the Rust benchmark port.

Standard-library bridge over the native Optimization Lab library (COCO 2.8.2 +
classic benchmarks). Requires `make optimization-native`. Writes
algorithmic-gas/crates/benchmarks/tests/fixtures/objective-goldens.json and
copies the upstream COCO test cases next to it.
"""

import ctypes
import json
import math
from pathlib import Path
import random
import shutil


root = Path(__file__).resolve().parents[2]
fixtures = root.parent / "algorithmic-gas/crates/benchmarks/tests/fixtures"
lib = ctypes.CDLL(str(root / "build-optimization-native/optimization/libfg_optimization.so"))
lib.fgo_catalog.restype = ctypes.c_char_p
lib.fgo_error.restype = ctypes.c_char_p
lib.fgo_create.argtypes = [ctypes.c_char_p]
lib.fgo_create.restype = ctypes.c_uint32
lib.fgo_config.argtypes = [ctypes.c_uint32]
lib.fgo_config.restype = ctypes.c_char_p
lib.fgo_destroy.argtypes = [ctypes.c_uint32]
lib.fgo_sample64.argtypes = [
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
]


def sample(config, points):
    handle = lib.fgo_create(json.dumps(config).encode())
    if not handle:
        raise RuntimeError(f"{config}: {lib.fgo_error().decode()}")
    try:
        resolved = json.loads(lib.fgo_config(handle).decode())
        d = int(resolved["dimensions"])
        flat = (ctypes.c_double * (len(points) * d))(*[v for p in points for v in p])
        out = (ctypes.c_double * len(points))()
        if lib.fgo_sample64(handle, flat, len(points), out) < 0:
            raise RuntimeError(lib.fgo_error().decode())
        return resolved, [v if math.isfinite(v) else None for v in out]
    finally:
        lib.fgo_destroy(handle)


def points_for(rng, d, low, high, outside):
    span = high - low
    interior = [[low + span * rng.random() for _ in range(d)] for _ in range(2)]
    corner = [high if rng.random() < 0.5 else low for _ in range(d)]
    result = interior + [corner]
    if outside:
        result.append([1.5 * (high if rng.random() < 0.5 else low) for _ in range(d)])
    return result


rng = random.Random(20260918)
catalog = json.loads(lib.fgo_catalog().decode())
goldens = {"source": "fractal-gas-web native Optimization Lab (COCO 2.8.2)", "catalog": catalog}

bbob = []
for function in range(1, 25):
    # Upstream coco-fixtures.json already covers d=20/40 at instances 1, 2, 15.
    for d, instances in ((2, (1, 2, 7, 15, 1000)), (5, (1, 2, 7, 15, 1000)), (20, (7, 1000)), (40, (1000,))):
        for instance in instances:
            points = points_for(rng, d, -5.0, 5.0, outside=True)
            config = {"benchmark": f"bbob_{function}", "dimensions": d, "coco_instance": instance}
            resolved, values = sample(config, points)
            bbob.append({
                "function": function,
                "dimensions": d,
                "instance": instance,
                "minimum": resolved["reference_minimum"],
                "problem_id": resolved["coco_problem_id"],
                "points": points,
                "values": values,
            })
goldens["bbob"] = bbob

classics = []
cases = [
    ({"benchmark": "sphere"}, (2, 7)),
    ({"benchmark": "quadratic"}, (2, 7)),
    ({"benchmark": "quadratic", "alpha": 2.5}, (3,)),
    ({"benchmark": "mexican_hat"}, (1, 2, 6)),
    ({"benchmark": "mexican_hat", "lambda_h": 0.5, "vev": 300, "field_scale": 100, "tilt": 0.3}, (3,)),
    ({"benchmark": "rastrigin"}, (2, 7)),
    ({"benchmark": "eggholder"}, (2,)),
    ({"benchmark": "styblinski_tang"}, (2, 7)),
    ({"benchmark": "rosenbrock"}, (2, 7)),
    ({"benchmark": "easom"}, (2,)),
    ({"benchmark": "holder_table"}, (2,)),
    ({"benchmark": "lennard_jones", "n_atoms": 3}, (9,)),
    ({"benchmark": "lennard_jones"}, (30,)),
    ({"benchmark": "constant"}, (3,)),
    ({"benchmark": "stochastic_gaussian"}, (3,)),
    ({"benchmark": "gaussian_mixture"}, (2, 5)),
    ({"benchmark": "gaussian_mixture", "n_gaussians": 5, "benchmark_seed": 7}, (3,)),
]
bounds = {entry["id"]: entry["bounds"] for entry in catalog["benchmarks"]}
for base, dimensions in cases:
    for d in dimensions:
        low, high = bounds[base["benchmark"]]
        if base["benchmark"] == "easom":
            low, high = 1.0, 5.0  # The bump underflows elsewhere in the domain.
        points = points_for(rng, d, low, high, outside=False)
        resolved, values = sample({**base, "dimensions": d}, points)
        case = {"config": {**base, "dimensions": d}, "points": points, "values": values}
        for key in ("centers", "stds", "weights"):
            if base["benchmark"] == "gaussian_mixture":
                case[key] = resolved[key]
        classics.append(case)
goldens["classics"] = classics

fixtures.mkdir(parents=True, exist_ok=True)
(fixtures / "objective-goldens.json").write_text(json.dumps(goldens, separators=(",", ":")) + "\n")
shutil.copyfile(root / "tests/optimization/coco-fixtures.json", fixtures / "coco-fixtures.json")
print(f"{len(bbob)} BBOB cases, {len(classics)} classic cases -> {fixtures}")
