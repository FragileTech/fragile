"""Compare ten exchanging Wave swarms against the native CMA-ES reference."""

import argparse
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time

HELPER = Path(__file__).with_name("benchmark-long-budget.py")
SPEC = importlib.util.spec_from_file_location("reference", HELPER)
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)
SCALES = [0.0001 * 2000 ** (i / 9) for i in range(10)]


def population(problem, seed, dimensions, budget):
    lib = ctypes.CDLL(str(REFERENCE.LIBRARY))
    lib.fgp_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.fgp_create.restype = ctypes.c_uint32
    lib.fgp_request.argtypes = [ctypes.c_uint32, ctypes.c_char_p]
    lib.fgp_request.restype = ctypes.c_char_p
    lib.fgp_destroy.argtypes = [ctypes.c_uint32]
    lib.fgo_error.restype = ctypes.c_char_p
    config = dict(seed=seed, concurrency=10, exchange_every=1, global_elites=20,
                  max_evaluations=budget,
                  defaults=dict(algorithm="wave", benchmark=problem, dimensions=dimensions,
                                walkers=32, max_walkers=32, elites=5, perturbation="gaussian",
                                periodic=False, controller_enabled=False, population_auto=False,
                                scale_auto=False, potential_force=False, gas_local_search=False),
                  members=[dict(id=f"swarm-{i}", exchange_count=5,
                                settings=dict(perturbation_std=scale))
                           for i, scale in enumerate(SCALES)])
    start = time.perf_counter()
    handle = lib.fgp_create(json.dumps(config).encode(), 0)
    if not handle:
        raise RuntimeError(lib.fgo_error().decode())

    def request(op, **kwargs):
        raw = lib.fgp_request(handle, json.dumps(dict(op=op, **kwargs)).encode())
        if not raw:
            raise RuntimeError(lib.fgo_error().decode())
        return json.loads(raw)

    try:
        resolved = request("config")
        minimum = resolved["members"][0]["settings"].get("reference_minimum", 0)
        status = request("status")
        initial = status
        assert status["evaluations"] == 320
        previous = 320
        imports = 0
        while not status["budget_exhausted"]:
            status = request("step")
            assert not status["failed"]
            assert previous < status["evaluations"] <= budget
            previous = status["evaluations"]
            assert len(status["members"]) == 10
            for member, scale in zip(status["members"], SCALES):
                assert member["settings"]["walkers"] == 32
                assert member["settings"]["perturbation_std"] == scale
                assert len(member["exports"]) == len(member["imports"]) == 5
                assert member["shortfall"] == 0
                assert all(item["source"] != member["id"] for item in member["imports"])
                assert len({item["destination"] for item in member["imports"]}) == 5
                assert not set(member["exports"]) & {item["destination"] for item in member["imports"]}
                imports += 5
        snapshots = [request("snapshot", member=i) for i in range(10)]
        assert all(int(frame[1]) == 32 for frame in snapshots)
        assert sum(int(frame[5]) for frame in snapshots) == status["evaluations"]
        best = min(frame[9] for frame in snapshots)
        assert best >= minimum - 1e-7
        return dict(problem=problem, dimensions=dimensions, variant="Fractal Populations",
                    seed=seed, budget=budget, evaluations=status["evaluations"], best=best,
                    reference_minimum=minimum, regret=max(0, best-minimum),
                    iterations=status["round"], imports_verified=imports,
                    seconds=time.perf_counter()-start, error="", config=config,
                    resolved_config=resolved, initial_status=initial, final_status=status)
    finally:
        lib.fgp_destroy(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--dimensions", type=int, default=20)
    parser.add_argument("--budget", type=int, default=100000)
    parser.add_argument("--output", type=Path, default=REFERENCE.ROOT / "tests/optimization/reports/fractal-populations-10x32-20d-100k")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(seeds=args.seeds, dimensions=args.dimensions, budget=args.budget,
                    scales=SCALES, problems=REFERENCE.PROBLEMS,
                    library_sha256=hashlib.sha256(REFERENCE.LIBRARY.read_bytes()).hexdigest(),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest())
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    rows = []
    with (args.output / "runs.jsonl").open("w") as stream:
        for problem in REFERENCE.PROBLEMS:
            for seed in range(args.seeds):
                for variant in ("Fractal Populations", "BIPOP-active CMA-ES"):
                    if variant == "Fractal Populations":
                        row = population(problem, seed, args.dimensions, args.budget)
                    else:
                        row = REFERENCE.run_case((problem, args.dimensions, variant, seed, args.budget))
                    rows.append(row)
                    stream.write(json.dumps(row)+"\n")
                    stream.flush()
                    print(f"{problem} seed={seed} {variant}: error={row['regret']:.6g}, evals={row['evaluations']}, seconds={row['seconds']:.2f}, failure={row['error']!r}", flush=True)
    lines = ["# Ten-swarm Fractal Populations versus CMA-ES", "",
             f"{args.dimensions} dimensions; {args.seeds} seeds (0–{args.seeds-1}); {args.budget:,} shared evaluations per run, including initialization. COCO instance 1.", "",
             "Fractal: ten concurrent Wave swarms, each with 32 walkers, five protected/exported elites and five foreign imports every step. Fixed Gaussian jump scales (absolute coordinate units): " + ", ".join(f"{s:.6g}" for s in SCALES) + ". Population and scale adaptation and basin restarts are disabled to preserve the requested swarm sizes/scales. Each exchange is checked for counts, foreign provenance, and distinct unprotected destinations.", "",
             "CMA: freshly executed BIPOP-active CMA-ES, native defaults (own population, 20%-of-domain initial sigma, bounded mapping). Fractal uses nonperiodic boundaries without repair, matching the earlier comparison. Wave uses float32 coordinates; CMA uses float64. Equal evaluation caps do not imply equal populations or identical boundary handling. Complete rounds/generations may leave unused budget. Timings include Python/JSON verification overhead and are not pure optimizer speed measurements.", "",
             "Error is best evaluated objective minus known optimum; lower is better. Success means error ≤ 1e-6. Five seeds are descriptive, not a significance test.", "",
             "| Problem | Method | Median error | Error min–max | Successes | Evaluations min–max | Median seconds |", "|---|---|---:|---|---:|---|---:|"]
    for problem, label in REFERENCE.PROBLEMS.items():
        for variant in ("Fractal Populations", "BIPOP-active CMA-ES"):
            group = [r for r in rows if r["problem"] == problem and r["variant"] == variant]
            errors = [r["regret"] for r in group]
            lines.append(f"| {label} | {variant} | {statistics.median(errors):.6g} | {min(errors):.6g}–{max(errors):.6g} | {sum(e<=1e-6 for e in errors)}/{len(errors)} | {min(r['evaluations'] for r in group)}–{max(r['evaluations'] for r in group)} | {statistics.median(r['seconds'] for r in group):.3f} |")
    lines += ["", f"Reported runtime errors: {sum(bool(r['error']) for r in rows)}. Total verified foreign imports: {sum(r.get('imports_verified', 0) for r in rows):,}.", "", "Full per-seed results and resolved configurations: runs.jsonl. Build/runner fingerprints: manifest.json.", "", "Reproduce: `uv run python fractal-gas-web/tools/benchmark-fractal-populations.py --output /tmp/fractal-populations-comparison`."]
    (args.output / "report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
