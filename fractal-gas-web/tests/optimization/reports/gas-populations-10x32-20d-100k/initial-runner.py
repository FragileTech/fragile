"""Compare exchanging GAS swarms with single GAS and CMA-ES at equal budgets."""

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
GAS_SCALES = [0.1 * 100 ** (i / 9) for i in range(10)]
SCALES = [0.0001 * 2000 ** (i / 9) for i in range(10)]


def population(problem, seed, dimensions, budget, movement):
    lib = ctypes.CDLL(str(REFERENCE.LIBRARY))
    lib.fgp_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.fgp_create.restype = ctypes.c_uint32
    lib.fgp_request.argtypes = [ctypes.c_uint32, ctypes.c_char_p]
    lib.fgp_request.restype = ctypes.c_char_p
    lib.fgp_destroy.argtypes = [ctypes.c_uint32]
    lib.fgo_error.restype = ctypes.c_char_p
    config = dict(seed=seed, concurrency=10, exchange_every=1, global_elites=20,
                  max_evaluations=budget,
                  defaults=dict(algorithm="gas", benchmark=problem, dimensions=dimensions,
                                walkers=32, max_walkers=32, elites=5, perturbation=movement,
                                periodic=False, controller_enabled=False, population_auto=False,
                                scale_auto=False, potential_force=False, gas_local_search=False),
                  members=[dict(id=f"swarm-{i}", exchange_count=5,
                                settings=(dict(gas_scale_multiplier=scale) if movement == "gas_adaptive" else dict(perturbation_std=scale)))
                           for i, scale in enumerate(GAS_SCALES if movement == "gas_adaptive" else SCALES)])
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
            for member, scale in zip(status["members"], GAS_SCALES if movement == "gas_adaptive" else SCALES):
                assert member["settings"]["walkers"] == 32
                assert member["settings"]["gas_scale_multiplier" if movement == "gas_adaptive" else "perturbation_std"] == scale
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
        return dict(problem=problem, dimensions=dimensions, variant="Population GAS " + movement,
                    seed=seed, budget=budget, evaluations=status["evaluations"], best=best,
                    reference_minimum=minimum, regret=max(0, best-minimum),
                    iterations=status["round"], imports_verified=imports,
                    seconds=time.perf_counter()-start, error="", config=config,
                    resolved_config=resolved, initial_status=initial, final_status=status)
    finally:
        lib.fgp_destroy(handle)


def run_case(case):
    problem, seed, dimensions, budget, variant = case
    if variant.startswith("Population"):
        return population(problem, seed, dimensions, budget, variant.split()[-1])
    if variant == "BIPOP-active CMA-ES":
        return REFERENCE.run_case((problem, dimensions, variant, seed, budget))
    movement = variant.split()[-1]
    REFERENCE.VARIANTS[variant] = dict(algorithm="gas", perturbation=movement)
    row = REFERENCE.run_case((problem, dimensions, variant, seed, budget),
                             dict(walkers=320, max_walkers=320, elites=5,
                                  controller_enabled=False, population_auto=False, scale_auto=False,
                                  gas_scale_multiplier=1, gas_local_search=False))
    return row


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budget", type=int, default=100000)
    parser.add_argument("--dimensions", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    variants = ["Population GAS gas_adaptive", "Single GAS gas_adaptive",
                "Population GAS local_covariance", "Single GAS local_covariance", "BIPOP-active CMA-ES"]
    manifest = dict(dimensions=args.dimensions, budget=args.budget, seeds=args.seeds,
                    workers=args.workers, variants=variants, covariance_scales=SCALES,
                    gas_multipliers=GAS_SCALES, problems=REFERENCE.PROBLEMS,
                    library_sha256=hashlib.sha256(REFERENCE.LIBRARY.read_bytes()).hexdigest(),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest())
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    rows = []
    cases = [(problem, seed, args.dimensions, args.budget, variant)
             for problem in REFERENCE.PROBLEMS for seed in range(args.seeds) for variant in variants]
    with (args.output / "runs.jsonl").open("w") as stream, ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(run_case, case) for case in cases]):
            row = future.result()
            rows.append(row)
            stream.write(json.dumps(row)+"\n"); stream.flush()
            print(f"{len(rows)}/{len(cases)} {row['problem']} seed={row['seed']} {row['variant']}: error={row['regret']:.6g}, evals={row['evaluations']}, seconds={row['seconds']:.2f}, failure={row['error']!r}", flush=True)
    lines = ["# GAS populations versus a single GAS and CMA-ES", "",
             f"{args.dimensions} dimensions; {args.seeds} seeds (0–{args.seeds-1}); {args.budget:,} total evaluation cap per run. COCO instance 1.", "",
             "Populations: ten concurrent 32-walker GAS instances, exporting five current best walkers and importing five foreign walkers every round. GAS does not protect historical elite slots; its algorithm is unchanged between exchanges. Replacement uses the same retention criterion as shrinking. All exchange counts, foreign provenance, disjoint import/export indices and fixed swarm sizes are verified.", "",
             "Single GAS: 320 walkers, equal total population and budget. GAS-adaptive uses multiplier 1; local covariance uses standard deviation 0.2. Population GAS-adaptive multipliers: " + ", ".join(f"{s:.6g}" for s in GAS_SCALES) + ". Population local covariance standard deviations (absolute units): " + ", ".join(f"{s:.6g}" for s in SCALES) + ". No post-result tuning. Comparing multi-scale populations with these single-scale baselines does not isolate the effect of migration.", "",
             "GAS-adaptive noise scales each coordinate by domain width × multiplier × 10^(-5 + 4φ), where φ is normalized objective. Local covariance learns covariance shape from accepted movement evidence, with fixed standard deviation per swarm. GAS tabu memory remains enabled and private to each swarm. Local search, automatic restarts and population/scale scheduling are disabled for both single and population GAS to isolate movement and exchange.", "",
             "CMA-ES is freshly rerun BIPOP-active with its own default population and restart schedule. Nonperiodic GAS uses no boundary repair; CMA uses its bounded mapping. GAS coordinates are float32, CMA float64. Whole-operation budget admission may leave evaluations unused. Timings include benchmark overhead and concurrent contention; do not interpret as isolated speed rankings.", "",
             "Median error from the known optimum; lower is better. † marks any runtime failure in a group.", "",
             "| Problem | " + " | ".join(variants) + " |", "|---|" + "---:|" * len(variants)]
    for problem, label in REFERENCE.PROBLEMS.items():
        cells = []
        for variant in variants:
            group = [r for r in rows if r['problem'] == problem and r['variant'] == variant]
            cells.append(f"{statistics.median(r['regret'] for r in group):.6g}" + ("†" if any(r['error'] for r in group) else ""))
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    lines += ["", "| Problem | Method | Error min–max | Success ≤1e-6 | Evaluations min–max | Errors |", "|---|---|---|---:|---|---:|"]
    for problem, label in REFERENCE.PROBLEMS.items():
        for variant in variants:
            group = [r for r in rows if r['problem'] == problem and r['variant'] == variant]
            errors = [r['regret'] for r in group]
            lines.append(f"| {label} | {variant} | {min(errors):.6g}–{max(errors):.6g} | {sum(v<=1e-6 for v in errors)}/{len(group)} | {min(r['evaluations'] for r in group)}–{max(r['evaluations'] for r in group)} | {sum(bool(r['error']) for r in group)} |")
    lines += ["", f"{len(rows)} runs; {sum(bool(r['error']) for r in rows)} runtime errors. {sum(r.get('imports_verified', 0) for r in rows):,} foreign imports verified.", "", "Full records: runs.jsonl; library and runner fingerprints: manifest.json.", "", "Reproduce: `uv run python fractal-gas-web/tools/benchmark-gas-populations.py --output /tmp/gas-populations-comparison`."]
    (args.output / "report.md").write_text("\n".join(lines)+"\n")
    assert hashlib.sha256(REFERENCE.LIBRARY.read_bytes()).hexdigest() == manifest['library_sha256']
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
