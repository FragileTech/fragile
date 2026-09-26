"""Population sweep sharing the validated long-budget native runner."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import random
import statistics


HELPER = Path(__file__).with_name("benchmark-long-budget.py")
SPEC = importlib.util.spec_from_file_location("long_budget", HELPER)
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)
ROOT = BENCHMARK.ROOT
VARIANTS = [name for name in BENCHMARK.VARIANTS if name != "BIPOP-active CMA-ES"]
REFERENCE = ROOT / "tests/optimization/reports/long-budget-100k-20d-5elites"


def run_population_case(case):
    problem, variant, seed, budget, population = case
    result = BENCHMARK.run_case(
        (problem, 20, variant, seed, budget),
        overrides={"walkers": population, "max_walkers": 5000},
    )
    result["initial_walkers"] = population
    result["rounds"] = result["final_status"].get("controller", {}).get("round", 0) + 1
    settings = result["initial_status"]["effective_settings"]
    assert settings["walkers"] == population and settings["elites"] == 5
    assert settings["dimensions"] == 20 and settings["max_walkers"] == 5000
    assert result["maximum_population"] <= 5000
    if variant != "Bounded + basin restarts":
        assert result["minimum_population"] == result["maximum_population"] == population
    return result


def write_report(output, rows, reference, manifest):
    columns = [key for key in rows[0] if key not in ("config", "initial_status", "final_status")]
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    sizes = manifest["populations"]
    lines = ["# Population sweep at 20 dimensions", "",
             f"Starting populations: {sizes}. {manifest['seeds']} seeds per case; "
             f"{manifest['budget']:,} evaluations allowed per run; five elites; maximum population 5,000. "
             f"{len(rows)} measured fractal runs, {sum(bool(r['error']) for r in rows)} early-stop errors, "
             f"{sum(r['evaluations'] for r in rows):,} actual objective evaluations.", "",
             "All fractal methods use Wave. The three non-restarting variants keep the specified population. "
             "The basin controller retains its automatic population and scale schedules, so its starting "
             "population can change between rounds, up to 5,000. Initial populations are varied between "
             "independent runs, not edited during one run. Other settings remain those of the previous "
             "20D benchmark: Gaussian/local covariance standard deviation 0.2, adaptive scale bounds "
             "[0.0001, 0.2], nonperiodic boundaries, and COCO instance 1. No optimizer changes or tuning.", "",
             "CMA-ES uses its own default population and selection rule. Its unchanged 20D results are "
             "reused once as a reference, with the native-library fingerprint and evaluation budget "
             "verified. They are not relabeled as 128–5,000-walker CMA runs.", "",
             "Error is best evaluated objective minus the known optimum; lower is better. Success means "
             "error ≤ 1e-6. Failed runs retain their last best objective. A dagger (†) marks groups "
             "containing errors; inspect their evaluation counts. Only complete operations are admitted, "
             "so actual counts may fall below the allowance even without an error. The known outer "
             "session guard can stop an invalid population before saved elites are restored.", "",
             "At 100,000 evaluations, larger populations allow fewer movement/adaptation steps. The "
             "controller's round allowance is 50 × dimension × population and its stagnation interval "
             "is 10 × dimension × population. A controller-enabled run can therefore complete without "
             "any restart; round counts below distinguish that case from actual multi-round search.", "",
             "## Median error by starting population", "",
             "| Problem | Variant | " + " | ".join(f"{n:,}" for n in sizes) + " |",
             "|---|---|" + "---:|" * len(sizes)]
    groups = {}
    for problem, label in BENCHMARK.PROBLEMS.items():
        for variant in VARIANTS:
            values = []
            for population in sizes:
                group = [r for r in rows if (r["problem"], r["variant"], r["initial_walkers"]) == (problem, variant, population)]
                groups[problem, variant, population] = group
                values.append(f"{statistics.median(r['regret'] for r in group):.6g}" + ("†" if any(r["error"] for r in group) else ""))
            lines.append(f"| {label} | {variant} | " + " | ".join(values) + " |")
    lines += ["", "## Unchanged CMA-ES reference", "", "| Problem | Median error | Successes |", "|---|---:|---:|"]
    for problem, label in BENCHMARK.PROBLEMS.items():
        group = [r for r in reference if r["problem"] == problem]
        lines.append(f"| {label} | {statistics.median(r['regret'] for r in group):.6g} "
                     f"| {sum(r['regret'] <= 1e-6 for r in group)}/{len(group)} |")
    lines += ["", "## Variation, accounting, and actual population", "",
              "IQR uses inclusive 25th–75th percentiles. CPU time includes engine work and benchmark bookkeeping. "
              "Concurrent workers affect timings; these are not isolated latency measurements. "
              "Success counts and variability are descriptive, without significance tests.", "",
              "| Problem | Variant | Initial N | Median error | IQR | Successes | Errors | Evaluations min–max | Steps median | Actual N min–max | Rounds min–max | Median CPU seconds |",
              "|---|---|---:|---:|---|---:|---:|---|---:|---|---|---:|"]
    for (problem, variant, population), group in groups.items():
        values = [r["regret"] for r in group]
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [values[0]] * 3
        lines.append(
            f"| {BENCHMARK.PROBLEMS[problem]} | {variant} | {population} | {statistics.median(values):.6g} "
            f"| {q1:.4g}–{q3:.4g} | {sum(v <= 1e-6 for v in values)}/{len(values)} "
            f"| {sum(bool(r['error']) for r in group)} "
            f"| {min(r['evaluations'] for r in group)}–{max(r['evaluations'] for r in group)} "
            f"| {statistics.median(r['iterations'] for r in group):g} "
            f"| {min(r['minimum_population'] for r in group)}–{max(r['maximum_population'] for r in group)} "
            f"| {min(r['rounds'] for r in group)}–{max(r['rounds'] for r in group)} "
            f"| {statistics.median(r['cpu_seconds'] for r in group):.3f} |"
        )
    lines += ["", "## Reproduction", "", "```sh",
              "python3 fractal-gas-web/tools/benchmark-population-sweep.py "
              f"--populations {' '.join(map(str, sizes))} --seeds {manifest['seeds']} "
              f"--budget {manifest['budget']} --workers {manifest['workers']} "
              f"--output {output.relative_to(ROOT)}", "```", "",
              "Use `--report-only` to regenerate tables from saved results. "
              "The referenced CMA data and its provenance are saved alongside the measured runs. "
              "Full requested/effective configurations and final diagnostics are in runs.jsonl."]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--populations", type=int, nargs="+", default=[128, 512, 1024, 2048, 5000])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--budget", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", default="tests/optimization/reports/population-sweep-20d-100k")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if sorted(set(args.populations)) != args.populations or not all(128 <= n <= 5000 for n in args.populations):
        parser.error("Populations must be distinct ascending values from 128 to 5,000")
    if not 1 <= args.seeds <= 10 or args.budget != 100000 or args.workers < 1:
        parser.error("This comparison uses 1–10 seeds, 100,000 evaluations, and positive worker count")
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("A benchmark already owns this output directory")
    if args.report_only:
        manifest = json.loads((output / "manifest.json").read_text())
        rows = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
        reference = [json.loads(line) for line in (output / "cma-reference.jsonl").read_text().splitlines()]
    else:
        native_hash = hashlib.sha256(BENCHMARK.LIBRARY.read_bytes()).hexdigest()
        reference_manifest = json.loads((REFERENCE / "manifest.json").read_text())
        if native_hash != reference_manifest["library_sha256"] or reference_manifest["budget"] != args.budget:
            parser.error("CMA reference does not match the current native engine and budget")
        reference = [json.loads(line) for line in (REFERENCE / "runs.jsonl").read_text().splitlines()]
        reference = [r for r in reference if r["variant"] == "BIPOP-active CMA-ES" and r["seed"] < args.seeds]
        assert len(reference) == len(BENCHMARK.PROBLEMS) * args.seeds
        manifest = dict(populations=args.populations, seeds=args.seeds, budget=args.budget,
                        workers=args.workers, dimensions=20, elites=5, max_walkers=5000,
                        library_sha256=native_hash, reference_directory=str(REFERENCE.relative_to(ROOT)),
                        reference_sha256=hashlib.sha256((REFERENCE / "runs.jsonl").read_bytes()).hexdigest(),
                        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest(),
                        engine_sources={path: digest for path, digest in reference_manifest["sources"].items() if path.startswith("src/")})
        for path, digest in manifest["engine_sources"].items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
                parser.error(f"Engine source changed from measured reference: {path}")
        rows = []
        if args.resume:
            previous = json.loads((output / "manifest.json").read_text())
            for key in ("populations", "seeds", "budget", "library_sha256", "runner_sha256", "helper_sha256"):
                if previous[key] != manifest[key]:
                    parser.error(f"Resume mismatch: {key}")
            rows = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
        elif (output / "runs.jsonl").exists() and (output / "runs.jsonl").stat().st_size:
            parser.error("Output already contains results; use another directory or --resume")
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (output / "cma-reference.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reference))
        cases = [(problem, variant, seed, args.budget, population)
                 for population in args.populations for problem in BENCHMARK.PROBLEMS
                 for variant in VARIANTS for seed in range(args.seeds)]
        total = len(cases)
        done = {(r["problem"], r["variant"], r["seed"], r["budget"], r["initial_walkers"]) for r in rows}
        random.Random(20260925).shuffle(cases)
        cases = [case for case in cases if case not in done]
        with (output / "runs.jsonl").open("a") as stream, ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for future in as_completed([pool.submit(run_population_case, case) for case in cases]):
                row = future.result()
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(f"{len(rows)}/{total} N={row['initial_walkers']} {row['problem']} "
                      f"{row['variant']} seed={row['seed']} error={row['regret']:.6g} "
                      f"evals={row['evaluations']} rounds={row['rounds']} {row['error']}", flush=True)
    keys = {(r["problem"], r["variant"], r["seed"], r["initial_walkers"]) for r in rows}
    expected = len(BENCHMARK.PROBLEMS) * len(VARIANTS) * manifest["seeds"] * len(manifest["populations"])
    assert len(rows) == len(keys) == expected
    rows.sort(key=lambda r: (r["problem"], r["variant"], r["initial_walkers"], r["seed"]))
    write_report(output, rows, reference, manifest)


if __name__ == "__main__":
    main()
