"""Run and deeply verify the finite native parameter matrix, preserving raw data."""

from concurrent.futures import as_completed, ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "proof-validation/experiment-configs/estimate-completion-20261004"
OUTPUT = ROOT / "outputs/convergence/estimate-completion-20261004/parameters"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(name):
    target = OUTPUT / name
    command = [
        str(ROOT / "target/release/gas-structural-landscape"),
        str(CONFIG / "matrix.json"),
        str(target),
        "selected",
        str(CONFIG / f"{name}.json"),
    ]
    with (OUTPUT / f"{name}.log").open("w") as log:
        result = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode:
        raise RuntimeError(f"{name} failed; complete output retained, see log")
    verified = subprocess.run(
        [str(ROOT / "target/release/gas-proof-experiments"), "verify", str(target), "--deep"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    (OUTPUT / f"{name}-verification.json").write_text(verified.stdout)
    report = json.loads((target / "report.json").read_text())
    return {
        "profile": name,
        "dataset": str(target.relative_to(ROOT)),
        "native_parameters": report["native_parameters"],
        "summary": report["summary"],
        "verification": json.loads(verified.stdout),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    names = [
        "weak-selection-certified",
        "timestep-002",
        "timestep-006",
        "friction-05",
        "friction-2",
        "cap1-jitter0-diffusion025",
        "cap4-jitter02-diffusion005",
    ]
    snapshots = []
    for path in [
        Path(__file__),
        ROOT / "crates/benchmarks/src/bin/gas-structural-landscape.rs",
        ROOT / "crates/benchmarks/src/convergence_structural_rates.rs",
        ROOT / "crates/benchmarks/src/convergence_experiments_chapter05_dimension.rs",
        CONFIG / "matrix.json",
        *(CONFIG / f"{name}.json" for name in names),
    ]:
        snapshots.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": digest(path),
            "text": path.read_text(),
        })
    (OUTPUT / "execution-provenance.json").write_text(
        json.dumps(
            {
                "executable_sha256": digest(ROOT / "target/release/gas-structural-landscape"),
                "source_and_input_snapshots": snapshots,
                "parallel_native_jobs": 2,
                "profiles": names,
            },
            indent=2,
        )
        + "\n"
    )
    completed = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            item = future.result()
            completed.append(item)
            (OUTPUT / "progress.json").write_text(json.dumps(completed, indent=2) + "\n")
            print(item["profile"], json.dumps(item["summary"]), flush=True)
    summary = {
        "profiles": len(completed),
        "cases": sum(r["summary"]["cases"] for r in completed),
        "native_steps": sum(r["summary"]["native_steps"] for r in completed),
        "checks": sum(r["summary"]["checks"] for r in completed),
        "failures": sum(r["summary"]["failures"] for r in completed),
    }
    (OUTPUT / "catalog.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "datasets": completed,
                "scope": "Finite native parameter tests with normalized permutation-invariant observables. Moment and law-mixing certificates retain their own hypotheses.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
