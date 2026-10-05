"""Bind retained exact Gaussian cubature to the current quadratic weak theorem."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.archive_root.resolve()
    if args.output.resolve().is_relative_to(root):
        msg = "Derived certificate must preserve the immutable archive"
        raise ValueError(msg)
    index = json.loads((root / "archive-index.json").read_text())
    for entry in index["entries"]:
        if digest(root / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {entry['path']}")
        if entry["kind"] == "native_run_archive":
            config = entry["metadata"]["native_config"]
            kinetic = config["kinetic"]
            assert kinetic["velocity_cap"] is None
            assert kinetic["position_diffusion"] == 0
            assert kinetic["integrator"]["friction"] == 1
            assert config["boundary"]["kind"] == "unbounded"
            assert config["fitness"]["reward_exponent"] == 0
            assert config["fitness"]["diversity_exponent"] == 0
            assert entry["metadata"]["providers"]["gradient"].endswith("unit_quadratic/v1")
    entry = next(item for item in index["entries"] if item["tag"] == "cubature/full-node-plan")
    plan = json.loads(gzip.decompress((root / entry["path"]).read_bytes()))
    records = plan["records"]
    assert len(records) == 96
    assert abs(sum(record["weight"] for record in records) - 1) < 1e-14
    for record in records:
        assert math.isclose(record["weight"], 1 / 96, rel_tol=0, abs_tol=1e-18)
        assert abs(record["coordinate_value"]) == math.sqrt(48)
        assert record["input"] == [0.8, 0.4]
        coordinates = [
            coordinate
            for leaf in record["leaf_inputs"]
            for coordinate in leaf["standard_gaussian_coordinates"]
        ]
        assert sum(coordinate != 0 for coordinate in coordinates) == 1
        assert coordinates[record["nonzero_coordinate"]] == record["coordinate_value"]
    assert sorted(
        (record["nonzero_coordinate"], math.copysign(1, record["coordinate_value"]))
        for record in records
    ) == [(i, sign) for i in range(48) for sign in [-1.0, 1.0]]
    h_max, horizon, split_norm = 0.04, 0.16, 3
    generator_norm = (1 + math.sqrt(5)) / 2
    u = math.sqrt((h_max / 2) ** 2 + (1 + h_max**2 / 4) ** 2)
    u1, u2 = math.sqrt(0.25 + (h_max / 2) ** 2), 0.5
    ca = (
        split_norm**3 * math.exp(split_norm * h_max)
        + generator_norm**3 * math.exp(generator_norm * h_max)
    ) / 6
    cq = (
        4 * u**2
        + 12 * u * u1
        + 6 * (u1**2 + u * u2)
        + 6 * h_max * u1 * u2
        + 4 * generator_norm**2 * math.exp(2 * generator_norm * h_max)
    ) / 6
    m2 = math.exp(2 * generator_norm * horizon) * (0.8 + horizon)
    weak = (
        horizon
        * math.exp(2 * split_norm * horizon)
        * (ca * (math.exp(split_norm * h_max) + math.exp(generator_norm * h_max)) * m2 + 2 * cq)
    )
    source = (
        Path(__file__).resolve().parents[2]
        / "docs/source/2_fractal_gas/convergence_program/05_kinetic_contraction.md"
    )
    formula = r"""|\mathbb E f(Z_T^{h})-\mathbb E f(Z_T)|\le C_{\rm weak}h^2,
\qquad C_{\rm weak}=12.29925819\ldots.                 \tag{5.D6}"""
    assert formula in source.read_text()
    rows = []
    original = json.loads((root / "report.json").read_text())
    for level in range(3):
        h = records[0]["native_outputs"][level]["h"]
        bias = sum(
            record["weight"]
            * (
                sum(value**2 for value in record["native_outputs"][level]["output"])
                - sum(value**2 for value in record["exact_output"])
            )
            for record in records
        )
        old = next(
            row
            for row in original["comparisons"]
            if row["id"] == f"cubature/h{h}/analytic_weak_prefactor"
        )
        assert abs(abs(bias) - old["observed"]) < 1e-14
        assert abs(weak * h**2 - old["bound"]) < 1e-14
        rows.append({
            "id": f"current_theorem/h{h}/quadratic_weak_bound",
            "source_labels": ["prop-explicit-constants"],
            "source_formula": formula,
            "observed": abs(bias),
            "bound": weak * h**2,
            "relation": "upper",
            "passed": abs(bias) <= weak * h**2,
            "standard_error": 0.0,
            "scope": "Exact degree-two Gaussian cubature of the uncapped unit-quadratic specialization, not independent stochastic replicas",
            "hypotheses": {
                "h": h,
                "T": horizon,
                "H": h_max,
                "initial_squared_norm": 0.8,
                "normalized_population_coefficient": True,
            },
        })
    report = {
        "chapter": 5,
        "source_path": str(source),
        "source_sha256": digest(source),
        "source_snapshot_scope": "Current repaired formal source; retained raw experiment snapshots remain unchanged",
        "raw_dataset": str(root),
        "raw_plan": entry["path"],
        "raw_plan_sha256": entry["sha256"],
        "raw_report_sha256": digest(root / "report.json"),
        "analysis_script_sha256": digest(Path(__file__)),
        "new_engine_steps": 0,
        "constants": {"C_A": ca, "C_Q": cq, "M2": m2, "C_weak": weak},
        "comparisons": rows,
        "summary": {
            "comparisons": len(rows),
            "comparisons_failed": sum(not row["passed"] for row in rows),
            "archive_checksums_verified": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
