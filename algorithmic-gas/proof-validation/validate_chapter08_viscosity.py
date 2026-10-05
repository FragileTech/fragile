"""Verify both explicit viscosity conventions using retained actual kinetic inputs."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from read_native_cbor import first_native_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    index = json.loads((args.dataset / "archive-index.json").read_text(encoding="utf-8"))
    if index["status"] not in {"complete", "completed"}:
        message = "Only completed native archives may be reused"
        raise ValueError(message)
    batches, comparisons, failed, updates = [], 0, 0, 0
    source = (
        Path(__file__).parents[2]
        / "docs/source/2_fractal_gas/convergence_program/08_mean_field.md"
    )
    profiles = {}
    for entry in index["entries"]:
        if entry["kind"] != "native_run_archive":
            continue
        if entry["metadata"]["recorded_steps"] != 1:
            message = "This explicit replay expects one complete step per retained archive"
            raise ValueError(message)
        compressed = (args.dataset / entry["path"]).read_bytes()
        if hashlib.sha256(compressed).hexdigest() != entry["sha256"]:
            message = "Native compressed archive checksum mismatch"
            raise ValueError(message)
        archive = first_native_step(gzip.decompress(compressed))
        cfg = archive["gas_config"]
        viscosity = cfg["qft"]["viscosity"]
        if not viscosity or cfg["qft"]["graph_viscosity"] or cfg["qft"]["curl"]:
            message = "Requires the explicit dense viscosity specialization"
            raise ValueError(message)
        step = archive["steps"][0]
        rows = []
        for stage in ["B1", "B2"]:
            before = next(s for s in step["stages"] if s["stage"] == stage + "_input")
            xfield, vfield = before["fields"]["positions"], before["fields"]["velocities"]
            x, v, n, d = (
                xfield["values"],
                vfield["values"],
                len(xfield["values"]) // xfield["item_shape"][0],
                xfield["item_shape"][0],
            )
            # The stored experiment has no interior boundary test: every
            # revived row enters each kick. Check the actual marks rather than
            # use a pre-clone donor count for this normalization.
            alive = [
                not (a["terminated"] or a["out_of_bounds"] or a["truncated"])
                for a in before["validity"]
            ]
            if not all(alive):
                message = "Expected fully revived canonical kinetic input"
                raise ValueError(message)
            values = {
                f["field"]: f["values"] for f in step["field_evaluations"] if f["stage"] == stage
            }
            for i in range(n):
                weights = [0.0] * n
                for j in range(n):
                    if i != j:
                        squared = sum((x[i * d + a] - x[j * d + a]) ** 2 for a in range(d))
                        weights[j] = math.exp(-squared / (2 * viscosity["bandwidth"] ** 2))
                mass = sum(weights)
                denominator = mass if viscosity["row_normalized"] else n
                for a in range(d):
                    actual = values["viscous_force"][i * d + a]
                    numerator = sum(
                        w * (v[j * d + a] - v[i * d + a]) for j, w in enumerate(weights)
                    )
                    prediction = (
                        viscosity["coefficient"] * numerator / denominator if denominator else 0.0
                    )
                    allowance = 1e-10 * (1 + abs(actual) + abs(prediction))
                    passed = abs(actual - prediction) <= allowance
                    failed += not passed
                    comparisons += 1
                    row = {
                        "id": f"{entry['tag']}-{stage}-{i}-{a}",
                        "source_label": "remark-separation-kinetic-death",
                        "source_formula_ids": [
                            "chapter08-expression-0075",
                            "chapter08-expression-0076"
                            if viscosity["row_normalized"]
                            else "chapter08-expression-0077",
                        ],
                        "actual": actual,
                        "predicted": prediction,
                        "passed": passed,
                        "allowance": allowance,
                        "stage": stage,
                        "row": i,
                        "coordinate": a,
                        "N": n,
                        "d": d,
                        "nu": viscosity["coefficient"],
                        "bandwidth": viscosity["bandwidth"],
                        "normalization": "row" if viscosity["row_normalized"] else "count",
                        "N_normalized_numerator": numerator / n,
                        "N_normalized_denominator": denominator / n,
                        "actual_current_input_position": x[i * d : (i + 1) * d],
                        "actual_current_input_velocity": v[i * d : (i + 1) * d],
                        "source_archive": entry["path"],
                        "source_sha256": entry["sha256"],
                        "scope": "Actual simultaneous full-component collision output and actual intermediate kinetic population; every donor velocity change remains in the inputs. This verifies the separate viscosity formula, not canonical zero-viscosity chaos rates.",
                    }
                    if viscosity["row_normalized"]:
                        empirical_population_force = (
                            viscosity["coefficient"] * numerator / (mass + 1)
                        )
                        finite_self_error = actual - empirical_population_force
                        expected_self_error = prediction / (mass + 1)
                        row.update({
                            "full_atomic_empirical_population_force": empirical_population_force,
                            "finite_self_exclusion_error": finite_self_error,
                            "predicted_self_exclusion_error": expected_self_error,
                            "self_error_passed": abs(finite_self_error - expected_self_error)
                            <= allowance,
                        })
                        failed += not row["self_error_passed"]
                        comparisons += 1
                    total = values["total_force"][i * d + a]
                    predicted_total = -values["potential_gradient"][i * d + a] + prediction
                    row["total_force_passed"] = abs(total - predicted_total) <= 1e-10 * (
                        1 + abs(total) + abs(predicted_total)
                    )
                    row["actual_total_force"] = total
                    row["predicted_total_force"] = predicted_total
                    failed += not row["total_force_passed"]
                    comparisons += 1
                    rows.append(row)
            profile = f"d{d}-N{n}-{'row' if viscosity['row_normalized'] else 'count'}"
            profiles[profile] = profiles.get(profile, 0) + 1
        updates += 1
        batches.extend(rows)
        if len(batches) >= 4096:
            path = args.output / f"rows-{updates}.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                json.dump({"rows": batches}, stream, separators=(",", ":"))
            batches = []
    if batches:
        with gzip.open(args.output / "rows-final.json.gz", "wt", encoding="utf-8") as stream:
            json.dump({"rows": batches}, stream, separators=(",", ":"))
    verification = []
    for path in sorted(args.output.glob("rows-*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            decoded = json.load(stream)
        verification.append({
            "path": path.name,
            "rows": len(decoded["rows"]),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "compressed_bytes": path.stat().st_size,
            "deep_json_verified": True,
        })
    report = {
        "chapter": 8,
        "source_label": "remark-separation-kinetic-death",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "input_archive_index_sha256": hashlib.sha256(
            (args.dataset / "archive-index.json").read_bytes()
        ).hexdigest(),
        "input_dataset": str(args.dataset),
        "summary": {
            "retained_native_complete_updates": updates,
            "new_native_updates": 0,
            "comparisons": comparisons,
            "failed": failed,
        },
        "profiles": profiles,
        "derived_archives": verification,
        "scope": "Both explicit native count/row Gaussian viscosity operators at their actual B1/B2 populations. Row finite self-exclusion is retained and measured separately from the full atomic empirical population integral; component-Haar canonical ν0 theorem remains unchanged.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "helper.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "read_native_cbor.py").write_bytes(
        Path(__file__).with_name("read_native_cbor.py").read_bytes()
    )
    print(json.dumps(report["summary"]))
    if failed:
        message = "Viscosity formula comparison failed"
        raise SystemExit(message)


if __name__ == "__main__":
    main()
