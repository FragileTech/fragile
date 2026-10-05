"""Extract checksum-bound frozen source frames for the existing KUR register."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

from read_native_cbor import first_native_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    frames = []
    snapshots = []
    for root in args.roots:
        index_bytes = (root / "archive-index.json").read_bytes()
        index = json.loads(index_bytes)
        snapshots.append({
            "root": str(root.resolve()),
            "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        })
        for entry in index["entries"]:
            match = re.fullmatch(
                r"(.+)(?:/rep(\d+)/left_through\d+|-rep(\d+)-left-\d+)", entry["tag"]
            )
            if entry["kind"] != "native_run_archive" or not match:
                continue
            replicate = int(match[2] or match[3])
            if replicate != 0 and entry["metadata"]["first_step"] != 1:
                continue
            raw = (root / entry["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                msg = "Retained native artifact hash mismatch"
                raise ValueError(msg)
            archive = first_native_step(gzip.decompress(raw), verify_remainder=False)
            step = archive["steps"][0]

            def stage(name, field):
                return next(s["fields"][field] for s in step["stages"] if s["stage"] == name)

            source = stage("pre_clone", "positions")
            velocities = stage("pre_clone", "velocities")
            matching = [
                a
                for a in archive["anchors"]
                if a["population"]["observations"]["fields"]["positions"]["values"]
                == source["values"]
            ]
            eligible = bool(matching) and all(
                not any(mark.values()) for mark in matching[-1]["population"]["validity"]
            )
            force_records = []
            for kick in ["B1", "B2"]:
                gradient = next(
                    f["values"]
                    for f in step["field_evaluations"]
                    if f["stage"] == kick and f["field"] == "potential_gradient"
                )
                force_records.append({
                    "kick": kick,
                    "positions": stage(f"{kick}_input", "positions")["values"],
                    "gradient": gradient,
                })

            def centered_variance(field):
                d = field["item_shape"][0]
                rows = [field["values"][i : i + d] for i in range(0, len(field["values"]), d)]
                means = [sum(row[j] for row in rows) / len(rows) for j in range(d)]
                return sum(
                    sum((x - mean) ** 2 for x, mean in zip(row, means, strict=True))
                    for row in rows
                ) / len(rows)

            frames.append({
                "id": f"{root.name}/{entry['tag']}/first_recorded_source_frame",
                "artifact": {
                    "root": str(root.resolve()),
                    "path": entry["path"],
                    "sha256": entry["sha256"],
                    "first_recorded_step": entry["metadata"]["first_step"],
                },
                "gas_config": archive["gas_config"],
                "providers": archive["providers"],
                "dimension": source["item_shape"][0],
                "group": f"{root.name}/{match[1]}",
                "replicate": replicate,
                "first_recorded_step": entry["metadata"]["first_step"],
                "zero_jitter_source_variance": centered_variance(
                    stage("literal_clone", "positions")
                ),
                "completed_positional_variance": centered_variance(stage("terminal", "positions")),
                "frozen_source_positions": source["values"],
                "frozen_source_velocities": velocities["values"],
                "all_entering_rows_eligible": eligible,
                "force_records": force_records,
            })
    result = {
        "native_steps": 0,
        "selection": "First step of every independent left replica for ensemble KUR comparisons, plus first step of later rep0 chunks for source-phase crossing diagnostics. Common-noise right sides are not additional samples; duplicate roots stay in separate groups.",
        "decoding_scope": "Every compressed artifact SHA is checked. Source configuration/anchors/selected first-step CBOR prefix decoded; this extraction does not replace original complete archive deep validation.",
        "index_snapshots": snapshots,
        "frames": frames,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Extracted {len(frames)} checksum-bound source frames")


if __name__ == "__main__":
    main()
