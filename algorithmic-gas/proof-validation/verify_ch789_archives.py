"""Independently audit immutable final artifact integrity, without theory credit."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from read_native_cbor import FirstStepReader


DATASETS = [
    "chapter07/native-laws-v1",
    "chapter07/poisson-clouds-v1",
    "chapter07/diffusion-reference-v1",
    "chapter08/full-v1",
    "chapter08/final-ledger-v3",
    "chapter08/viscosity-v2",
    "chapter09/bridge-full-v1",
    "chapter09/independent-native-memory-v1",
    "chapter09/native-resonance-v1",
    "chapter09/distance-variation-review-v3",
    "chapter09/population-scaling-v1",
    "chapter09/horizon-variance-v1",
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class CompleteReader(FirstStepReader):
    """Validate all definite CBOR items, retaining only top-level metadata."""

    native_steps = 0

    def first_array(self, depth):
        major, _, payload = self.header()
        if major != 4:
            message = "Native steps must be a definite CBOR array"
            raise ValueError(message)
        count = payload if isinstance(payload, int) else int.from_bytes(payload, "big")
        self.native_steps = count
        for _ in range(count):
            self.value(depth + 1, keep=False)
        return []


def resolve(path, cwd):
    path = Path(path)
    if path.is_absolute():
        return path
    for candidate in [cwd / path, cwd / "algorithmic-gas" / path]:
        if candidate.exists():
            return candidate
    return cwd / path


def audit_artifact(path, value, failures, sources):
    row = {"path": str(path), "references": value["references"]}
    previous_failures = len(failures)
    compressed = path.read_bytes()
    row.update({"sha256": sha(compressed), "compressed_bytes": len(compressed)})
    with gzip.open(path, "rb") as stream:
        decoded = stream.read()
    row.update({
        "decoded_sha256": sha(decoded),
        "decoded_bytes": len(decoded),
        "gzip_crc_verified": True,
    })
    if path.name.endswith(".cbor.gz"):
        decoder = CompleteReader(decoded)
        payload = decoder.value()
        if decoder.position != len(decoded):
            message = "Trailing CBOR bytes"
            raise ValueError(message)
        row.update({
            "decode": "complete definite CBOR structural parse",
            "native_steps": decoder.native_steps,
        })
    elif path.name.endswith((".py.gz", ".md.gz")):
        payload = decoded.decode("utf-8")
        row["decode"] = "complete UTF8 source text decode"
    else:
        payload = json.loads(decoded)
        row["decode"] = "complete UTF8 JSON parse"
    for reference in value["references"]:
        entry = reference["manifest"]
        for key in ["sha256", "compressed_bytes", "decoded_sha256", "decoded_bytes"]:
            if key in entry and entry[key] != row[key]:
                failures.append({
                    "path": str(path),
                    "dataset": reference["dataset"],
                    "field": key,
                    "expected": entry[key],
                    "actual": row[key],
                })
    if isinstance(payload, dict):
        for field in ["chapter", "chapter09"]:
            text = payload.get(field)
            if isinstance(text, str):
                inventory = payload.get("inventory")
                expected = (
                    json.loads(inventory).get("source_sha256")
                    if isinstance(inventory, str)
                    else None
                )
                snapshot_sha = sha(text.encode())
                if expected is not None and snapshot_sha != expected:
                    failures.append({
                        "path": str(path),
                        "error": "execution chapter does not match archived inventory hash",
                        "expected": expected,
                        "actual": snapshot_sha,
                    })
                sources.append({
                    "archive": str(path),
                    "field": field,
                    "snapshot_sha256": snapshot_sha,
                    "inventory_source_sha256": expected,
                    "snapshot_inventory_matches": expected is None or expected == snapshot_sha,
                })
    row["passed"] = len(failures) == previous_failures
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cwd = Path.cwd()
    datasets = []
    artifacts = {}
    failures = []
    missing_hashes = []
    sources = []
    hash_checks = []

    def record_hash(path, expected, owner, role):
        path = Path(path)
        row = {"path": str(path), "owner": owner, "role": role, "expected": expected}
        if not path.exists():
            row.update({"passed": False, "error": "missing file"})
            failures.append(row)
        else:
            row["actual"] = sha(path.read_bytes())
            row["passed"] = expected is None or expected == row["actual"]
            if not row["passed"]:
                failures.append(row)
        hash_checks.append(row)
        return row

    def add_artifact(directory, entry, owner):
        path = directory / entry["path"]
        key = str(path.resolve())
        if key not in artifacts:
            artifacts[key] = {"path": str(path), "references": []}
        artifacts[key]["references"].append({"dataset": owner, "manifest": entry})
        if not entry.get("sha256"):
            missing_hashes.append({
                "path": str(path),
                "dataset": owner,
                "reason": "manifest has no compressed SHA256",
            })

    for name in DATASETS:
        directory = args.root / name
        row = {"dataset": name, "directory": str(directory), "metadata": {}}
        if not directory.exists():
            failures.append({"dataset": name, "error": "missing requested dataset"})
            datasets.append(row)
            continue
        metadata = {}
        for filename in [
            "archive-index.json",
            "verification.json",
            "report.json",
            "provenance.json",
        ]:
            path = directory / filename
            if path.exists():
                metadata[filename] = read_json(path)
                row["metadata"][filename] = {"path": str(path), "sha256": sha(path.read_bytes())}
        index = metadata.get("archive-index.json", {})
        report = metadata.get("report.json", {})
        verification = metadata.get("verification.json", {})
        row["reported_summary"] = report.get(
            "summary", metadata.get("provenance.json", {}).get("summary")
        )
        row["reported_verification"] = verification
        if "failures" in report:
            row["reported_failures"] = report["failures"]
        if "check_count" in report:
            row["reported_check_count"] = report["check_count"]
        if "native_population_comparisons" in report:
            native = report["native_population_comparisons"]
            row["reported_native_population_summary"] = native.get("summary")
        entries = index.get("entries", []) or verification.get("artifacts", [])
        if name == "chapter08/final-ledger-v3":
            base = resolve(report["joined_inputs"]["base"]["path"], cwd).parent
            for entry in report["supplement_checks"]["archives"]:
                add_artifact(base, entry, name)
            record_hash(
                args.root / "chapter08/full-v1/archive-index.json",
                report["input_archive_index_sha256"],
                name,
                "native upstream index",
            )
            record_hash(
                Path(__file__).with_name("chapter08_inventory.json"),
                report["source_inventory_sha256"],
                name,
                "source inventory",
            )
            for role, joined in report["joined_inputs"].items():
                record_hash(
                    resolve(joined["path"], cwd), joined["sha256"], name, f"joined {role} report"
                )
            for filename, expected in verification["outputs"].items():
                record_hash(directory / filename, expected, name, "final ledger output")
        elif name == "chapter08/viscosity-v2":
            entries = report["derived_archives"]
            upstream = resolve(report["input_dataset"], cwd)
            record_hash(
                upstream / "archive-index.json",
                report["input_archive_index_sha256"],
                name,
                "upstream index",
            )
        elif name == "chapter07/diffusion-reference-v1":
            provenance = metadata["provenance.json"]
            entries = [
                {
                    "path": "raw.json.gz",
                    "sha256": provenance["compressed_sha256"],
                    "decoded_sha256": provenance["raw_sha256"],
                    "decoded_bytes": provenance["decoded_bytes"],
                    "compressed_bytes": provenance["compressed_bytes"],
                }
            ]
            record_hash(
                directory / "raw.json",
                provenance["raw_sha256"],
                name,
                "uncompressed reference data",
            )
            record_hash(
                directory / "source.rs",
                provenance["source_sha256"],
                name,
                "execution source snapshot",
            )
        for entry in entries:
            add_artifact(directory, entry, name)
        if "source_sha256" in report:
            chapter = int(name.split("/")[0].removeprefix("chapter"))
            filenames = {
                7: "07_discrete_qsd.md",
                8: "08_mean_field.md",
                9: "09_propagation_chaos.md",
            }
            current = cwd / "docs/source/2_fractal_gas/convergence_program" / filenames[chapter]
            actual = sha(current.read_bytes())
            snapshot = directory / "source.md"
            sources.append({
                "dataset": name,
                "expected_source_sha256": report["source_sha256"],
                "current_source_sha256": actual,
                "matches_current": actual == report["source_sha256"],
                "snapshot_path": str(snapshot) if snapshot.exists() else None,
                "snapshot_sha256": sha(snapshot.read_bytes()) if snapshot.exists() else None,
                "scope": "Current text may differ from the immutable execution snapshot; source difference alone is not an artifact-integrity failure.",
            })
            if snapshot.exists():
                record_hash(snapshot, report["source_sha256"], name, "execution chapter snapshot")
        if name == "chapter09/distance-variation-review-v3":
            record_hash(directory / "helper.py", report["helper_sha256"], name, "execution helper")
            inventory = directory / "inventory.json"
            if inventory.exists():
                record_hash(
                    inventory, report["inventory_sha256"], name, "source inventory snapshot"
                )
            for filename, expected in [
                ("execution-helper.py.gz", report["helper_sha256"]),
                ("chapter09-source.md.gz", report["source_sha256"]),
                ("chapter09-inventory.json.gz", report["inventory_sha256"]),
            ]:
                with gzip.open(directory / filename, "rb") as stream:
                    actual = sha(stream.read())
                check = {
                    "path": str(directory / filename),
                    "owner": name,
                    "role": "decoded execution snapshot",
                    "expected": expected,
                    "actual": actual,
                    "passed": actual == expected,
                }
                hash_checks.append(check)
                if not check["passed"]:
                    failures.append(check)
            record_hash(
                args.root / "chapter08/full-v1/archive-index.json",
                report["native_archive_index_sha256"],
                name,
                "native upstream index",
            )
        if name == "chapter09/population-scaling-v1":
            upstream = args.root / "chapter08/full-v1"
            record_hash(
                upstream / "archive-index.json",
                report["input_index_sha256"],
                name,
                "native upstream index",
            )
            record_hash(
                upstream / "report.json",
                report["input_report_sha256"],
                name,
                "native upstream report",
            )
            for path in directory.glob("*.py"):
                if sha(path.read_bytes()) == report["runner_sha256"]:
                    record_hash(path, report["runner_sha256"], name, "execution scaling helper")
        for upstream in report.get("upstream", []):
            record_hash(
                resolve(upstream["dataset"], cwd) / "archive-index.json",
                upstream["index_sha256"],
                name,
                "upstream index",
            )
        if name == "chapter07/native-laws-v1":
            upstream_paths = sorted({entry["dataset"] for entry in report["consumed_sources"]})
            row["retained_input_indexes"] = []
            for path in upstream_paths:
                source_index = resolve(path, cwd) / "archive-index.json"
                row["retained_input_indexes"].append(
                    record_hash(source_index, None, name, "retained source index")
                )
        row["manifest_artifacts"] = (
            len(entries)
            if name != "chapter08/final-ledger-v3"
            else len(report["supplement_checks"]["archives"])
        )
        datasets.append(row)

    audited = []
    for number, value in enumerate(artifacts.values(), 1):
        path = Path(value["path"])
        row = {"path": str(path), "references": value["references"]}
        try:
            row = audit_artifact(path, value, failures, sources)
        except Exception as exc:
            row.update({"passed": False, "error": str(exc)})
            failures.append({"path": str(path), "error": str(exc)})
        audited.append(row)
        if number % 250 == 0:
            print(
                json.dumps({
                    "audited_artifacts": number,
                    "total": len(artifacts),
                    "failures": len(failures),
                }),
                flush=True,
            )

    for row in datasets:
        selected = [
            entry
            for entry in audited
            if any(ref["dataset"] == row["dataset"] for ref in entry["references"])
        ]
        row["audited_artifacts"] = len(selected)
        row["audited_compressed_bytes"] = sum(
            entry.get("compressed_bytes", 0) for entry in selected
        )
        row["audited_native_steps"] = sum(entry.get("native_steps", 0) for entry in selected)
        verification = row.get("reported_verification", {})
        for field, observed in [
            ("archives", len(selected)),
            ("compressed_bytes", row["audited_compressed_bytes"]),
            ("native_recorded_steps", row["audited_native_steps"]),
        ]:
            if (
                field in verification
                and isinstance(verification[field], int)
                and verification[field] != observed
            ):
                failures.append({
                    "dataset": row["dataset"],
                    "field": field,
                    "expected": verification[field],
                    "actual": observed,
                })
    result = {
        "scope": "Independent final artifact integrity audit only. Gzip CRC, compressed SHA, recorded decoded SHA/length, complete JSON/CBOR structure, source snapshots and upstream report/index hashes. This does not assert mathematical theorem validation.",
        "excluded": "Obsolete versions and smoke datasets; pending Chapter7 replacement native audit is owned by root.",
        "summary": {
            "datasets": len(datasets),
            "unique_compressed_artifacts": len(audited),
            "compressed_bytes": sum(entry.get("compressed_bytes", 0) for entry in audited),
            "decoded_bytes": sum(entry.get("decoded_bytes", 0) for entry in audited),
            "native_recorded_steps": sum(entry.get("native_steps", 0) for entry in audited),
            "failures": len(failures),
            "materially_missing_hashes": len(missing_hashes),
        },
        "datasets": datasets,
        "artifacts": audited,
        "hash_checks": hash_checks,
        "source_snapshots": sources,
        "failures": failures,
        "missing_hashes": missing_hashes,
        "audit_runner_sha256": sha(Path(__file__).read_bytes()),
        "cbor_decoder_sha256": sha(Path(__file__).with_name("read_native_cbor.py").read_bytes()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
