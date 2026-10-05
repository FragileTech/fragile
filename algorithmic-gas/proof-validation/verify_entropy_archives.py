"""Independent full artifact and source audit for Chapters 10--12 outputs."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path

from verify_ch789_archives import CompleteReader


def digest(data):
    return hashlib.sha256(data).hexdigest()


def audit(path, refs, failures, cached=None):
    previous_failures = len(failures)
    row = {"path": str(path), "references": refs}
    encoded = path.read_bytes()
    if cached is not None and cached["sha256"] == digest(encoded):
        row.update({k: v for k, v in cached.items() if k not in {"references", "passed"}})
        for ref in refs:
            for key in ["sha256", "decoded_sha256", "compressed_bytes", "decoded_bytes"]:
                if key in ref["manifest"] and ref["manifest"][key] != row[key]:
                    failures.append({
                        "path": str(path),
                        "field": key,
                        "expected": ref["manifest"][key],
                        "actual": row[key],
                    })
        row["complete_parse_reused_after_exact_compressed_SHA_match"] = True
        row["passed"] = len(failures) == previous_failures
        return row
    raw = gzip.decompress(encoded)
    row.update(
        sha256=digest(encoded),
        decoded_sha256=digest(raw),
        compressed_bytes=len(encoded),
        decoded_bytes=len(raw),
        gzip_crc_verified=True,
    )
    for ref in refs:
        for key in ["sha256", "decoded_sha256", "compressed_bytes", "decoded_bytes"]:
            if key in ref["manifest"] and ref["manifest"][key] != row[key]:
                failures.append({
                    "path": str(path),
                    "field": key,
                    "expected": ref["manifest"][key],
                    "actual": row[key],
                })
    if path.name.endswith(".cbor.gz"):
        reader = CompleteReader(raw)
        reader.value()
        if reader.position != len(raw):
            message = "Trailing CBOR bytes"
            raise ValueError(message)
        row["complete_native_steps"] = reader.native_steps
        row["decode"] = "all definite CBOR items"
    elif path.name.endswith((".py.gz", ".md.gz")):
        raw.decode("utf-8")
        row["decode"] = "complete UTF8 source"
    else:
        value = json.loads(raw)
        row["decode"] = "complete JSON"
        if isinstance(value, dict) and isinstance(value.get("inventory"), str):
            inventory = json.loads(value["inventory"])
            for field in ["chapter", "chapter_source"]:
                if isinstance(value.get(field), str):
                    actual = digest(value[field].encode())
                    expected = inventory["source_sha256"]
                    if actual != expected:
                        failures.append({
                            "path": str(path),
                            "error": "Chapter/inventory source mismatch",
                            "expected": expected,
                            "actual": actual,
                        })
    row["passed"] = len(failures) == previous_failures
    return row


def audit_job(job):
    path, refs, cached = job
    failures = []
    try:
        row = audit(path, refs, failures, cached)
    except Exception as err:
        row = {"path": str(path), "references": refs, "passed": False, "error": str(err)}
        failures.append({"path": str(path), "error": str(err)})
    return row, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verified-cache", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    cache = {}
    if args.verified_cache:
        previous = json.loads(args.verified_cache.read_text())
        if previous["summary"]["failures"] or any(not r["passed"] for r in previous["artifacts"]):
            message = "Only a complete passing audit can supply parse reuse"
            raise ValueError(message)
        cache = {Path(r["path"]).resolve(): r for r in previous["artifacts"]}
    if args.output.exists():
        message = "Immutable audit output already exists"
        raise ValueError(message)
    datasets = []
    artifacts = {}
    metadata = []
    failures = []
    for index_path in sorted(root.rglob("archive-index.json")):
        index = json.loads(index_path.read_text())
        name = str(index_path.parent.relative_to(root))
        if index["status"] != "complete":
            failures.append({"dataset": name, "error": "Incomplete native dataset"})
        datasets.append({
            "dataset": name,
            "entries": len(index["entries"]),
            "index_sha256": digest(index_path.read_bytes()),
        })
        for entry in index["entries"]:
            path = (index_path.parent / entry["path"]).resolve()
            artifacts.setdefault(path, []).append({"dataset": name, "manifest": entry})
    # Derived Python ledgers retain their own lossless archive manifests.
    for report_path in sorted(root.rglob("*.json")):
        if report_path.name == "archive-index.json":
            continue
        value = json.loads(report_path.read_text())
        metadata.append({
            "path": str(report_path.relative_to(root)),
            "sha256": digest(report_path.read_bytes()),
        })
        # Audit reports quote manifests in their original dataset context.
        # Their nested copies are evidence, not new relative-path manifests.
        if report_path.is_relative_to(root / "integrity"):
            continue

        def visit(item):
            if isinstance(item, dict):
                if "path" in item and "sha256" in item and str(item["path"]).endswith(".gz"):
                    path = Path(item["path"])
                    if not path.is_absolute():
                        candidates = {
                            (base / path).resolve()
                            for base in [report_path.parent, Path.cwd(), *root.parents]
                            if (base / path).is_file()
                        }
                        if len(candidates) != 1:
                            failures.append({
                                "report": str(report_path),
                                "path": str(path),
                                "error": "Archive manifest path must resolve uniquely",
                                "candidates": sorted(map(str, candidates)),
                            })
                            return
                        path = candidates.pop()
                    if path.exists():
                        artifacts.setdefault(path.resolve(), []).append({
                            "dataset": str(report_path.relative_to(root)),
                            "manifest": item,
                        })
                for child in item.values():
                    visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)

        visit(value)
    # Every compressed file under the new experiment root must be manifest-bound.
    for path in root.rglob("*.gz"):
        if path.resolve() not in artifacts:
            failures.append({"path": str(path), "error": "Compressed artifact lacks SHA manifest"})
    rows = []
    jobs = [(path, refs, cache.get(path)) for path, refs in sorted(artifacts.items())]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for number, (row, local_failures) in enumerate(executor.map(audit_job, jobs), 1):
            rows.append(row)
            failures.extend(local_failures)
            if number % 100 == 0:
                print(
                    json.dumps({
                        "audited": number,
                        "total": len(artifacts),
                        "failures": len(failures),
                    }),
                    flush=True,
                )
    result = {
        "helper_sha256": digest(Path(__file__).read_bytes()),
        "reused_parse_audit": str(args.verified_cache.resolve()) if args.verified_cache else None,
        "reused_parse_audit_sha256": digest(args.verified_cache.read_bytes())
        if args.verified_cache
        else None,
        "datasets": datasets,
        "metadata": metadata,
        "artifacts": rows,
        "failures": failures,
        "summary": {
            "datasets": len(datasets),
            "unique_archives": len(rows),
            "compressed_bytes": sum(r.get("compressed_bytes", 0) for r in rows),
            "decoded_bytes": sum(r.get("decoded_bytes", 0) for r in rows),
            "complete_native_recorded_steps": sum(r.get("complete_native_steps", 0) for r in rows),
            "failures": len(failures),
        },
        "scope": "Integrity and complete parsing; empirical theory credit remains in the exact expression ledgers.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]))
    if failures:
        message = "Artifact audit failed; retained details identify each failure"
        raise SystemExit(message)


if __name__ == "__main__":
    main()
