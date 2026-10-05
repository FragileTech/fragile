"""Verify and freeze the completed Chapter 6 derived evidence catalog."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports, artifacts = [], []
    for path in sorted(args.root.glob("*/report.json")):
        if path.parent.name in {"native-displacement-v1", "weak-native-moment-closure"}:
            continue
        value = json.loads(path.read_text())
        summary = value["summary"]
        if summary.get("comparisons_failed", 0):
            message = f"Failed Chapter 6 report: {path}"
            raise ValueError(message)
        reports.append({"path": str(path.resolve()), "sha256": sha(path), "summary": summary})
        index_path = path.parent / "archive-index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            if index["status"] != "complete":
                message = f"Incomplete index: {index_path}"
                raise ValueError(message)
            for entry in index["entries"]:
                archive = path.parent / entry["path"]
                digest = sha(archive)
                if (
                    digest != entry["sha256"]
                    or archive.stat().st_size != entry["compressed_bytes"]
                ):
                    message = f"Derived checksum mismatch: {archive}"
                    raise ValueError(message)
                decoded = json.loads(gzip.decompress(archive.read_bytes()))
                if entry["kind"] != "raw_json" or not isinstance(decoded, (dict, list)):
                    message = f"Unexpected derived artifact schema: {archive}"
                    raise ValueError(message)
                artifacts.append({
                    "path": str(archive.resolve()),
                    "sha256": digest,
                    "compressed_bytes": entry["compressed_bytes"],
                    "deep_decode": "gzip JSON success",
                })
    extra_bindings = []
    for name in ["statement-ledger-v2", "estimates-table-v5"]:
        for path in sorted((args.root / name).glob("*")):
            if path.is_file():
                extra_bindings.append({"path": str(path.resolve()), "sha256": sha(path)})
    summary = {
        "current_evidence_reports": len(reports),
        "derived_indexed_artifacts": len(artifacts),
        "derived_compressed_bytes": sum(e["compressed_bytes"] for e in artifacts),
        "native_steps_added": 0,
        "checksums_verified": True,
        "deep_decode": True,
        "status": "complete",
    }
    report = {
        "summary": summary,
        "reports": reports,
        "artifacts": artifacts,
        "ledger_and_table_files": extra_bindings,
        "executed_helper_sha256": sha(Path(__file__)),
        "scope": "Every indexed derived gzipJSON checked/decompressed/parsed. Native CBOR archives were checksum checked and validated upon original ingestion and parent deep verification; source/native manifest and binary provenance remain in each immutable source artifact. This is derived-data verification, not a second native simulation.",
    }
    (args.output / "catalog.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "executed-helper.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
