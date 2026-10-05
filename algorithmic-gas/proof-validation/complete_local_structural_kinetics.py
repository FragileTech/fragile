"""Extend a closed local kinetic audit to final native indexes using verified ledgers."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import audit_local_structural_kinetics as audit


def add_cached_records(stats, payload, certificates):
    """Recompute counters and every pass predicate; preserve individual residuals."""
    noise_failures = force_failures = 0
    for record in payload["records"]:
        noise_failures += not record["shared_noise"]
        force_failures += not record["potential_force_identity"]
        for radius, data in record["radii"].items():
            st = stats[radius]
            walkers = data["walkers"]
            for row, walker in enumerate(walkers):
                qualified = (
                    walker["query1_inside"]
                    and walker["query2_inside"]
                    and record["shared_noise"]
                    and record["potential_force_identity"]
                    and all(side[row] for masks in record["live_masks"].values() for side in masks)
                )
                expected = audit.pathwise_result(
                    walker["input_cost"],
                    walker["observed"],
                    certificates[float(radius)]["rho_upper"],
                    qualified,
                )
                if any(walker[k] != value for k, value in expected.items()):
                    msg = "Cached individual residual or applicability predicate changed"
                    raise ValueError(msg)
                st["walker_candidates"] += 1
                st["query1_inside"] += walker["query1_inside"]
                st["query2_inside"] += walker["query2_inside"]
                if qualified:
                    st["qualified_checks"] += 1
                    st["failed_checks"] += not expected["passed"]
                    old = st["maximum_signed_residual"]
                    residual = expected["signed_residual"]
                    st["maximum_signed_residual"] = residual if old is None else max(old, residual)
                    if walker["input_cost"] > 0:
                        st["maximum_observed_ratio"] = max(
                            st["maximum_observed_ratio"] or 0.0,
                            walker["observed"] / walker["input_cost"],
                        )
            full = all(w["qualified"] for w in walkers)
            expected = audit.pathwise_result(
                math.fsum(w["input_cost"] for w in walkers) / len(walkers),
                math.fsum(w["observed"] for w in walkers) / len(walkers),
                certificates[float(radius)]["rho_upper"],
                full,
            )
            if expected != data["normalized_swarm"]:
                msg = "Cached normalized swarm residual changed"
                raise ValueError(msg)
            st["full_swarm_steps"] += full
            st["full_swarm_failed"] += full and not expected["passed"]
    return noise_failures, force_failures


def complete_dataset(root, output, cache_root, cached, certificates):
    entries, provenance = audit.registered_entries(root)
    if provenance["status"] != "complete":
        msg = "Final local audit requires a complete source index"
        raise ValueError(msg)
    by_path = {e["path"]: e for e in entries}
    cached_artifacts, cached_paths = [], set()
    for entry in cached["derived_chunks"]:
        raw = (cache_root / entry["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            msg = "Cached ledger SHA mismatch"
            raise ValueError(msg)
        payload = json.loads(gzip.decompress(raw))
        for source in payload["source_archives"].values():
            if by_path.get(source["path"]) != source:
                msg = "Final native source differs from cached archive metadata"
                raise ValueError(msg)
            compressed = (root / source["path"]).read_bytes()
            if hashlib.sha256(compressed).hexdigest() != source["sha256"]:
                msg = "Cached native source bytes changed"
                raise ValueError(msg)
            cached_paths.add(source["path"])
        cached_artifacts.append((entry, raw, payload))
    original_register = audit.registered_entries
    missing_entries = [e for e in entries if e["path"] not in cached_paths]
    audit.registered_entries = lambda _: (missing_entries, provenance)
    try:
        result = audit.audit_dataset(root, output, certificates)
    finally:
        audit.registered_entries = original_register
    for entry, raw, payload in cached_artifacts:
        noise, force = add_cached_records(result["radii"], payload, certificates)
        result["shared_noise_failures"] += noise
        result["force_identity_failures"] += force
        result["paired_steps"] += len(payload["records"])
        result["paired_chunks"] += 1
        (output / entry["path"]).write_bytes(raw)
        result["derived_chunks"].append(entry)
    native_count = sum(e["kind"] == "native_run_archive" for e in entries)
    native_steps = sum(
        e["metadata"]["recorded_steps"] for e in entries if e["kind"] == "native_run_archive"
    )
    if result["missing_pairs"] or result["paired_chunks"] * 2 != native_count:
        msg = "Final audit does not cover every completed native archive pair"
        raise ValueError(msg)
    if result["paired_steps"] * 2 != native_steps:
        msg = "Final audit does not cover every completed native step"
        raise ValueError(msg)
    result["cache_reused_chunks"] = len(cached_artifacts)
    result["newly_decoded_chunks"] = result["paired_chunks"] - len(cached_artifacts)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cached_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--skip-library", required=True, type=Path)
    args = parser.parse_args()
    raw = args.cached_report.read_bytes()
    cached = json.loads(raw)
    if cached["helper_sha256"] != hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest():
        msg = "Cached numerical helper does not match the final extension helper"
        raise ValueError(msg)
    args.output.mkdir(parents=True, exist_ok=False)
    audit.load_skipper(args.skip_library)
    certs = {float(k): v for k, v in cached["rates"].items()}
    final = dict(cached)
    final["status"] = "complete_both_final_native_datasets"
    final["cache_report_sha256"] = hashlib.sha256(raw).hexdigest()
    final["extension_helper_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    final["extension_helper_source"] = Path(__file__).read_text(encoding="utf-8")
    final["datasets"] = [
        complete_dataset(
            Path(d["provenance"]["root"]), args.output, args.cached_report.parent, d, certs
        )
        for d in cached["datasets"]
    ]
    (args.output / "report.json").write_text(json.dumps(final, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            [
                {
                    k: d[k]
                    for k in (
                        "paired_steps",
                        "paired_chunks",
                        "cache_reused_chunks",
                        "newly_decoded_chunks",
                        "radii",
                    )
                }
                for d in final["datasets"]
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
