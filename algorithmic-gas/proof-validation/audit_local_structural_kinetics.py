"""Check local native nonlinear kinetic contraction on retained paired archives.

Membership is recorded before filtering. These are pathwise coupled kinetic
checks; conditioning on residence does not certify an expectation or mixing rate.
"""

import argparse
from collections import defaultdict
import ctypes
from decimal import Decimal
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

from read_native_cbor import FirstStepReader


SKIPPER = {}


def load_skipper(path):
    library = ctypes.CDLL(str(path.resolve()))
    function = library.skip_native_cbor
    function.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    SKIPPER.update({"library": library, "function": function})


class AllStepsReader(FirstStepReader):
    """Parse complete definite CBOR, discarding irrelevant large graph fields."""

    def first_array(self, depth):
        return self.value(depth)

    def value(self, depth=0, keep=True):
        if not keep and SKIPPER:
            position = ctypes.c_size_t(self.position)
            pointer = ctypes.cast(ctypes.c_char_p(self.data.obj), ctypes.POINTER(ctypes.c_uint8))
            if SKIPPER["function"](pointer, len(self.data), ctypes.byref(position), depth) != 1:
                msg = "Malformed/unsupported CBOR discarded subtree"
                raise ValueError(msg)
            self.position = position.value
            return None
        if keep and self.position < len(self.data) and self.data[self.position] >> 5 == 5:
            if depth > 64:
                msg = "CBOR nesting exceeds native archive limit"
                raise ValueError(msg)
            _, _, payload = self.header()
            count = payload if isinstance(payload, int) else int.from_bytes(payload, "big")
            result = {}
            for _ in range(count):
                key = self.value(depth + 1)
                retain = key not in {
                    "influences",
                    "graph",
                    "before",
                    "donor_fitness",
                    "noise",
                    "elite_injection",
                    "rewards",
                    "observations",
                    "final_population",
                    "report",
                    "generations",
                }
                if key in {"fields", "validity"} and "stage" in result:
                    retain = result["stage"] in {"B1_input", "B2_input", "terminal"}
                if key == "values" and "field" in result:
                    retain = result["field"] in {"executed_noise", "potential_force"}
                item = self.value(depth + 1, keep=retain)
                if retain:
                    result[key] = item
            return result
        return super().value(depth, keep)


def checked_payload(root, entry):
    raw = (root / entry["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        msg = f"Artifact SHA mismatch: {entry['path']}"
        raise ValueError(msg)
    if len(raw) != entry["compressed_bytes"]:
        msg = f"Artifact byte-count mismatch: {entry['path']}"
        raise ValueError(msg)
    decoded = gzip.decompress(raw)
    if len(decoded) > 256 * 1024 * 1024:
        msg = "Decoded archive exceeds native read budget"
        raise ValueError(msg)
    return decoded


def read_archive(root, entry):
    payload = checked_payload(root, entry)
    reader = AllStepsReader(payload)
    archive = reader.value()
    if reader.position != len(payload):
        msg = "Trailing CBOR data"
        raise ValueError(msg)
    if len(archive["steps"]) != entry["metadata"]["recorded_steps"]:
        msg = "Native recorded-step count mismatch"
        raise ValueError(msg)
    return archive


def registered_entries(root):
    index_bytes = (root / "archive-index.json").read_bytes()
    index = json.loads(index_bytes)
    journal_path = root / "archive-journal.jsonl"
    journal_bytes = journal_path.read_bytes() if journal_path.exists() else b""
    # A writer can be appending a record. Only complete newline-terminated
    # records are committed; previously snapshotted entries must agree.
    journal = [json.loads(line) for line in journal_bytes.split(b"\n")[:-1]]
    entries = index["entries"]
    if journal:
        if journal[: len(entries)] != entries:
            msg = "Index and durable journal disagree"
            raise ValueError(msg)
        entries = journal
    return entries, {
        "root": str(root),
        "status": index["status"],
        "snapshot_entries": len(index["entries"]),
        "committed_entries": len(entries),
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "journal_snapshot_sha256": hashlib.sha256(journal_bytes).hexdigest(),
    }


def field(step, name, key):
    value = next(s["fields"][key]["values"] for s in step["stages"] if s["stage"] == name)
    if not all(math.isfinite(x) for x in value):
        msg = f"Nonfinite stage: {name}/{key}"
        raise ValueError(msg)
    return value


def evaluated(step, stage, key):
    return next(
        f["values"] for f in step["field_evaluations"] if f["stage"] == stage and f["field"] == key
    )


def live_mask(step, stage, n):
    validity = next(s["validity"] for s in step["stages"] if s["stage"] == stage)
    if len(validity) != n:
        msg = "Native live-mask dimension disagrees with tag"
        raise ValueError(msg)
    return [not any(row.values()) for row in validity]


def same_well_membership(query1, query2, radius):
    left1, right1 = query1
    left2, right2 = query2
    center = [round(x) for x in left1]
    first = all(abs(x - k) <= radius for v in (left1, right1) for x, k in zip(v, center))
    second = all(abs(x - k) <= radius for v in (left2, right2) for x, k in zip(v, center))
    return center, first, second


def metric(dx, dv, omega, alpha, beta):
    return math.fsum(
        alpha * omega * x * x + 2 * beta * math.sqrt(omega) * x * v + v * v for x, v in zip(dx, dv)
    )


def pathwise_result(cost_in, cost_out, rho, qualified):
    bound = rho * cost_in
    tolerance = 1e-11 * max(1.0, abs(cost_out), abs(bound))
    return {
        "qualified": qualified,
        "observed": cost_out,
        "bound": bound,
        "signed_residual": cost_out - bound,
        "tolerance": tolerance,
        "passed": cost_out <= bound + tolerance if qualified else None,
    }


def interval_endpoint(value, upper):
    parts = value.strip("[]").split(",")
    return Decimal(parts[int(upper)].strip())


def validate_config(config, providers):
    kinetic = config["kinetic"]
    integrator = kinetic["integrator"]
    qft = config["qft"]
    ok = (
        config["boundary"]["kind"] == "unbounded"
        and kinetic["boundary_schedule"] == "end_of_step"
        and integrator["kind"] == "baoab"
        # Exact parameter matching is deliberate for the retained certificate.
        and (integrator["dt"], integrator["friction"], kinetic["velocity_cap"]) == (0.04, 1.0, 2.0)
        and kinetic["noise"]
        == {
            "geometry": {"kind": "isotropic", "scale": {"kind": "constant", "values": [1.0]}},
            "innovation": "gaussian",
        }
        and not any(qft[k] for k in ("curl", "graph_viscosity", "viscosity", "innovation_shifts"))
        and providers["gradient"] == "analytic-gradient/Rastrigin/Minimize/v1"
    )
    if not ok:
        msg = "Native configuration does not meet interval-certificate hypotheses"
        raise ValueError(msg)


def audit_dataset(root, output, certificates):
    entries, provenance = registered_entries(root)
    source_entry = next(e for e in entries if e["tag"] == "structural-source")
    source = json.loads(checked_payload(root, source_entry))
    provenance["embedded_source_sha256"] = {
        k: hashlib.sha256(source[k].encode()).hexdigest()
        for k in ("module", "runner", "chapter6a", "algorithm_source")
    }
    provenance["source_artifact"] = source_entry
    pairs = defaultdict(dict)
    pattern = re.compile(r"(rastrigin-d(\d+)-n(\d+)-([\w]+)-rep\d+)-(left|right)-(\d+)$")
    for entry in entries:
        if entry["kind"] != "native_run_archive":
            continue
        match = pattern.fullmatch(entry["tag"])
        if not match:
            msg = f"Unknown native stream tag {entry['tag']}"
            raise ValueError(msg)
        prefix, d, n, zone, side, last = match.groups()
        pairs[prefix, int(last), int(d), int(n), zone][side] = entry
    stats = {
        str(r): {
            "walker_candidates": 0,
            "query1_inside": 0,
            "query2_inside": 0,
            "qualified_checks": 0,
            "failed_checks": 0,
            "full_swarm_steps": 0,
            "full_swarm_failed": 0,
            "maximum_signed_residual": None,
            "maximum_observed_ratio": None,
        }
        for r in certificates
    }
    paths, steps, noise_failures, potential_failures, missing = [], 0, 0, 0, []
    for key, pair in sorted(pairs.items()):
        prefix, last, d, n, zone = key
        if set(pair) != {"left", "right"}:
            missing.append({"tag": prefix, "last_step": last, "available_sides": list(pair)})
            continue
        for entry in pair.values():
            validate_config(entry["metadata"]["native_config"], entry["metadata"]["providers"])
        left, right = [read_archive(root, pair[s]) for s in ("left", "right")]
        if len(left["steps"]) != len(right["steps"]):
            msg = "Paired native chunks have unequal lengths"
            raise ValueError(msg)
        records = []
        for offset, (l, r) in enumerate(zip(left["steps"], right["steps"])):
            steps += 1
            first = pair["left"]["metadata"]["first_step"]
            if first != pair["right"]["metadata"]["first_step"]:
                msg = "Paired chunk step addresses disagree"
                raise ValueError(msg)
            shared_noise = all(
                evaluated(l, s, "executed_noise") == evaluated(r, s, "executed_noise")
                for s in ("O", "position_diffusion")
            )
            noise_failures += not shared_noise
            potential_ok = all(
                abs(force + 2 * x + 20 * math.pi * math.sin(2 * math.pi * x))
                <= 1e-10 * max(1.0, abs(force))
                for step in (l, r)
                for s in ("B1", "B2")
                for x, force in zip(
                    field(step, s + "_input", "positions"), evaluated(step, s, "potential_force")
                )
            )
            potential_failures += not potential_ok
            lx, rx = [field(s, "B1_input", "positions") for s in (l, r)]
            lv, rv = [field(s, "B1_input", "velocities") for s in (l, r)]
            l2, r2 = [field(s, "B2_input", "positions") for s in (l, r)]
            ox, oy = [field(s, "terminal", "positions") for s in (l, r)]
            ov, ow = [field(s, "terminal", "velocities") for s in (l, r)]
            if any(len(v) != n * d for v in (lx, rx, lv, rv, l2, r2, ox, oy, ov, ow)):
                msg = "Native field dimensions disagree with tag"
                raise ValueError(msg)
            record = {
                "step": first + offset,
                "shared_noise": shared_noise,
                "potential_force_identity": potential_ok,
                "radii": {},
            }
            masks = {
                name: [live_mask(step, name, n) for step in (l, r)]
                for name in ("B1_input", "B2_input", "terminal")
            }
            record["live_masks"] = masks
            for radius, cert in certificates.items():
                st = stats[str(radius)]
                walkers = []
                for i in range(n):
                    sl = slice(i * d, (i + 1) * d)
                    center, q1, q2 = same_well_membership(
                        (lx[sl], rx[sl]), (l2[sl], r2[sl]), radius
                    )
                    cost_in = metric(
                        [a - b for a, b in zip(lx[sl], rx[sl])],
                        [a - b for a, b in zip(lv[sl], rv[sl])],
                        cert["omega"],
                        cert["alpha"],
                        0.025,
                    )
                    cost_out = metric(
                        [a - b for a, b in zip(ox[sl], oy[sl])],
                        [a - b for a, b in zip(ov[sl], ow[sl])],
                        cert["omega"],
                        cert["alpha"],
                        0.025,
                    )
                    check = pathwise_result(
                        cost_in,
                        cost_out,
                        cert["rho_upper"],
                        q1
                        and q2
                        and shared_noise
                        and potential_ok
                        and all(side[i] for mask in masks.values() for side in mask),
                    )
                    check.update({
                        "coupling_row": i,
                        "well_center": center,
                        "query1_inside": q1,
                        "query2_inside": q2,
                        "input_cost": cost_in,
                    })
                    walkers.append(check)
                    st["walker_candidates"] += 1
                    st["query1_inside"] += q1
                    st["query2_inside"] += q2
                    if check["qualified"]:
                        st["qualified_checks"] += 1
                        st["failed_checks"] += not check["passed"]
                        residual = check["signed_residual"]
                        previous = st["maximum_signed_residual"]
                        st["maximum_signed_residual"] = (
                            residual if previous is None else max(previous, residual)
                        )
                        if cost_in > 0:
                            st["maximum_observed_ratio"] = max(
                                st["maximum_observed_ratio"] or 0.0, cost_out / cost_in
                            )
                full = all(w["qualified"] for w in walkers)
                swarm = pathwise_result(
                    math.fsum(w["input_cost"] for w in walkers) / n,
                    math.fsum(w["observed"] for w in walkers) / n,
                    cert["rho_upper"],
                    full,
                )
                st["full_swarm_steps"] += full
                st["full_swarm_failed"] += full and not swarm["passed"]
                record["radii"][str(radius)] = {"walkers": walkers, "normalized_swarm": swarm}
            records.append(record)
        name = f"{root.name}-{prefix}-{last}.json.gz"
        payload = json.dumps(
            {
                "source_archives": pair,
                "zone": zone,
                "dimension": d,
                "population": n,
                "records": records,
            },
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        compressed = gzip.compress(payload, mtime=0)
        (output / name).write_bytes(compressed)
        paths.append({
            "path": name,
            "sha256": hashlib.sha256(compressed).hexdigest(),
            "paired_steps": len(records),
        })
    return {
        "provenance": provenance,
        "paired_chunks": len(paths),
        "paired_steps": steps,
        "missing_pairs": missing,
        "shared_noise_failures": noise_failures,
        "force_identity_failures": potential_failures,
        "radii": stats,
        "derived_chunks": paths,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--skip-library", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.certificate.read_bytes()
    certificate = json.loads(raw)
    if args.skip_library:
        load_skipper(args.skip_library)
    certificates = {}
    for row in certificate["rows"]:
        radius = row["well_half_width"]
        if radius not in {0.01, 0.03}:
            continue
        if not row["harmonic_certificate_pass"] or not row["contracts"]:
            msg = "Requested local rate has no independent interval certificate"
            raise ValueError(msg)
        omega = float(interval_endpoint(row["harmonic_curvature_interval"], True))
        certificates[radius] = {
            "omega": omega,
            "alpha": 1 - omega * 0.04**2 / 4,
            "rho_upper": math.nextafter(
                float(interval_endpoint(row["rho_interval"], True)), math.inf
            ),
            "interval_row": row,
        }
    report = {
        "status": "complete_zero_step_reanalysis",
        "new_native_steps": 0,
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_source": Path(__file__).read_text(encoding="utf-8"),
        "decoder_sha256": hashlib.sha256(
            Path(__file__).with_name("read_native_cbor.py").read_bytes()
        ).hexdigest(),
        "discarded_subtree_skipper": {
            "source_sha256": hashlib.sha256(
                Path(__file__).with_name("skip_native_cbor.c").read_bytes()
            ).hexdigest(),
            "source": Path(__file__).with_name("skip_native_cbor.c").read_text(encoding="utf-8"),
            "library_sha256": hashlib.sha256(args.skip_library.read_bytes()).hexdigest()
            if args.skip_library
            else None,
            "scope": "Optional bounded definite-CBOR structural traversal only; measured stage values are decoded by the retained Python decoder. No native simulation or numerical-rate computation in the skipper.",
        },
        "certificate_path": str(args.certificate),
        "certificate_sha256": hashlib.sha256(raw).hexdigest(),
        "source_label": certificate["source_label"],
        "source_sha256": certificate["source_sha256"],
        "rates": certificates,
        "scope": "Pathwise G contraction on each prepared B1-input coupling row whose two trajectories have both actual B1 and B2 queries in the same integer well box, with actual shared OU/position noise. Coupling rows are a witness; no persistent walker labels, empirical-law optimality, post-residence conditional expectation, full selected-update contraction, or global/QSD mixing rate is inferred. Swarm costs are normalized by N.",
        "arithmetic_scope": "Independent real interval certificate plus binary64 metric evaluations with explicit 1e-11 relative/absolute residual tolerance. Every qualified individual residual checked, no SEM allowance.",
        "datasets": [audit_dataset(root, args.output, certificates) for root in args.datasets],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "datasets": [
                    {
                        "root": d["provenance"]["root"],
                        "source_status": d["provenance"]["status"],
                        "paired_steps": d["paired_steps"],
                        "radii": d["radii"],
                    }
                    for d in report["datasets"]
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
