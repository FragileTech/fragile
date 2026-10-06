"""Verify absolute native harmonic noise transport from retained first-step CBOR."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

from read_native_cbor import first_native_step
from reanalyze_chapter05_stratified_uncertainty import stratified_estimate


def stage(step, name, field):
    return next(s["fields"][field]["values"] for s in step["stages"] if s["stage"] == name)


def evaluated(step, name, field):
    return next(
        s["values"]
        for s in step["field_evaluations"]
        if s["stage"] == name and s["field"] == field
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.archive_root.resolve()
    if args.output.resolve().is_relative_to(root):
        msg = "Write the derived noise review outside the immutable native root"
        raise ValueError(msg)
    index = json.loads((root / "archive-index.json").read_text())
    plans = {}
    for entry in index["entries"]:
        if entry["kind"] == "raw_json" and entry["tag"].endswith("/coupling_plan"):
            data = (root / entry["path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                msg = "Plan checksum mismatch"
                raise ValueError(msg)
            plans[entry["tag"].removesuffix("/coupling_plan")] = json.loads(gzip.decompress(data))
    cases = {
        key: {
            "hypotheses": plan["hypotheses"],
            "moments": {},
            "artifacts": [],
            "maximum_absolute_map_residual": 0.0,
            "maximum_force_residual": 0.0,
        }
        for key, plan in plans.items()
    }
    for entry in index["entries"]:
        match = re.fullmatch(r"(.+)/rep(\d+)/left_through\d+", entry["tag"])
        if (
            not match
            or entry["kind"] != "native_run_archive"
            or entry["metadata"]["first_step"] != 1
        ):
            continue
        key, replicate = match[1], int(match[2])
        if key not in cases:
            continue
        case = cases[key]
        data = (root / entry["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Native compressed hash mismatch: {entry['path']}")
        archive = first_native_step(gzip.decompress(data), verify_remainder=False)
        step = archive["steps"][0]
        config = archive["gas_config"]["kinetic"]
        innovation = config["noise"]
        if (
            innovation["innovation"] != "gaussian"
            or innovation["geometry"]["kind"] != "isotropic"
            or innovation["geometry"]["scale"]["kind"] != "constant"
        ):
            msg = "This harmonic constant-noise reviewer cannot transfer to adaptive/non-Gaussian noise"
            raise ValueError(msg)
        integrator = config["integrator"]
        if integrator["kind"] != "baoab":
            msg = "The recorded integrator is not BAOAB"
            raise ValueError(msg)
        h, gamma = integrator["dt"], integrator["friction"]
        diffusion = innovation["geometry"]["scale"]["values"][0]
        omega = case["hypotheses"].get("curvature", 1.0)
        c, a = h / 2, math.exp(-gamma * h)
        k = 1 - omega * c * c
        scale = math.sqrt(h if gamma == 0 else -math.expm1(-2 * gamma * h) / (2 * gamma))
        sigma_squared = (k * scale * diffusion) ** 2
        xx, vv = stage(step, "B1_input", "positions"), stage(step, "B1_input", "velocities")
        xout, vout = stage(step, "A2", "positions"), stage(step, "B2", "velocities")
        eta = evaluated(step, "O", "executed_noise")
        ax, b = 1 - omega * c * c * (1 + a), c * (1 + a)
        av = a - omega * c * c * (1 + a)
        h21 = -omega * b * k
        errors = [vo - h21 * x - av * v for x, v, vo in zip(xx, vv, vout, strict=True)]
        residuals = [
            abs(error - k * scale * noise) for error, noise in zip(errors, eta, strict=True)
        ]
        residuals += [
            abs(xo - ax * x - b * v - c * scale * noise)
            for x, v, xo, noise in zip(xx, vv, xout, eta, strict=True)
        ]
        case["maximum_absolute_map_residual"] = max(
            case["maximum_absolute_map_residual"], *residuals
        )
        for input_name, kick in [("B1_input", "B1"), ("B2_input", "B2")]:
            force_residual = max(
                abs(gradient - omega * x)
                for x, gradient in zip(
                    stage(step, input_name, "positions"),
                    evaluated(step, kick, "potential_gradient"),
                    strict=True,
                )
            )
            case["maximum_force_residual"] = max(case["maximum_force_residual"], force_residual)
        case["moments"][replicate] = sum(e * e for e in errors) / len(errors)
        case["expected_sigma_squared"] = sigma_squared
        case["artifacts"].append({"path": entry["path"], "sha256": entry["sha256"]})
    source_path = (
        Path(__file__).resolve().parents[1]
        / "docs/source/2_fractal_gas/convergence_program/05_kinetic_contraction.md"
    )
    source_bytes = source_path.read_bytes()
    source = source_bytes.decode()
    begin = source.index(
        "H=\\begin{pmatrix}", source.index(":label: thm-kinetic-dimension-curvature-cap")
    )
    start = source.rfind("$$", 0, begin)
    end = source.index("$$", begin) + 2
    result = []
    for key, case in cases.items():
        keys = sorted(case["moments"])
        if keys != list(range(len(keys))):
            msg = "First-step replicate indices are incomplete"
            raise ValueError(msg)
        expected_samples = case["hypotheses"].get(
            "samples", case["hypotheses"].get("independent_replicates")
        )
        if len(keys) != expected_samples:
            msg = "Missing first-step left-side replicas; common-noise right sides are not extra samples"
            raise ValueError(msg)
        mean, se = stratified_estimate([case["moments"][i] for i in keys])
        expected = case["expected_sigma_squared"]
        variance_passed = abs(mean - expected) <= 6 * se + 2e-10 * (1 + mean + expected)
        map_passed = (
            case["maximum_absolute_map_residual"] <= 2e-12
            and case["maximum_force_residual"] <= 2e-12
        )
        case["case"] = key
        case["conditional_noise_variance"] = {
            "observed": mean,
            "bound": expected,
            "standard_error": se,
            "relation": "equal",
            "passed": variance_passed,
            "scope": "Per-coordinate variance of native pre-cap innovation, centered with entering actual states; complete independent seed replicas supply uncertainty",
        }
        case["absolute_native_map"] = {
            "observed": case["maximum_absolute_map_residual"],
            "bound": 0,
            "numerical_tolerance": 2e-12,
            "relation": "equal",
            "passed": map_passed,
            "source_labels": ["thm-kinetic-dimension-curvature-cap"],
            "source_quotes": [source[start:end]],
            "scope": "Complete position/velocity pre-cap map and sigma=kq innovation identity, both actual force evaluations checked",
        }
        del case["moments"]
        result.append(case)
    report = {
        "chapter": 5,
        "source_path": str(source_path),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "archive_root": str(root),
        "native_steps": 0,
        "decoding_scope": "Compressed indexed SHA256 verified for each artifact; configuration and first native step decoded. This selected-prefix reanalysis does not replace complete archive deep validation.",
        "cases": result,
        "summary": {
            "first_step_native_archives_verified": sum(len(c["artifacts"]) for c in result),
            "comparisons": 2 * len(result),
            "comparisons_failed": sum(
                not c[key]["passed"]
                for c in result
                for key in ["conditional_noise_variance", "absolute_native_map"]
            ),
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
