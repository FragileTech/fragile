"""Derive exact extinction times and kinetic reference constants from saved runs."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("empty_output", type=Path)
    args = parser.parse_args()
    index_path = args.dataset / "archive-index.json"
    index = json.loads(index_path.read_text())
    if index["status"] != "complete":
        message = "A complete native dataset is required"
        raise ValueError(message)
    source_report = args.dataset / "report.json"
    report = json.loads(source_report.read_text())
    entries = {e["path"]: e for e in index["entries"]}
    seeds = set()
    rows = []
    extinctions = []
    maximum_steps = report["summary"]["requested_steps"]
    for case in report["cases"]:
        gas = case["config"]["gas"]
        gamma = gas["kinetic"]["integrator"]["friction"]
        factor = gas["kinetic"]["noise"]["geometry"]["scale"]["values"][0]
        diffusion = factor**2 / 2
        theta = diffusion / gamma
        curvature = 2.0
        oscillation = 20.0 if case["profile"].startswith("rastrigin") else 0.0
        hessian = 2.0 + (40 * math.pi**2 if oscillation else 0.0)
        log_cx = math.log(theta / curvature) + oscillation / theta
        log_lsi = max(math.log(theta), log_cx)
        eta = diffusion / (2 * (1 + 2 * hessian + (2 * hessian + gamma + 2) ** 2))
        log_rate = math.log(eta) - math.log(math.exp(log_lsi) / 2 + 3 * eta)
        died = 0
        updates = 0
        for trajectory in case["trajectories"]:
            seed = trajectory["seed"]
            if seed in seeds:
                message = "Native independent ensemble reused a seed"
                raise ValueError(message)
            seeds.add(seed)
            entry = entries[trajectory["archive"]]
            count = trajectory["recorded_steps"]
            if entry["metadata"]["recorded_steps"] != count:
                message = "Run summary differs from complete archive manifest"
                raise ValueError(message)
            updates += count
            mass = trajectory["terminal_observables"][3]
            first_killing_step = count if mass == 0 else None
            if count < maximum_steps and mass != 0:
                message = "A nonextinct run was censored before the designed horizon"
                raise ValueError(message)
            if trajectory["initial_observables"][3] <= 0:
                message = "Prepared initial native mass must be positive"
                raise ValueError(message)
            if first_killing_step is not None:
                died += 1
                extinctions.append({
                    "case": case["id"],
                    "tag": trajectory["tag"],
                    "seed": seed,
                    "archive": trajectory["archive"],
                    "first_killing_step": first_killing_step,
                    "physical_killing_time": first_killing_step * case["h"],
                    "detected_before_next_attempt": trajectory["extinction_before_step"],
                })
        rows.append({
            "case": case["id"],
            "profile": case["profile"],
            "N": case["N"],
            "d": case["d"],
            "trajectories": len(case["trajectories"]),
            "complete_native_updates": updates,
            "first_transition_extinctions": died,
            "velocity_noise_factor": factor,
            "velocity_diffusion_D": diffusion,
            "friction": gamma,
            "theta": theta,
            "curvature": curvature,
            "coordinate_perturbation_oscillation": oscillation,
            "global_Hessian_norm_upper": hessian,
            "log_spatial_LSI_upper": log_cx,
            "log_kinetic_LSI_upper": log_lsi,
            "eta": eta,
            "log_continuous_reference_rate": log_rate,
            "continuous_reference_rate": math.exp(log_rate),
            "reference_scope": "Conservative separable kinetic Gibbs law with the native velocity-noise coefficient and force potential. Native cap, selected cloning, position diffusion and killing change the full law; this reference rate is not assigned to those trajectories.",
        })
    actual_updates = sum(r["complete_native_updates"] for r in rows)
    if actual_updates != report["summary"]["new_complete_native_updates"]:
        message = "Full retained update count differs from native execution report"
        raise ValueError(message)
    args.empty_output.mkdir(parents=True, exist_ok=False)
    result = {
        "source_index_sha256": sha(index_path),
        "source_report_sha256": sha(source_report),
        "helper_sha256": sha(Path(__file__)),
        "cases": rows,
        "extinctions": extinctions,
        "summary": {
            "cases": len(rows),
            "independent_trajectories": len(seeds),
            "complete_native_updates": actual_updates,
            "correct_first_transition_extinctions": len(extinctions),
            "terminal_horizon_extinctions": sum(
                r["first_killing_step"] == maximum_steps for r in extinctions
            ),
        },
        "event_derivation": "All initial masses are positive. The unchanged native engine refuses its next update after a zero alive population. Each archived trajectory therefore first dies on its last recorded transition iff terminal alive mass is zero. This includes the terminal requested horizon.",
    }
    (args.empty_output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.empty_output / "helper.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
