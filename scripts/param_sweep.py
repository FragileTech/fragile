#!/usr/bin/env python3
"""Parameter space exploration for fractal gas coupling diagnostics.

Runs a simulation with a given parameter set, computes coupling diagnostics,
and saves both parameters and results as JSON for later analysis.

Usage:
    python scripts/param_sweep.py
    python scripts/param_sweep.py --N 50 --n-steps 10
    python scripts/param_sweep.py --seed 42 --output-dir outputs/my_sweep
    python scripts/param_sweep.py --param gamma=2.0 --param temperature=1.0
    python scripts/param_sweep.py --batch runs.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from fragile.physics.app.coupling_diagnostics import (
    CouplingDiagnosticsConfig,
    compute_coupling_diagnostics,
)
from fragile.physics.fractal_gas.cloning import CloneOperator
from fragile.physics.fractal_gas.euclidean_gas import EuclideanGas
from fragile.physics.fractal_gas.fitness import FitnessOperator
from fragile.physics.fractal_gas.kinetic_operator import KineticOperator


def get_default_params() -> dict:
    """Return dashboard-matching default parameters.

    Values match ``SimulationTab._apply_default_gas_config()`` in simulation.py.
    """
    return {
        # Simulation
        "N": 500,
        "d": 3,
        "n_steps": 750,
        "record_every": 1,
        # Initialization
        "init_offset": 0.0,
        "init_spread": 0.0,
        "init_velocity_scale": 0.0,
        # Kinetic operator
        "gamma": 1.0,
        "beta": 1.0,
        "auto_thermostat": True,
        "delta_t": 0.01,
        "temperature": 0.5,
        "n_kinetic_steps": 1,
        "integrator": "boris-baoab",
        "nu": 1.0,
        "use_viscous_coupling": True,
        "viscous_length_scale": 1.0,
        "viscous_neighbor_weighting": "riemannian_kernel_volume",
        "beta_curl": 1.0,
        # Cloning
        "p_max": 1.0,
        "epsilon_clone": 1e-6,
        "sigma_x": 0.01,
        "alpha_restitution": 1.0,
        # Fitness
        "fitness_alpha": 1.0,
        "fitness_beta": 1.0,
        "eta": 0.0,
        "sigma_min": 0.0,
        "A": 2.0,
        # Neighbor graph
        "neighbor_graph_update_every": 1,
        "neighbor_weight_modes": [
            "inverse_riemannian_distance",
            "kernel",
            "riemannian_kernel_volume",
        ],
        "clone_every": 1,
        "dtype": "float32",
    }


def run_and_diagnose(
    params: dict,
    seed: int | None = None,
    warmup_fraction: float = 0.1,
    run_name: str | None = None,
) -> dict:
    """Run simulation and compute coupling diagnostics.

    Args:
        params: Full parameter dict (use ``get_default_params()`` as base).
        seed: Optional RNG seed for reproducibility.
        warmup_fraction: Fraction of frames to discard as warmup (0-1).
        run_name: Optional label for this run.

    Returns:
        Dict with keys: run_name, params, seed, duration_seconds, timestamp,
        warmup_fraction, summary, regime_evidence.
    """
    p = params
    t0 = time.monotonic()

    # --- Build operators ---
    kinetic_op = KineticOperator(
        gamma=float(p["gamma"]),
        beta=float(p["beta"]),
        delta_t=float(p["delta_t"]),
        temperature=float(p["temperature"]),
        nu=float(p["nu"]),
        use_viscous_coupling=bool(p["use_viscous_coupling"]),
        viscous_length_scale=float(p["viscous_length_scale"]),
        viscous_neighbor_weighting=str(p["viscous_neighbor_weighting"]),
        beta_curl=float(p["beta_curl"]),
    )
    # n_kinetic_steps is a param attribute, not a constructor arg
    kinetic_op.n_kinetic_steps = int(p.get("n_kinetic_steps", 1))
    if bool(p.get("auto_thermostat", False)):
        kinetic_op.auto_thermostat = True

    cloning = CloneOperator(
        p_max=float(p["p_max"]),
        epsilon_clone=float(p["epsilon_clone"]),
        sigma_x=float(p["sigma_x"]),
        alpha_restitution=float(p["alpha_restitution"]),
    )

    fitness_op = FitnessOperator(
        alpha=float(p["fitness_alpha"]),
        beta=float(p["fitness_beta"]),
        eta=float(p["eta"]),
        sigma_min=float(p["sigma_min"]),
        A=float(p["A"]),
    )

    # --- Build gas ---
    N = int(p["N"])
    d = int(p["d"])
    gas = EuclideanGas(
        N=N,
        d=d,
        kinetic_op=kinetic_op,
        cloning=cloning,
        fitness_op=fitness_op,
        device=torch.device("cpu"),
        dtype=str(p["dtype"]),
        clone_every=int(p["clone_every"]),
        neighbor_graph_update_every=int(p["neighbor_graph_update_every"]),
        neighbor_weight_modes=list(p["neighbor_weight_modes"]),
    )

    # --- Initialize state ---
    x_init = torch.randn(N, d) * float(p["init_spread"]) + float(p["init_offset"])
    v_init = torch.randn(N, d) * float(p["init_velocity_scale"])

    # --- Run simulation ---
    history = gas.run(
        int(p["n_steps"]),
        x_init=x_init,
        v_init=v_init,
        record_every=int(p["record_every"]),
        seed=seed,
        show_progress=True,
    )

    # --- Compute diagnostics ---
    config = CouplingDiagnosticsConfig(warmup_fraction=warmup_fraction)
    output = compute_coupling_diagnostics(history, config=config)

    duration = time.monotonic() - t0

    # Sanitize NaN/Inf for JSON serialization
    summary = {k: _json_safe(v) for k, v in output.summary.items()}

    return {
        "run_name": run_name,
        "params": params,
        "seed": seed,
        "warmup_fraction": warmup_fraction,
        "duration_seconds": round(duration, 2),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "summary": summary,
        "regime_evidence": list(output.regime_evidence),
    }


def _json_safe(v):
    """Convert NaN/Inf floats to None for JSON compatibility."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def save_result(result: dict, output_dir: Path) -> Path:
    """Save result dict as JSON with timestamp-based filename."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_hash = hashlib.md5(json.dumps(result["params"], sort_keys=True).encode()).hexdigest()[:8]
    name_part = f"_{result['run_name']}" if result.get("run_name") else ""
    filename = f"{ts}{name_part}_{short_hash}.json"
    path = output_dir / filename
    path.write_text(json.dumps(result, indent=2))
    return path


def _parse_value(raw: str):
    """Parse a CLI value string into int, float, bool, or str."""
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def run_batch(batch_file: Path, seed: int, warmup_fraction: float, output_dir: Path):
    """Run a batch of experiments from a JSON config file.

    The JSON file should be a list of objects with "name" and "overrides" keys:
    [{"name": "gamma_0.1", "overrides": {"gamma": 0.1}}, ...]
    """
    runs = json.loads(batch_file.read_text())
    for i, run_spec in enumerate(runs):
        name = run_spec["name"]
        overrides = run_spec.get("overrides", {})
        params = get_default_params()
        params.update(overrides)
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(runs)}] {name}")
        print(f"  Overrides: {overrides}")
        print(f"{'='*60}")
        run_seed = run_spec.get("seed", seed)
        result = run_and_diagnose(params, seed=run_seed, warmup_fraction=warmup_fraction, run_name=name)
        path = save_result(result, output_dir)
        score = result["summary"].get("regime_score")
        print(f"  -> regime_score={score}, saved to {path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run fractal gas simulation and compute coupling diagnostics.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/param_sweep"),
        help="Directory for result JSON files (default: outputs/param_sweep)",
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.1,
        help="Fraction of frames to discard as warmup (default: 0.1)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional label for this run",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a default parameter, e.g. --param gamma=2.0",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        default=None,
        help="Path to JSON batch config file",
    )
    # Convenience shortcuts for the most common parameters
    parser.add_argument("--N", type=int, default=None, help="Number of walkers")
    parser.add_argument("--n-steps", type=int, default=None, help="Number of simulation steps")
    parser.add_argument("--d", type=int, default=None, help="Spatial dimension")

    args = parser.parse_args()

    if args.batch is not None:
        run_batch(args.batch, seed=args.seed, warmup_fraction=args.warmup_fraction,
                  output_dir=args.output_dir)
        return

    params = get_default_params()

    # Apply --param overrides
    for kv in args.param:
        if "=" not in kv:
            parser.error(f"--param must be KEY=VALUE, got: {kv!r}")
        key, raw_val = kv.split("=", 1)
        if key not in params:
            parser.error(f"Unknown parameter: {key!r}. Valid: {sorted(params)}")
        params[key] = _parse_value(raw_val)

    # Apply convenience shortcuts
    if args.N is not None:
        params["N"] = args.N
    if args.n_steps is not None:
        params["n_steps"] = args.n_steps
    if args.d is not None:
        params["d"] = args.d

    print(f"Running with params: N={params['N']}, d={params['d']}, n_steps={params['n_steps']}")
    result = run_and_diagnose(
        params, seed=args.seed, warmup_fraction=args.warmup_fraction, run_name=args.run_name,
    )

    path = save_result(result, args.output_dir)
    print(f"\nDone in {result['duration_seconds']:.1f}s")
    print(f"Regime score: {result['summary'].get('regime_score')}")
    print(f"Results saved to: {path}")


if __name__ == "__main__":
    main()
