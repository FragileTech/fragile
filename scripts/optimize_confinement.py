#!/usr/bin/env python3
"""Optimize parameters for string tension, screening length, and spectral gap."""

from __future__ import annotations

import math
import time

import torch

from fragile.physics.app.coupling_diagnostics import (
    compute_coupling_diagnostics,
    CouplingDiagnosticsConfig,
)
from fragile.physics.fractal_gas.cloning import CloneOperator
from fragile.physics.fractal_gas.euclidean_gas import EuclideanGas
from fragile.physics.fractal_gas.fitness import FitnessOperator
from fragile.physics.fractal_gas.kinetic_operator import KineticOperator


def run_and_diagnose(params: dict, seed: int = 42) -> dict:
    p = params
    t0 = time.monotonic()

    kinetic_op = KineticOperator(
        gamma=float(p["gamma"]),
        beta=float(p["beta"]),
        delta_t=float(p["delta_t"]),
        temperature=float(p["temperature"]),
        nu=float(p["nu"]),
        use_viscous_coupling=True,
        viscous_length_scale=float(p["viscous_length_scale"]),
        viscous_neighbor_weighting=str(p["viscous_neighbor_weighting"]),
        beta_curl=float(p["beta_curl"]),
    )
    if p.get("auto_thermostat", True):
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

    gas = EuclideanGas(
        N=int(p["N"]),
        d=int(p["d"]),
        kinetic_op=kinetic_op,
        cloning=cloning,
        fitness_op=fitness_op,
        device=torch.device("cpu"),
        dtype="float32",
        clone_every=int(p["clone_every"]),
        neighbor_graph_update_every=int(p["neighbor_graph_update_every"]),
        neighbor_weight_modes=list(p["neighbor_weight_modes"]),
        tessellation_timing="after_cloning",
    )

    N, d = int(p["N"]), int(p["d"])
    x_init = torch.randn(N, d) * float(p["init_spread"]) + float(p["init_offset"])
    v_init = torch.randn(N, d) * float(p["init_velocity_scale"])

    history = gas.run(
        int(p["n_steps"]),
        x_init=x_init,
        v_init=v_init,
        record_every=int(p["record_every"]),
        seed=seed,
        show_progress=False,
    )

    config = CouplingDiagnosticsConfig(warmup_fraction=0.15)
    output = compute_coupling_diagnostics(history, config=config)
    duration = time.monotonic() - t0

    summary = {}
    for k, v in output.summary.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            summary[k] = None
        else:
            summary[k] = v
    return {"summary": summary, "duration": round(duration, 1)}


def defaults(n_steps=1000, N=500):
    return {
        "N": N,
        "d": 3,
        "n_steps": n_steps,
        "record_every": 1,
        "init_offset": 0.0,
        "init_spread": 0.0,
        "init_velocity_scale": 0.0,
        "gamma": 1.0,
        "beta": 1.0,
        "auto_thermostat": True,
        "delta_t": 0.01,
        "temperature": 0.5,
        "nu": 1.0,
        "use_viscous_coupling": True,
        "viscous_length_scale": 1.0,
        "viscous_neighbor_weighting": "riemannian_kernel_volume",
        "beta_curl": 1.0,
        "p_max": 1.0,
        "epsilon_clone": 1e-6,
        "sigma_x": 0.01,
        "alpha_restitution": 1.0,
        "fitness_alpha": 1.0,
        "fitness_beta": 1.0,
        "eta": 0.0,
        "sigma_min": 0.0,
        "A": 2.0,
        "neighbor_graph_update_every": 1,
        "neighbor_weight_modes": [
            "inverse_riemannian_distance",
            "kernel",
            "riemannian_kernel_volume",
        ],
        "clone_every": 1,
        "dtype": "float32",
    }


def fmt(v, w=8):
    if v is None:
        return " " * (w - 4) + "None"
    return f"{v:{w}.4f}"


def report(name: str, result: dict):
    s = result["summary"]
    sigma = s.get("string_tension_sigma")
    xi = s.get("screening_length_xi")
    asym = s.get("re_im_asymmetry_mean")
    score = s.get("regime_score")
    poly = s.get("polyakov_abs")
    sg_fiedler = s.get("spectral_gap_fiedler")
    sg_ac = s.get("spectral_gap_autocorrelation")
    sg_tm = s.get("spectral_gap_transfer_matrix")
    dur = result["duration"]

    print(
        f"  {name:40s} | σ={fmt(sigma)} | ξ={fmt(xi)} | asym={fmt(asym)} | "
        f"poly={fmt(poly)} | fiedler={fmt(sg_fiedler)} | ac_gap={fmt(sg_ac)} | "
        f"tm_gap={fmt(sg_tm)} | score={fmt(score)} | {dur:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    header = (
        f"  {'name':40s} | {'σ':>8s} | {'ξ':>8s} | {'asym':>8s} | "
        f"{'poly':>8s} | {'fiedler':>8s} | {'ac_gap':>8s} | "
        f"{'tm_gap':>8s} | {'score':>8s} | time"
    )

    # Focused runs with 500 walkers, 1000 steps
    print("=" * 160)
    print("PARAMETER OPTIMIZATION (500 walkers, 1000 steps)")
    print("=" * 160)
    print(header)
    print("-" * 160)

    combos = [
        # Baseline
        ("default", {}),
        # Temperature sweep (moderate range, keeping asym healthy)
        ("T=0.5", {"temperature": 0.5}),
        ("T=0.7", {"temperature": 0.7}),
        ("T=1.0", {"temperature": 1.0}),
        # Best individual effects combined
        ("T=0.5_nu=2", {"temperature": 0.5, "nu": 2.0}),
        ("T=0.7_nu=2", {"temperature": 0.7, "nu": 2.0}),
        ("T=1.0_nu=2", {"temperature": 1.0, "nu": 2.0}),
        ("T=0.5_fit_a=2", {"temperature": 0.5, "fitness_alpha": 2.0}),
        ("T=0.7_fit_a=2", {"temperature": 0.7, "fitness_alpha": 2.0}),
        # viscous_length + nu
        ("T=0.5_nu=2_vl=2", {"temperature": 0.5, "nu": 2.0, "viscous_length_scale": 2.0}),
        ("T=0.7_nu=2_vl=0.1", {"temperature": 0.7, "nu": 2.0, "viscous_length_scale": 0.1}),
        # beta_curl combos
        ("T=0.5_bc=2_nu=2", {"temperature": 0.5, "beta_curl": 2.0, "nu": 2.0}),
        ("T=0.7_bc=0.5_nu=2", {"temperature": 0.7, "beta_curl": 0.5, "nu": 2.0}),
        # Moderate push on multiple fronts
        ("T=0.5_nu=2_fit_a=2", {"temperature": 0.5, "nu": 2.0, "fitness_alpha": 2.0}),
        ("T=0.7_nu=2_fit_a=2", {"temperature": 0.7, "nu": 2.0, "fitness_alpha": 2.0}),
        (
            "T=0.5_nu=2_fit_a=2_vl=2",
            {"temperature": 0.5, "nu": 2.0, "fitness_alpha": 2.0, "viscous_length_scale": 2.0},
        ),
        # Higher nu with moderate T
        ("T=0.5_nu=3", {"temperature": 0.5, "nu": 3.0}),
        ("T=0.7_nu=3", {"temperature": 0.7, "nu": 3.0}),
    ]

    for name, overrides in combos:
        p = defaults()
        p.update(overrides)
        try:
            r = run_and_diagnose(p)
            report(name, r)
        except Exception as e:
            print(f"  {name:40s} | ERROR: {e}", flush=True)
