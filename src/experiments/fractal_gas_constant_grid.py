"""
Calibrate and sweep Fractal Gas parameters on a constant reward landscape.

This script:
1) Runs a baseline ("calibration") simulation with constant reward, uniform init,
   and periodic boundary conditions.
2) Perturbs each selected parameter (positive/negative) and reruns the simulation.
3) Computes baryon/meson/glueball masses and ratios for each run.
4) Stores all run artifacts (history, metrics, arrays, config) for later analysis.

Defaults match the user request:
- N=200 walkers
- n_steps=200
- uniform positions across bounds
- pbc=True

Notes:
- Glueball mass requires building a FractalSet (enabled by default).
- By default this is a one-parameter-at-a-time sweep (not full factorial).
- Calibration defaults target proton/pion/f0(1710) mass ratios unless overridden.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from fragile.fractalai.bounds import TorchBounds
from fragile.fractalai.core.cloning import CloneOperator
from fragile.fractalai.core.companion_selection import CompanionSelection
from fragile.fractalai.core.euclidean_gas import EuclideanGas
from fragile.fractalai.core.fitness import FitnessOperator
from fragile.fractalai.core.fractal_set import FractalSet
from fragile.fractalai.core.kinetic_operator import KineticOperator
from fragile.fractalai.qft.analysis import (
    _compute_particle_observables,
    _compute_wilson_timeseries,
    _downsample_history,
    AnalysisConfig,
)


@dataclass
class EnvConfig:
    dims: int = 3
    bounds_extent: float = 10.0


@dataclass
class OperatorConfig:
    gamma: float = 1.0
    beta: float = 1.0
    delta_t: float = 0.01
    epsilon_F: float = 994.399
    use_fitness_force: bool = False
    use_potential_force: bool = True
    use_anisotropic_diffusion: bool = False
    diagonal_diffusion: bool = True
    epsilon_Sigma: float = 0.1
    nu: float = 0.948271
    use_viscous_coupling: bool = True
    viscous_length_scale: float = 0.00976705
    viscous_neighbor_mode: str = "all"
    viscous_neighbor_threshold: float | None = None
    viscous_neighbor_penalty: float = 0.0
    viscous_degree_cap: float | None = None
    beta_curl: float = 0.0
    use_velocity_squashing: bool = False
    V_alg: float = float("inf")
    companion_method: str = "softmax"
    companion_epsilon: float = 2.12029
    companion_epsilon_clone: float = 1.68419
    lambda_alg: float = 1.0
    exclude_self: bool = True
    p_max: float = 1.0
    epsilon_clone: float = 0.01
    sigma_x: float = 0.1
    alpha_restitution: float = 0.5
    fitness_alpha: float = 1.0
    fitness_beta: float = 1.0
    fitness_eta: float = 0.1
    fitness_A: float = 2.0
    fitness_sigma_min: float = 1e-8
    fitness_epsilon_dist: float = 1e-8
    fitness_rho: float | None = None


@dataclass
class RunConfig:
    N: int = 200
    n_steps: int = 200
    record_every: int = 1
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float32"
    pbc: bool = True
    record_rng_state: bool = True
    init_velocity_scale: float | None = None


DEFAULT_SWEEP_PARAMS = (
    "gamma",
    "beta",
    "delta_t",
    "epsilon_F",
    "nu",
    "viscous_length_scale",
    "companion_epsilon",
    "companion_epsilon_clone",
    "epsilon_clone",
    "sigma_x",
    "alpha_restitution",
    "fitness_eta",
)

POSITIVE_PARAMS = {
    "beta",
    "delta_t",
    "epsilon_F",
    "epsilon_Sigma",
    "viscous_length_scale",
    "companion_epsilon",
    "companion_epsilon_clone",
    "epsilon_clone",
    "sigma_x",
    "fitness_alpha",
    "fitness_beta",
    "fitness_eta",
    "fitness_A",
    "fitness_sigma_min",
    "fitness_epsilon_dist",
}

NONNEGATIVE_PARAMS = {
    "gamma",
    "nu",
    "viscous_neighbor_threshold",
    "viscous_neighbor_penalty",
    "viscous_degree_cap",
    "beta_curl",
    "lambda_alg",
    "p_max",
    "alpha_restitution",
    "fitness_rho",
}

PARAM_BOUNDS = {
    "p_max": (0.0, 1.0),
    "viscous_neighbor_threshold": (0.0, 1.0),
    "alpha_restitution": (0.0, 1.0),
}

DEFAULT_MASS_BARYON_MEV = 938.27208816
DEFAULT_MASS_MESON_MEV = 139.57039
DEFAULT_MASS_GLUEBALL_MEV = 1733.0
DEFAULT_TARGET_PARTICLES = {
    "baryon": "proton",
    "meson": "pi+",
    "glueball": "f0(1710)",
}


class ConstantPotential:
    def __init__(self, bounds: TorchBounds):
        self.bounds = bounds

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Keep autograd graph connected to x (zero potential with defined gradient).
        return x.sum(dim=-1) * 0.0


def _torch_dtype(dtype: str) -> torch.dtype:
    return torch.float64 if dtype == "float64" else torch.float32


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray | torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and (value == float("inf") or value == float("-inf")):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, float) and value != value:
        return "nan"
    return value


def _format_tag(value: float) -> str:
    text = f"{value:+.4g}"
    return text.replace("+", "plus_").replace("-", "minus_").replace(".", "p")


def build_bounds(env_cfg: EnvConfig, dtype: torch.dtype) -> TorchBounds:
    low = torch.full((env_cfg.dims,), -env_cfg.bounds_extent, dtype=dtype)
    high = torch.full((env_cfg.dims,), env_cfg.bounds_extent, dtype=dtype)
    return TorchBounds(low=low, high=high)


def uniform_init(
    bounds: TorchBounds,
    run_cfg: RunConfig,
    op_cfg: OperatorConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = _torch_dtype(run_cfg.dtype)
    device = torch.device(run_cfg.device)
    x_init = bounds.sample(run_cfg.N).to(device=device, dtype=dtype)

    if run_cfg.init_velocity_scale is None:
        v_std = 1.0 / torch.sqrt(torch.tensor(op_cfg.beta, dtype=dtype))
        v_init = v_std * torch.randn(run_cfg.N, bounds.shape[0], device=device, dtype=dtype)
    elif run_cfg.init_velocity_scale == 0.0:
        v_init = torch.zeros(run_cfg.N, bounds.shape[0], device=device, dtype=dtype)
    else:
        v_init = run_cfg.init_velocity_scale * torch.randn(
            run_cfg.N, bounds.shape[0], device=device, dtype=dtype
        )
    return x_init, v_init


def build_gas(
    env_cfg: EnvConfig,
    op_cfg: OperatorConfig,
    run_cfg: RunConfig,
) -> tuple[EuclideanGas, ConstantPotential, TorchBounds]:
    dtype = _torch_dtype(run_cfg.dtype)
    device = torch.device(run_cfg.device)
    bounds = build_bounds(env_cfg, dtype=dtype)
    potential = ConstantPotential(bounds)

    companion = CompanionSelection(
        method=op_cfg.companion_method,
        epsilon=op_cfg.companion_epsilon,
        lambda_alg=op_cfg.lambda_alg,
        exclude_self=op_cfg.exclude_self,
    )
    companion_clone = CompanionSelection(
        method=op_cfg.companion_method,
        epsilon=op_cfg.companion_epsilon_clone,
        lambda_alg=op_cfg.lambda_alg,
        exclude_self=op_cfg.exclude_self,
    )

    fitness_op = FitnessOperator(
        alpha=op_cfg.fitness_alpha,
        beta=op_cfg.fitness_beta,
        eta=op_cfg.fitness_eta,
        lambda_alg=op_cfg.lambda_alg,
        sigma_min=op_cfg.fitness_sigma_min,
        epsilon_dist=op_cfg.fitness_epsilon_dist,
        A=op_cfg.fitness_A,
        rho=op_cfg.fitness_rho,
    )

    kinetic_op = KineticOperator(
        gamma=op_cfg.gamma,
        beta=op_cfg.beta,
        delta_t=op_cfg.delta_t,
        epsilon_F=op_cfg.epsilon_F,
        use_fitness_force=op_cfg.use_fitness_force,
        use_potential_force=op_cfg.use_potential_force,
        use_anisotropic_diffusion=op_cfg.use_anisotropic_diffusion,
        diagonal_diffusion=op_cfg.diagonal_diffusion,
        epsilon_Sigma=op_cfg.epsilon_Sigma,
        nu=op_cfg.nu,
        use_viscous_coupling=op_cfg.use_viscous_coupling,
        viscous_length_scale=op_cfg.viscous_length_scale,
        viscous_neighbor_mode=op_cfg.viscous_neighbor_mode,
        viscous_neighbor_threshold=op_cfg.viscous_neighbor_threshold,
        viscous_neighbor_penalty=op_cfg.viscous_neighbor_penalty,
        viscous_degree_cap=op_cfg.viscous_degree_cap,
        beta_curl=op_cfg.beta_curl,
        use_velocity_squashing=op_cfg.use_velocity_squashing,
        V_alg=op_cfg.V_alg,
        potential=potential,
        device=device,
        dtype=dtype,
        bounds=bounds,
        pbc=run_cfg.pbc,
    )

    cloning = CloneOperator(
        p_max=op_cfg.p_max,
        epsilon_clone=op_cfg.epsilon_clone,
        sigma_x=op_cfg.sigma_x,
        alpha_restitution=op_cfg.alpha_restitution,
    )

    gas = EuclideanGas(
        N=run_cfg.N,
        d=env_cfg.dims,
        potential=potential,
        companion_selection=companion,
        companion_selection_clone=companion_clone,
        kinetic_op=kinetic_op,
        cloning=cloning,
        fitness_op=fitness_op,
        bounds=bounds,
        device=device,
        dtype=run_cfg.dtype,
        enable_cloning=True,
        enable_kinetic=True,
        pbc=run_cfg.pbc,
    )

    return gas, potential, bounds


def compute_glueball_data(history, stride: int) -> dict[str, np.ndarray] | None:
    history_small = _downsample_history(history, stride)
    fractal_set = FractalSet(history_small)
    wilson_timeseries = _compute_wilson_timeseries(fractal_set)
    time_idx = wilson_timeseries["time_index"]
    if time_idx.size == 0:
        return None
    recorded_steps = np.array(history_small.recorded_steps, dtype=np.float64)
    tau = recorded_steps[time_idx] * history_small.delta_t
    return {
        "time_index": time_idx,
        "tau": tau,
        "series": wilson_timeseries["action_mean"],
    }


def analyze_particles(
    history,
    analysis_cfg: AnalysisConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    operators = tuple(op.strip().lower() for op in analysis_cfg.particle_operators if op)
    glueball_data = None
    if "glueball" in operators:
        glueball_data = compute_glueball_data(history, analysis_cfg.fractal_set_stride)
    metrics, arrays = _compute_particle_observables(
        history=history,
        operators=operators,
        h_eff=analysis_cfg.h_eff,
        mass=analysis_cfg.particle_mass,
        ell0=analysis_cfg.particle_ell0,
        max_lag=analysis_cfg.particle_max_lag,
        fit_start=analysis_cfg.particle_fit_start,
        fit_stop=analysis_cfg.particle_fit_stop,
        fit_mode=analysis_cfg.particle_fit_mode,
        plateau_min_points=analysis_cfg.particle_plateau_min_points,
        plateau_max_points=analysis_cfg.particle_plateau_max_points,
        plateau_max_cv=analysis_cfg.particle_plateau_max_cv,
        use_connected=analysis_cfg.particle_use_connected,
        neighbor_method=analysis_cfg.particle_neighbor_method,
        knn_k=analysis_cfg.particle_knn_k,
        knn_sample=analysis_cfg.particle_knn_sample,
        meson_reduce=analysis_cfg.particle_meson_reduce,
        baryon_pairs=analysis_cfg.particle_baryon_pairs,
        warmup_fraction=analysis_cfg.warmup_fraction,
        glueball_data=glueball_data,
    )
    return metrics, arrays


def extract_masses(metrics: dict[str, Any]) -> dict[str, float]:
    operators = metrics.get("operators") or {}

    def _get(op: str) -> tuple[float, float, float]:
        fit = operators.get(op, {}).get("fit", {})
        return (
            float(fit.get("mass", float("nan"))),
            float(fit.get("r_squared", float("nan"))),
            float(fit.get("fit_points", 0.0)),
        )

    baryon_mass, baryon_r2, baryon_fit_points = _get("baryon")
    meson_mass, meson_r2, meson_fit_points = _get("meson")
    glueball_mass, glueball_r2, glueball_fit_points = _get("glueball")

    def _ratio(num: float, den: float) -> float:
        return float(num / den) if den and den > 0 else float("nan")

    return {
        "baryon_mass": baryon_mass,
        "baryon_r2": baryon_r2,
        "baryon_fit_points": baryon_fit_points,
        "meson_mass": meson_mass,
        "meson_r2": meson_r2,
        "meson_fit_points": meson_fit_points,
        "glueball_mass": glueball_mass,
        "glueball_r2": glueball_r2,
        "glueball_fit_points": glueball_fit_points,
        "baryon_over_meson": _ratio(baryon_mass, meson_mass),
        "glueball_over_meson": _ratio(glueball_mass, meson_mass),
        "baryon_over_glueball": _ratio(baryon_mass, glueball_mass),
    }


def _is_finite_positive(value: float) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) > 0.0


def score_calibration(
    result: dict[str, Any],
    ratio_targets: dict[str, float | None],
    r2_min: float,
    fit_points_min: int,
    ratio_weight: float,
    r2_weight: float,
) -> tuple[float, dict[str, Any]]:
    reasons: list[str] = []
    r2_penalty = 0.0

    for op in ("baryon", "meson", "glueball"):
        mass = result.get(f"{op}_mass", float("nan"))
        r2 = result.get(f"{op}_r2", float("nan"))
        fit_points = result.get(f"{op}_fit_points", 0.0)

        if not _is_finite_positive(mass):
            reasons.append(f"{op}_mass")
        if not math.isfinite(float(r2)):
            reasons.append(f"{op}_r2")
        if float(fit_points) < fit_points_min:
            reasons.append(f"{op}_fit_points")
        if math.isfinite(float(r2)):
            r2_penalty += max(0.0, r2_min - float(r2))

    if reasons:
        return float("inf"), {"score_reason": ";".join(reasons), "r2_penalty": None}

    ratio_penalty = 0.0
    ratio_errors: dict[str, float] = {}
    for ratio_key, target in ratio_targets.items():
        if target is None or not math.isfinite(float(target)):
            continue
        pred = result.get(ratio_key, float("nan"))
        if not math.isfinite(float(pred)):
            reasons.append(ratio_key)
            continue
        err = abs(float(pred) - float(target))
        if target != 0:
            err /= abs(float(target))
        ratio_errors[f"{ratio_key}_err"] = float(err)
        ratio_penalty += float(err)

    if reasons:
        return float("inf"), {"score_reason": ";".join(reasons), "r2_penalty": None}

    score = ratio_weight * ratio_penalty + r2_weight * r2_penalty
    details = {
        "score_reason": None,
        "ratio_penalty": ratio_penalty,
        "r2_penalty": r2_penalty,
        **ratio_errors,
    }
    return float(score), details


def apply_overrides(op_cfg: OperatorConfig, overrides: Iterable[str]) -> None:
    for item in overrides:
        if "=" not in item:
            msg = f"Invalid --set value '{item}'. Use param=value."
            raise ValueError(msg)
        name, value_raw = item.split("=", 1)
        name = name.strip()
        value_raw = value_raw.strip()
        if not hasattr(op_cfg, name):
            msg = f"Unknown operator parameter '{name}'."
            raise ValueError(msg)
        current = getattr(op_cfg, name)
        if value_raw.lower() in {"none", "null"}:
            value = None
        elif isinstance(current, bool):
            value = value_raw.lower() in {"1", "true", "yes", "y"}
        elif isinstance(current, int):
            value = int(value_raw)
        elif isinstance(current, float) or current is None:
            value = float(value_raw)
        else:
            value = value_raw
        setattr(op_cfg, name, value)


def _coerce_value(param: str, value: float) -> float:
    if param in PARAM_BOUNDS:
        low, high = PARAM_BOUNDS[param]
        value = max(low, min(high, value))
    if param in POSITIVE_PARAMS and value <= 0:
        return float("nan")
    if param in NONNEGATIVE_PARAMS and value < 0:
        return float("nan")
    return value


def generate_perturbations(
    param: str,
    base_value: Any,
    fracs: list[float],
    zero_perturb: float,
) -> list[float]:
    if base_value is None:
        return []
    if isinstance(base_value, bool) or isinstance(base_value, str):
        return []
    if isinstance(base_value, float | int) and not math.isfinite(float(base_value)):
        return []

    values = []
    if float(base_value) == 0.0:
        for frac in fracs:
            values.extend([-zero_perturb * frac, zero_perturb * frac])
    else:
        base_float = float(base_value)
        for frac in fracs:
            values.append(base_float * (1.0 - frac))
            values.append(base_float * (1.0 + frac))
    return values


def run_single(
    output_dir: Path,
    run_tag: str,
    env_cfg: EnvConfig,
    op_cfg: OperatorConfig,
    run_cfg: RunConfig,
    analysis_cfg: AnalysisConfig,
    seed: int,
    save_history: bool,
    save_arrays: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = RunConfig(**asdict(run_cfg))
    run_cfg.seed = seed

    if seed is not None:
        torch.manual_seed(seed)
        try:
            np.random.seed(seed)
        except Exception:
            pass

    gas, _potential, bounds = build_gas(env_cfg, op_cfg, run_cfg)
    x_init, v_init = uniform_init(bounds, run_cfg, op_cfg)

    history = gas.run(
        n_steps=run_cfg.n_steps,
        record_every=run_cfg.record_every,
        seed=run_cfg.seed,
        record_rng_state=run_cfg.record_rng_state,
        show_progress=False,
        x_init=x_init,
        v_init=v_init,
    )

    summary_text = history.summary()

    if save_history:
        history_path = run_dir / f"{run_tag}_history.pt"
        history.save(str(history_path))
        (run_dir / f"{run_tag}_summary.txt").write_text(summary_text, encoding="utf-8")
    else:
        history_path = None

    metrics, arrays = analyze_particles(history, analysis_cfg)
    masses = extract_masses(metrics)

    metrics_path = run_dir / f"{run_tag}_metrics.json"
    metrics_path.write_text(json.dumps(_json_safe(metrics), indent=2, sort_keys=True))

    arrays_path = None
    if save_arrays:
        arrays_path = run_dir / f"{run_tag}_arrays.npz"
        np.savez(arrays_path, **arrays)

    config = {
        "env": asdict(env_cfg),
        "operators": asdict(op_cfg),
        "run": asdict(run_cfg),
        "analysis": asdict(analysis_cfg),
        "history_path": str(history_path) if history_path else None,
        "metrics_path": str(metrics_path),
        "arrays_path": str(arrays_path) if arrays_path else None,
    }
    config_path = run_dir / f"{run_tag}_config.json"
    config_path.write_text(json.dumps(_json_safe(config), indent=2, sort_keys=True))

    return {
        "run_tag": run_tag,
        "run_dir": str(run_dir),
        "history_path": str(history_path) if history_path else None,
        "metrics_path": str(metrics_path),
        "arrays_path": str(arrays_path) if arrays_path else None,
        **masses,
    }


def _run_task(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        output_dir,
        run_tag,
        env_cfg,
        op_cfg,
        run_cfg,
        analysis_cfg,
        seed,
        save_history,
        save_arrays,
    ) = task
    torch.set_num_threads(1)
    return run_single(
        Path(output_dir),
        run_tag,
        env_cfg,
        op_cfg,
        run_cfg,
        analysis_cfg,
        seed,
        save_history,
        save_arrays,
    )


def _execute_tasks(tasks: list[tuple[Any, ...]], num_workers: int) -> list[dict[str, Any]]:
    if not tasks:
        return []
    if num_workers <= 1:
        return [_run_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(_run_task, tasks))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/fractal_gas_constant_grid")
    parser.add_argument("--grid-id", default=None)
    parser.add_argument("--dims", type=int, default=EnvConfig.dims)
    parser.add_argument("--bounds-extent", type=float, default=EnvConfig.bounds_extent)
    parser.add_argument("--N", type=int, default=RunConfig.N)
    parser.add_argument("--n-steps", type=int, default=RunConfig.n_steps)
    parser.add_argument("--record-every", type=int, default=RunConfig.record_every)
    parser.add_argument("--seed", type=int, default=RunConfig.seed)
    parser.add_argument(
        "--seed-mode",
        choices=["fixed", "offset"],
        default="fixed",
        help="fixed: same seed for each run; offset: seed+run_index",
    )
    parser.add_argument("--device", default=RunConfig.device)
    parser.add_argument("--dtype", choices=["float32", "float64"], default=RunConfig.dtype)
    default_workers = min(32, os.cpu_count() or 1)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=default_workers,
        help="Number of parallel worker processes to run simulations.",
    )
    parser.add_argument("--no-pbc", dest="pbc", action="store_false")
    parser.set_defaults(pbc=True)
    parser.add_argument(
        "--init-velocity-scale",
        type=float,
        default=None,
        help="If set, overrides thermal velocities. Use 0 for zero velocities.",
    )

    parser.add_argument(
        "--params",
        default=",".join(DEFAULT_SWEEP_PARAMS),
        help="Comma-separated operator parameters to perturb.",
    )
    parser.add_argument(
        "--perturb-fracs",
        default="0.02,0.05,0.1,0.2,0.3",
        help="Comma-separated fractional perturbations (5 values by default).",
    )
    parser.add_argument(
        "--zero-perturb",
        type=float,
        default=0.01,
        help="Absolute perturbation scale when base value is 0.",
    )
    parser.add_argument(
        "--calibration-params",
        default=None,
        help="Comma-separated operator parameters to calibrate (defaults to --params).",
    )
    parser.add_argument(
        "--calibration-perturb-fracs",
        default=None,
        help="Comma-separated calibration perturbations (defaults to --perturb-fracs).",
    )
    parser.add_argument(
        "--calibration-zero-perturb",
        type=float,
        default=None,
        help="Absolute perturbation when base value is 0 (defaults to --zero-perturb).",
    )
    parser.add_argument("--target-baryon-meson", type=float, default=None)
    parser.add_argument("--target-glueball-meson", type=float, default=None)
    parser.add_argument("--target-baryon-glueball", type=float, default=None)
    parser.add_argument("--calibration-r2-min", type=float, default=0.5)
    parser.add_argument("--calibration-fit-points-min", type=int, default=4)
    parser.add_argument("--calibration-ratio-weight", type=float, default=1.0)
    parser.add_argument("--calibration-r2-weight", type=float, default=1.0)

    parser.add_argument("--set", action="append", default=[], help="Override operator param.")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--no-save-history", dest="save_history", action="store_false")
    parser.add_argument("--no-save-arrays", dest="save_arrays", action="store_false")
    parser.set_defaults(save_history=True, save_arrays=True)

    # Particle analysis options
    parser.add_argument("--particle-max-lag", type=int, default=80)
    parser.add_argument("--particle-fit-start", type=int, default=7)
    parser.add_argument("--particle-fit-stop", type=int, default=16)
    parser.add_argument(
        "--particle-fit-mode",
        choices=["window", "plateau", "auto"],
        default="window",
    )
    parser.add_argument("--particle-plateau-min-points", type=int, default=3)
    parser.add_argument("--particle-plateau-max-points", type=int, default=None)
    parser.add_argument("--particle-plateau-max-cv", type=float, default=0.2)
    parser.add_argument("--particle-mass", type=float, default=1.0)
    parser.add_argument("--particle-ell0", type=float, default=None)
    parser.add_argument(
        "--particle-use-connected", dest="particle_use_connected", action="store_true"
    )
    parser.add_argument(
        "--no-particle-use-connected", dest="particle_use_connected", action="store_false"
    )
    parser.set_defaults(particle_use_connected=True)
    parser.add_argument(
        "--particle-neighbor-method",
        choices=["companion", "knn"],
        default="knn",
    )
    parser.add_argument("--particle-knn-k", type=int, default=4)
    parser.add_argument("--particle-knn-sample", type=int, default=512)
    parser.add_argument(
        "--particle-meson-reduce",
        choices=["mean", "first"],
        default="mean",
    )
    parser.add_argument("--particle-baryon-pairs", type=int, default=None)
    parser.add_argument("--fractal-set-stride", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env_cfg = EnvConfig(dims=args.dims, bounds_extent=args.bounds_extent)
    op_cfg = OperatorConfig()
    apply_overrides(op_cfg, args.set)
    base_op_cfg = OperatorConfig(**asdict(op_cfg))

    run_cfg = RunConfig(
        N=args.N,
        n_steps=args.n_steps,
        record_every=args.record_every,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        pbc=args.pbc,
        record_rng_state=True,
        init_velocity_scale=args.init_velocity_scale,
    )

    particle_ops = ("baryon", "meson", "glueball")
    analysis_cfg = AnalysisConfig(
        compute_particles=True,
        particle_operators=particle_ops,
        particle_max_lag=args.particle_max_lag,
        particle_fit_start=args.particle_fit_start,
        particle_fit_stop=args.particle_fit_stop,
        particle_fit_mode=args.particle_fit_mode,
        particle_plateau_min_points=args.particle_plateau_min_points,
        particle_plateau_max_points=args.particle_plateau_max_points,
        particle_plateau_max_cv=args.particle_plateau_max_cv,
        particle_mass=args.particle_mass,
        particle_ell0=args.particle_ell0,
        particle_use_connected=args.particle_use_connected,
        particle_neighbor_method=args.particle_neighbor_method,
        particle_knn_k=args.particle_knn_k,
        particle_knn_sample=args.particle_knn_sample,
        particle_meson_reduce=args.particle_meson_reduce,
        particle_baryon_pairs=args.particle_baryon_pairs,
        build_fractal_set=True,
        fractal_set_stride=args.fractal_set_stride,
    )

    grid_id = args.grid_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) / grid_id
    calibration_dir = output_root / "calibration"
    runs_dir = output_root / "runs"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    ratio_targets = {
        "baryon_over_meson": (
            args.target_baryon_meson
            if args.target_baryon_meson is not None
            else DEFAULT_MASS_BARYON_MEV / DEFAULT_MASS_MESON_MEV
        ),
        "glueball_over_meson": (
            args.target_glueball_meson
            if args.target_glueball_meson is not None
            else DEFAULT_MASS_GLUEBALL_MEV / DEFAULT_MASS_MESON_MEV
        ),
        "baryon_over_glueball": (
            args.target_baryon_glueball
            if args.target_baryon_glueball is not None
            else DEFAULT_MASS_BARYON_MEV / DEFAULT_MASS_GLUEBALL_MEV
        ),
    }

    calibration_params = (
        args.calibration_params if args.calibration_params is not None else args.params
    )
    calibration_fracs = (
        args.calibration_perturb_fracs
        if args.calibration_perturb_fracs is not None
        else args.perturb_fracs
    )
    calibration_zero = (
        args.calibration_zero_perturb
        if args.calibration_zero_perturb is not None
        else args.zero_perturb
    )

    calibration_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    def _seed_for(index: int) -> int:
        return run_cfg.seed if args.seed_mode == "fixed" else run_cfg.seed + index

    best_score = float("inf")
    best_op_cfg: OperatorConfig | None = None
    best_result: dict[str, Any] | None = None
    run_index = 0

    if not args.skip_calibration:
        calib_params = [p.strip() for p in calibration_params.split(",") if p.strip()]
        calib_fracs = [float(v.strip()) for v in calibration_fracs.split(",") if v.strip()]
        calibration_tasks: list[tuple[Any, ...]] = []
        calibration_meta: list[dict[str, Any]] = []

        def _add_calibration_task(
            tag: str,
            op_cfg_run: OperatorConfig,
            param: str,
            base_value: float | None,
            value: float | None,
        ) -> None:
            nonlocal run_index
            seed = _seed_for(run_index)
            run_index += 1
            calibration_tasks.append((
                str(calibration_dir),
                tag,
                env_cfg,
                op_cfg_run,
                run_cfg,
                analysis_cfg,
                seed,
                args.save_history,
                args.save_arrays,
            ))
            calibration_meta.append({
                "param": param,
                "base_value": base_value,
                "value": value,
                "op_cfg": op_cfg_run,
            })

        _add_calibration_task("baseline", op_cfg, "baseline", None, None)

        for param in calib_params:
            if not hasattr(op_cfg, param):
                print(f"Skipping unknown calibration parameter '{param}'.")
                continue
            base_value = getattr(op_cfg, param)
            perturb_values = generate_perturbations(
                param, base_value, calib_fracs, calibration_zero
            )
            if not perturb_values:
                print(f"Skipping parameter '{param}' (non-numeric or None).")
                continue
            for value in perturb_values:
                value = _coerce_value(param, float(value))
                if not math.isfinite(value):
                    continue
                tag = f"{param}_{_format_tag(value)}"
                op_cfg_run = OperatorConfig(**asdict(op_cfg))
                setattr(op_cfg_run, param, value)
                _add_calibration_task(tag, op_cfg_run, param, base_value, value)

        results = _execute_tasks(calibration_tasks, args.num_workers)
        for result, meta in zip(results, calibration_meta):
            score, details = score_calibration(
                result,
                ratio_targets,
                r2_min=args.calibration_r2_min,
                fit_points_min=args.calibration_fit_points_min,
                ratio_weight=args.calibration_ratio_weight,
                r2_weight=args.calibration_r2_weight,
            )
            result.update(details)
            result["score"] = score
            result.update({
                "param": meta["param"],
                "delta": (
                    float(meta["value"]) - float(meta["base_value"])
                    if meta["value"] is not None and meta["base_value"] is not None
                    else None
                ),
                "value": float(meta["value"]) if meta["value"] is not None else None,
            })
            calibration_rows.append(result)
            if score < best_score:
                best_score = score
                best_op_cfg = OperatorConfig(**asdict(meta["op_cfg"]))
                best_result = result

        if best_op_cfg is not None:
            op_cfg = best_op_cfg

    manifest = {
        "grid_id": grid_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "env": asdict(env_cfg),
        "operators_initial": asdict(base_op_cfg),
        "operators": asdict(op_cfg),
        "run": asdict(run_cfg),
        "analysis": asdict(analysis_cfg),
        "calibration": {
            "skipped": args.skip_calibration,
            "params": calibration_params,
            "perturb_fracs": calibration_fracs,
            "zero_perturb": calibration_zero,
            "targets": ratio_targets,
            "target_masses_mev": {
                "baryon": DEFAULT_MASS_BARYON_MEV,
                "meson": DEFAULT_MASS_MESON_MEV,
                "glueball": DEFAULT_MASS_GLUEBALL_MEV,
            },
            "target_particles": DEFAULT_TARGET_PARTICLES,
            "r2_min": args.calibration_r2_min,
            "fit_points_min": args.calibration_fit_points_min,
            "ratio_weight": args.calibration_ratio_weight,
            "r2_weight": args.calibration_r2_weight,
            "best_score": None if not math.isfinite(best_score) else float(best_score),
            "best_run": best_result,
            "runs": calibration_rows,
        },
        "runs": [],
    }

    if calibration_rows:
        _write_summary(output_root, "calibration_summary", calibration_rows)

    if args.calibration_only:
        _write_manifest(output_root, manifest)
        print(f"Saved outputs to {output_root}")
        return

    sweep_tasks: list[tuple[Any, ...]] = []
    sweep_meta: list[dict[str, Any]] = []

    def _add_sweep_task(
        tag: str,
        op_cfg_run: OperatorConfig,
        param: str,
        base_value: float | None,
        value: float | None,
    ) -> None:
        nonlocal run_index
        seed = _seed_for(run_index)
        run_index += 1
        sweep_tasks.append((
            str(runs_dir),
            tag,
            env_cfg,
            op_cfg_run,
            run_cfg,
            analysis_cfg,
            seed,
            args.save_history,
            args.save_arrays,
        ))
        sweep_meta.append({
            "param": param,
            "base_value": base_value,
            "value": value,
        })

    baseline_tag = "baseline" if args.skip_calibration else "calibrated_baseline"
    _add_sweep_task(baseline_tag, op_cfg, "baseline", None, None)

    params = [p.strip() for p in args.params.split(",") if p.strip()]
    fracs = [float(v.strip()) for v in args.perturb_fracs.split(",") if v.strip()]

    for param in params:
        if not hasattr(op_cfg, param):
            print(f"Skipping unknown parameter '{param}'.")
            continue
        base_value = getattr(op_cfg, param)
        perturb_values = generate_perturbations(param, base_value, fracs, args.zero_perturb)
        if not perturb_values:
            print(f"Skipping parameter '{param}' (non-numeric or None).")
            continue
        for value in perturb_values:
            value = _coerce_value(param, float(value))
            if not math.isfinite(value):
                continue
            tag = f"{param}_{_format_tag(value)}"
            op_cfg_run = OperatorConfig(**asdict(op_cfg))
            setattr(op_cfg_run, param, value)
            _add_sweep_task(tag, op_cfg_run, param, base_value, value)

    sweep_results = _execute_tasks(sweep_tasks, args.num_workers)
    for result, meta in zip(sweep_results, sweep_meta):
        result.update({
            "param": meta["param"],
            "delta": (
                float(meta["value"]) - float(meta["base_value"])
                if meta["value"] is not None and meta["base_value"] is not None
                else None
            ),
            "value": float(meta["value"]) if meta["value"] is not None else None,
        })
        summary_rows.append(result)
        manifest["runs"].append(result)

    _write_summary(output_root, "summary", summary_rows)
    _write_manifest(output_root, manifest)
    print(f"Saved outputs to {output_root}")


def _write_summary(output_root: Path, prefix: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    summary_path = output_root / f"{prefix}.csv"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(",".join(str(row.get(key, "")) for key in keys) + "\n")

    jsonl_path = output_root / f"{prefix}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row)) + "\n")


def _write_manifest(output_root: Path, manifest: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
