#!/usr/bin/env python3
"""Baseline QFT run + channels-only mass extraction.

Runs a Euclidean Gas simulation with the same channel analysis used in the QFT
Dashboard Channels tab, then stores extracted masses and the selected parameters.
Also saves full channel analysis payloads to support tuning against ratio targets.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fragile.fractalai.qft.correlator_channels import (
    ChannelConfig,
    compute_all_channels,
    compute_correlator_fft,
    compute_effective_mass_torch,
    get_channel_class,
)
from fragile.fractalai.qft.simulation import (
    OperatorConfig,
    PotentialWellConfig,
    RunConfig,
    run_simulation,
    save_outputs,

)
TARGET_RATIOS = {
    "rho_pi": 5.5,
    "nucleon_pi": 6.7,
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (value == float("inf") or value == float("-inf")):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, float) and value != value:
        return "nan"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _parse_window_widths(spec: str) -> list[int]:
    """Parse '5-50' or '5,10,15' into list of ints."""
    if "-" in spec and "," not in spec:
        parts = spec.split("-")
        if len(parts) == 2:
            try:
                start, end = int(parts[0]), int(parts[1])
                return list(range(start, end + 1))
            except ValueError:
                pass
    try:
        return [int(x.strip()) for x in spec.split(",") if x.strip()]
    except ValueError:
        return list(range(5, 51))


def _build_channel_config(args: argparse.Namespace) -> tuple[ChannelConfig, dict[str, Any]]:
    # Match the Channels tab defaults where possible. Extra keys are ignored.
    window_widths = _parse_window_widths(args.window_widths)
    desired = {
        "warmup_fraction": args.warmup_fraction,
        "max_lag": args.max_lag,
        "h_eff": args.h_eff,
        "mass": args.channel_mass,
        "ell0": args.ell0,
        "knn_k": args.knn_k,
        "knn_sample": args.knn_sample,
        "use_connected": args.use_connected,
        "window_widths": window_widths,
        "fit_mode": args.fit_mode,
        "fit_start": args.fit_start,
        "fit_stop": args.fit_stop,
        "min_fit_points": args.min_fit_points,
    }
    sig = inspect.signature(ChannelConfig)
    kwargs = {k: v for k, v in desired.items() if k in sig.parameters}
    return ChannelConfig(**kwargs), kwargs

def _extract_fit_value(fit: Any, key: str) -> float | None:
    if isinstance(fit, dict):
        value = fit.get(key)
    else:
        value = getattr(fit, key, None)
    return float(value) if isinstance(value, (int, float)) else None


def _extract_masses(
    results: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    masses: dict[str, float] = {}
    r2s: dict[str, float] = {}
    summaries: dict[str, Any] = {}

    for name, result in results.items():
        if isinstance(result, dict):
            fit = result.get("mass_fit") or result.get("fit")
            summary = {
                "mass_fit": result.get("mass_fit"),
                "fit": result.get("fit"),
                "plateau": result.get("plateau"),
                "window": result.get("window"),
                "aic": result.get("aic"),
            }
        else:
            fit = getattr(result, "mass_fit", None) or getattr(result, "fit", None)
            summary = {
                "mass_fit": getattr(result, "mass_fit", None),
                "fit": getattr(result, "fit", None),
                "plateau": getattr(result, "plateau", None),
                "window": getattr(result, "window", None),
                "aic": getattr(result, "aic", None),
            }

        mass = _extract_fit_value(fit, "mass") if fit is not None else None
        if mass is None and isinstance(fit, dict):
            mass = _extract_fit_value(fit, "m")
        r2 = _extract_fit_value(fit, "r_squared") if fit is not None else None
        if r2 is None and isinstance(fit, dict):
            r2 = _extract_fit_value(fit, "r2")

        if mass is not None:
            masses[name] = mass
        if r2 is not None:
            r2s[name] = r2

        summaries[name] = summary

    return masses, r2s, summaries


def _pack_results(obj: Any, arrays: dict[str, np.ndarray], prefix: str = "arr") -> Any:
    if isinstance(obj, torch.Tensor):
        key = f"{prefix}_{len(arrays)}"
        arrays[key] = obj.detach().cpu().numpy()
        return {"__tensor__": key, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, np.ndarray):
        key = f"{prefix}_{len(arrays)}"
        arrays[key] = obj
        return {"__ndarray__": key, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: _pack_results(v, arrays, f"{prefix}_{k}") for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_pack_results(v, arrays, f"{prefix}_i") for v in obj]
    if hasattr(obj, "__dict__"):
        return _pack_results(obj.__dict__, arrays, prefix)
    return repr(obj)


def _save_full_analysis(
    results: dict[str, Any],
    output_dir: Path,
    run_id: str,
) -> tuple[Path, Path | None]:
    arrays: dict[str, np.ndarray] = {}
    packed = _pack_results(results, arrays, "chan")

    arrays_path = None
    if arrays:
        arrays_path = output_dir / f"{run_id}_channels_arrays.npz"
        np.savez_compressed(arrays_path, **arrays)

    full_path = output_dir / f"{run_id}_channels_full.json"
    _write_json(
        full_path,
        {
            "results": packed,
            "arrays_path": str(arrays_path) if arrays_path is not None else None,
        },

    )
    return full_path, arrays_path


def _compute_channel_transform(
    history: Any,
    config: ChannelConfig,
    channel: str,
    mode: str,
) -> dict[str, Any]:
    channel_class = get_channel_class(channel)
    channel_obj = channel_class(history, config)
    series = channel_obj.compute_series()
    if series.numel() == 0:
        return {
            "channel_name": f"{channel}_{mode}",
            "correlator": torch.zeros(config.max_lag + 1),
            "correlator_err": None,
            "effective_mass": torch.zeros(config.max_lag),
            "mass_fit": {"mass": 0.0, "mass_error": float("inf")},
            "series": series,
            "n_samples": 0,
            "dt": float(history.delta_t * history.record_every),
            "window_masses": None,
            "window_aic": None,
            "window_widths": None,
            "window_r2": None,
        }

    if mode == "abs":
        series_transformed = series.abs()
    elif mode == "abs2":
        series_transformed = series.abs() ** 2
    else:
        series_transformed = series
    correlator = compute_correlator_fft(
        series_transformed,
        max_lag=config.max_lag,
        use_connected=False,
    )
    dt = float(history.delta_t * history.record_every)
    effective_mass = compute_effective_mass_torch(correlator, dt)
    if config.fit_mode == "linear_abs":
        mass_fit = channel_obj.extract_mass_linear_abs(correlator)
    elif config.fit_mode == "linear":
        mass_fit = channel_obj.extract_mass_linear(correlator)
        window_masses = None
        window_aic = None
        window_widths = None
        window_r2 = None
    else:
        mass_fit = channel_obj.extract_mass_aic(correlator)
        window_masses = mass_fit.pop("window_masses", None)
        window_aic = mass_fit.pop("window_aic", None)
        window_widths = mass_fit.pop("window_widths", None)
        window_r2 = mass_fit.pop("window_r2", None)

    return {
        "channel_name": f"{channel}_{mode}",
        "correlator": correlator,
        "correlator_err": None,
        "effective_mass": effective_mass,
        "mass_fit": mass_fit,
        "series": series_transformed,
        "series_raw": series,
        "n_samples": int(series_transformed.numel()),
        "dt": dt,
        "window_masses": window_masses,
        "window_aic": window_aic,
        "window_widths": window_widths,
        "window_r2": window_r2,
    }


def _compute_ratios(
    masses: dict[str, float],
    use_nucleon_abs: bool,
    use_vector_abs2: bool,
) -> dict[str, Any]:
    ratios: dict[str, Any] = {}

    pi = masses.get("pseudoscalar") or masses.get("pi")
    rho = masses.get("vector") or masses.get("rho")
    rho_abs2 = masses.get("vector_abs2")
    nucleon = masses.get("nucleon") or masses.get("baryon") or masses.get("N")
    nucleon_abs = masses.get("nucleon_abs")
    nucleon_abs2 = masses.get("nucleon_abs2")

    if pi and nucleon_abs and pi > 0:
        ratios["nucleon_pi_abs"] = nucleon_abs / pi
    if pi and nucleon_abs2 and pi > 0:
        ratios["nucleon_pi_abs2"] = nucleon_abs2 / pi

    if use_nucleon_abs and nucleon_abs2 is not None:
        nucleon = nucleon_abs2
        ratios["nucleon_pi_source"] = "nucleon_abs2"
    elif use_nucleon_abs and nucleon_abs is not None:
        nucleon = nucleon_abs
        ratios["nucleon_pi_source"] = "nucleon_abs"

    if pi and rho_abs2 and pi > 0:
        ratios["rho_pi_abs2"] = rho_abs2 / pi
    if use_vector_abs2 and rho_abs2 is not None:
        rho = rho_abs2
        ratios["rho_pi_source"] = "vector_abs2"

    if pi and rho and pi > 0:
        ratios["rho_pi"] = rho / pi
        ratios["rho_pi_error"] = ratios["rho_pi"] - TARGET_RATIOS["rho_pi"]
    if pi and nucleon and pi > 0:
        ratios["nucleon_pi"] = nucleon / pi
        ratios["nucleon_pi_error"] = ratios["nucleon_pi"] - TARGET_RATIOS["nucleon_pi"]

    return ratios


def _build_curl_field(dims: int, mode: str, strength: float) -> callable | None:
    if mode == "none":
        return None
    if mode == "constant":
        if dims == 3:
            def _curl(x: torch.Tensor) -> torch.Tensor:
                curl = torch.zeros_like(x)
                curl[:, 2] = strength
                return curl
            return _curl
        if dims >= 2:
            def _curl(x: torch.Tensor) -> torch.Tensor:
                N = x.shape[0]
                curl = torch.zeros((N, dims, dims), device=x.device, dtype=x.dtype)
                curl[:, 0, 1] = strength
                curl[:, 1, 0] = -strength
                return curl
            return _curl
        raise ValueError("curl-mode constant requires dims >= 2.")
    if mode == "radial":
        if dims != 3:
            raise ValueError("curl-mode radial is only supported for dims=3.")
        def _curl(x: torch.Tensor) -> torch.Tensor:
            norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1e-8)
            return strength * x / norm
        return _curl
    if mode == "plane":
        if dims < 2:
            raise ValueError("curl-mode plane requires dims >= 2.")
        def _curl(x: torch.Tensor) -> torch.Tensor:
            N = x.shape[0]
            curl = torch.zeros((N, dims, dims), device=x.device, dtype=x.dtype)
            curl[:, 0, 1] = strength
            curl[:, 1, 0] = -strength
            return curl
        return _curl
    raise ValueError(f"Unknown curl-mode {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/qft_baseline")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--dims", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--bounds-extent", type=float, default=10.0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--zero-reward",
        action="store_true",
        default=True,
        help="Use constant zero reward (dashboard Constant benchmark).",
    )
    parser.add_argument(
        "--no-zero-reward",
        action="store_false",
        dest="zero_reward",
        help="Disable constant zero reward and use potential-based reward.",
    )

    # Selected operator overrides (tuning knobs)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--delta-t", type=float, default=None)
    parser.add_argument("--epsilon-F", type=float, default=None)
    parser.add_argument("--epsilon-sigma", type=float, default=None)
    parser.add_argument("--nu", type=float, default=None)
    parser.add_argument("--viscous-length-scale", type=float, default=None)
    parser.add_argument("--viscous-neighbor-threshold", type=float, default=None)
    parser.add_argument("--viscous-neighbor-penalty", type=float, default=None)
    parser.add_argument("--companion-epsilon", type=float, default=None)
    parser.add_argument("--companion-method", default=None)
    parser.add_argument("--companion-epsilon-clone", type=float, default=None)
    parser.add_argument("--lambda-alg", type=float, default=None)
    parser.add_argument("--epsilon-clone", type=float, default=None)
    parser.add_argument("--p-max", type=float, default=None)
    parser.add_argument("--use-fitness-force", action="store_true", default=False)
    parser.add_argument("--use-potential-force", action="store_true", default=False)
    parser.add_argument("--use-velocity-squashing", action="store_true", default=False)
    parser.add_argument("--v-alg", type=float, default=None)
    parser.add_argument("--beta-curl", type=float, default=None)
    parser.add_argument(
        "--curl-mode",
        default="none",
        choices=["none", "constant", "radial", "plane"],
        help="Curl field mode (boris rotation).",
    )
    parser.add_argument(
        "--curl-strength",
        type=float,
        default=1.0,
        help="Curl field strength (scaled by beta_curl).",
    )
    parser.add_argument("--use-anisotropic-diffusion", action="store_true", default=False)
    parser.add_argument("--use-diagonal-diffusion", action="store_true", default=False)
    parser.add_argument(
        "--diffusion-mode",
        default=None,
        choices=["hessian", "grad_proxy"],
        help="Anisotropic diffusion mode (Hessian or gradient-proxy).",
    )
    parser.add_argument(
        "--diffusion-grad-scale",
        type=float,
        default=None,
        help="Scale factor for gradient-proxy diffusion.",
    )
    parser.add_argument("--fitness-sigma-min", type=float, default=None)
    parser.add_argument(
        "--fitness-grad-mode",
        default=None,
        choices=["exact", "sum"],
        help="Fitness gradient mode (exact per-walker or summed backward).",
    )
    parser.add_argument(
        "--fitness-detach-stats",
        action="store_true",
        default=False,
        help="Detach mean/std statistics in fitness gradients.",
    )
    parser.add_argument(
        "--fitness-detach-companions",
        action="store_true",
        default=False,
        help="Detach companions in fitness gradients to avoid cross terms.",
    )

    # Channels-tab analysis settings
    parser.add_argument(
        "--channels",
        default="scalar,pseudoscalar,vector,nucleon,glueball",
        help="Comma-separated channel list for correlator extraction.",
    )
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--max-lag", type=int, default=80)
    parser.add_argument("--h-eff", type=float, default=1.0)
    parser.add_argument("--channel-mass", type=float, default=1.0)
    parser.add_argument("--ell0", type=float, default=None)
    parser.add_argument("--knn-k", type=int, default=4)
    parser.add_argument("--knn-sample", type=int, default=512)
    parser.add_argument("--window-widths", default="5-50")
    parser.add_argument("--use-connected", action="store_true", default=True)
    parser.add_argument("--no-connected", action="store_false", dest="use_connected")
    parser.add_argument("--use-nucleon-abs", action="store_true", default=False)
    parser.add_argument("--use-nucleon-abs2", action="store_true", default=False)
    parser.add_argument("--use-vector-abs2", action="store_true", default=False)
    parser.add_argument("--fit-mode", default="aic", choices=["aic", "linear", "linear_abs"])
    parser.add_argument("--fit-start", type=int, default=2)
    parser.add_argument("--fit-stop", type=int, default=None)
    parser.add_argument("--min-fit-points", type=int, default=2)
    parser.add_argument("--nucleon-fit-mode", default=None, choices=["aic", "linear", "linear_abs"])
    parser.add_argument("--nucleon-fit-start", type=int, default=None)
    parser.add_argument("--nucleon-fit-stop", type=int, default=None)
    parser.add_argument("--nucleon-min-fit-points", type=int, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.p_max is not None and not (0.0 < args.p_max <= 1.0):
        raise ValueError("--p-max must be in (0, 1].")

    # Constant zero reward by default (matches "Constant" benchmark in dashboard).
    potential_cfg = PotentialWellConfig(
        dims=args.dims,
        alpha=args.alpha,
        bounds_extent=args.bounds_extent,

    )
    # QFT dashboard defaults (GasConfigPanel.create_qft_config) + viscosity-only forces.
    operator_cfg = OperatorConfig()
    operator_cfg.gamma = 1.0
    operator_cfg.beta = 1.0
    operator_cfg.delta_t = 0.1005
    operator_cfg.epsilon_F = 38.6373
    operator_cfg.use_fitness_force = args.use_fitness_force
    operator_cfg.use_potential_force = args.use_potential_force
    operator_cfg.use_anisotropic_diffusion = args.use_anisotropic_diffusion
    operator_cfg.diagonal_diffusion = args.use_diagonal_diffusion
    operator_cfg.nu = 1.10
    operator_cfg.use_viscous_coupling = True
    operator_cfg.viscous_length_scale = 0.251372
    operator_cfg.viscous_neighbor_mode = "all"
    operator_cfg.viscous_neighbor_threshold = 0.75
    operator_cfg.viscous_neighbor_penalty = 1.1

    operator_cfg.companion_method = "uniform"
    if args.companion_method is not None:
        operator_cfg.companion_method = args.companion_method
    operator_cfg.companion_epsilon = 2.80
    operator_cfg.companion_epsilon_clone = 1.68419
    operator_cfg.lambda_alg = 1.0
    operator_cfg.exclude_self = True

    operator_cfg.p_max = 1.0
    operator_cfg.epsilon_clone = 0.01
    operator_cfg.sigma_x = 0.1
    operator_cfg.alpha_restitution = 0.5
    operator_cfg.use_velocity_squashing = args.use_velocity_squashing

    operator_cfg.fitness_alpha = 1.0
    operator_cfg.fitness_beta = 1.0
    operator_cfg.fitness_eta = 0.1
    operator_cfg.fitness_A = 2.0
    operator_cfg.fitness_sigma_min = 1e-8
    operator_cfg.fitness_epsilon_dist = 1e-8
    operator_cfg.fitness_rho = 0.251372
    if args.gamma is not None:
        operator_cfg.gamma = args.gamma
    if args.beta is not None:
        operator_cfg.beta = args.beta
    if args.delta_t is not None:
        operator_cfg.delta_t = args.delta_t
    if args.epsilon_F is not None:
        operator_cfg.epsilon_F = args.epsilon_F
    if args.diffusion_mode is not None:
        operator_cfg.diffusion_mode = args.diffusion_mode
    if args.diffusion_grad_scale is not None:
        operator_cfg.diffusion_grad_scale = args.diffusion_grad_scale
    if args.epsilon_sigma is not None:
        operator_cfg.epsilon_Sigma = args.epsilon_sigma
    if args.nu is not None:
        operator_cfg.nu = args.nu
    if args.viscous_length_scale is not None:
        operator_cfg.viscous_length_scale = args.viscous_length_scale
    if args.viscous_neighbor_threshold is not None:
        operator_cfg.viscous_neighbor_threshold = args.viscous_neighbor_threshold
    if args.viscous_neighbor_penalty is not None:
        operator_cfg.viscous_neighbor_penalty = args.viscous_neighbor_penalty
    if args.companion_epsilon is not None:
        operator_cfg.companion_epsilon = args.companion_epsilon
    if args.companion_epsilon_clone is not None:
        operator_cfg.companion_epsilon_clone = args.companion_epsilon_clone
    if args.lambda_alg is not None:
        operator_cfg.lambda_alg = args.lambda_alg
    if args.epsilon_clone is not None:
        operator_cfg.epsilon_clone = args.epsilon_clone
    if args.p_max is not None:
        operator_cfg.p_max = args.p_max
    if args.v_alg is not None:
        operator_cfg.V_alg = args.v_alg
    if args.beta_curl is not None:
        operator_cfg.beta_curl = args.beta_curl
    if args.fitness_sigma_min is not None:
        operator_cfg.fitness_sigma_min = args.fitness_sigma_min
    if args.fitness_grad_mode is not None:
        operator_cfg.fitness_grad_mode = args.fitness_grad_mode
    operator_cfg.fitness_detach_stats = args.fitness_detach_stats
    operator_cfg.fitness_detach_companions = args.fitness_detach_companions

    run_cfg = RunConfig(
        N=args.N,
        n_steps=args.n_steps,
        record_every=args.record_every,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,

    )
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reward_1form = None
    if args.zero_reward:
        def reward_1form(x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

    curl_mode = args.curl_mode
    if operator_cfg.beta_curl > 0 and curl_mode == "none":
        curl_mode = "constant"
    curl_field = _build_curl_field(args.dims, curl_mode, args.curl_strength)

    history, _ = run_simulation(
        potential_cfg,
        operator_cfg,
        run_cfg,
        show_progress=not args.no_progress,
        reward_1form=reward_1form,
        curl_field=curl_field,
    )
    paths = save_outputs(history, output_dir, run_id, potential_cfg, operator_cfg, run_cfg)

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    channel_cfg, channel_kwargs = _build_channel_config(args)
    results = compute_all_channels(history, channels=channels, config=channel_cfg)

    nucleon_override: dict[str, Any] = {}
    if args.nucleon_fit_mode is not None:
        nucleon_override["fit_mode"] = args.nucleon_fit_mode
    if args.nucleon_fit_start is not None:
        nucleon_override["fit_start"] = args.nucleon_fit_start
    if args.nucleon_fit_stop is not None:
        nucleon_override["fit_stop"] = args.nucleon_fit_stop
    if args.nucleon_min_fit_points is not None:
        nucleon_override["min_fit_points"] = args.nucleon_min_fit_points

    nucleon_cfg = channel_cfg
    if nucleon_override:
        nucleon_cfg = replace(channel_cfg, **nucleon_override)
        if "nucleon" in results:
            nucleon_channel = get_channel_class("nucleon")(history, nucleon_cfg)
            results["nucleon"] = nucleon_channel.compute()

    if args.use_nucleon_abs:
        results["nucleon_abs"] = _compute_channel_transform(history, nucleon_cfg, "nucleon", "abs")
    if args.use_nucleon_abs2:
        results["nucleon_abs2"] = _compute_channel_transform(history, nucleon_cfg, "nucleon", "abs2")
    if args.use_vector_abs2:
        results["vector_abs2"] = _compute_channel_transform(history, channel_cfg, "vector", "abs2")

    full_json_path, arrays_path = _save_full_analysis(results, output_dir, run_id)
    masses, r2s, summaries = _extract_masses(results)
    ratios = _compute_ratios(
        masses,
        use_nucleon_abs=args.use_nucleon_abs or args.use_nucleon_abs2,
        use_vector_abs2=args.use_vector_abs2,
    )

    baseline = {
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "paths": {
            "history": str(paths["history"]),
            "summary": str(paths["summary"]),
            "config": str(paths["metadata"]),
            "channels_full": str(full_json_path),
            "channels_arrays": str(arrays_path) if arrays_path is not None else None,
        },
        "configs": {
            "potential": asdict(potential_cfg),
            "operators": asdict(operator_cfg),
            "run": asdict(run_cfg),
            "reward_mode": "zero" if args.zero_reward else "potential",
            "curl_mode": curl_mode,
            "curl_strength": args.curl_strength if curl_mode != "none" else None,
        },
        "channels": {
            "list": list(results.keys()),
            "settings": channel_kwargs,
            "nucleon_fit_override": nucleon_override if nucleon_override else None,
            "masses": masses,
            "r_squared": r2s,
            "summaries": summaries,
            "ratios": ratios,
            "target_ratios": TARGET_RATIOS,
        },
    }

    baseline_path = output_dir / f"{run_id}_baseline.json"
    _write_json(baseline_path, baseline)

    print("Baseline run complete.")
    print(f"History: {paths['history']}")
    print(f"Channels (full): {full_json_path}")
    if arrays_path is not None:
        print(f"Channels arrays: {arrays_path}")
    print(f"Baseline JSON: {baseline_path}")


if __name__ == "__main__":
    main()
