"""
Calibrate Fractal Gas QFT parameters from measured Standard Model constants.

This script converts measured couplings (alpha_em, sin^2(theta_W), alpha_s)
into algorithmic parameters using the Volume 3 Fractal Set formulas. QSD
normalizations must be supplied from simulations (or left as order-one defaults).

Usage:
    python src/experiments/calibrate_fractal_gas_qft.py
    python src/experiments/calibrate_fractal_gas_qft.py --m-gev 91.1876 --d 3
    python src/experiments/calibrate_fractal_gas_qft.py --history-path outputs/.../run_history.pt
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from math import pi, sqrt
from pathlib import Path
from typing import Any

import torch

from fragile.fractalai.core.companion_selection import compute_algorithmic_distance_matrix
from fragile.fractalai.core.distance import compute_periodic_distance_matrix
from fragile.fractalai.core.history import RunHistory

try:
    from constants_check import alpha as DEFAULT_ALPHA_EM
    from constants_check import alpha_s_MZ as DEFAULT_ALPHA_S
    from constants_check import sin2_theta_W as DEFAULT_SIN2_THETA_W
except ImportError:
    DEFAULT_ALPHA_EM = 1 / 137.035999084
    DEFAULT_ALPHA_S = 0.1179
    DEFAULT_SIN2_THETA_W = 0.23121


@dataclass
class MeasuredConstants:
    alpha_em: float = DEFAULT_ALPHA_EM
    sin2_theta_w: float = DEFAULT_SIN2_THETA_W
    alpha_s: float = DEFAULT_ALPHA_S
    scale_label: str = "M_Z"


@dataclass
class CalibrationInputs:
    d: int = 3
    hbar_eff: float = 1.0
    m_gev: float = 1.0
    qsd_n1: float = 1.0
    qsd_kvisc2: float = 1.0
    lambda_gap: float | None = None
    gamma: float | None = None


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


def _fmt(value: float | None, precision: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}g}"


def casimir_su(n: int) -> float:
    if n <= 1:
        raise ValueError("Casimir requires n > 1.")
    return (n**2 - 1) / (2 * n)


def gauge_couplings(
    alpha_em: float, sin2_theta_w: float, alpha_s: float
) -> dict[str, float]:
    if alpha_em <= 0:
        raise ValueError("alpha_em must be positive.")
    if not 0.0 < sin2_theta_w < 1.0:
        raise ValueError("sin2_theta_w must be in (0, 1).")
    if alpha_s <= 0:
        raise ValueError("alpha_s must be positive.")

    e_em = sqrt(4.0 * pi * alpha_em)
    sin_theta = sqrt(sin2_theta_w)
    cos_theta = sqrt(1.0 - sin2_theta_w)
    g2 = e_em / sin_theta
    g1 = e_em / cos_theta
    g3 = sqrt(4.0 * pi * alpha_s)
    return {
        "e_em": e_em,
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "sin_theta_w": sin_theta,
        "cos_theta_w": cos_theta,
    }


def epsilon_c_from_g2(hbar_eff: float, g2: float, d: int) -> float:
    if hbar_eff <= 0:
        raise ValueError("hbar_eff must be positive.")
    if g2 <= 0:
        raise ValueError("g2 must be positive.")
    c2_d = casimir_su(d)
    c2_2 = casimir_su(2)
    return sqrt(2.0 * hbar_eff * c2_2 / (c2_d * g2**2))


def epsilon_d_from_g1(hbar_eff: float, g1: float, qsd_n1: float) -> float:
    if hbar_eff <= 0:
        raise ValueError("hbar_eff must be positive.")
    if g1 <= 0:
        raise ValueError("g1 must be positive.")
    if qsd_n1 <= 0:
        raise ValueError("qsd_n1 must be positive.")
    return sqrt(hbar_eff * qsd_n1 / (g1**2))


def epsilon_f_from_em(m_gev: float, e_em: float) -> float:
    if m_gev <= 0:
        raise ValueError("m_gev must be positive.")
    if e_em <= 0:
        raise ValueError("e_em must be positive.")
    return m_gev / (e_em**2)


def nu_from_g3(hbar_eff: float, g3: float, d: int, qsd_kvisc2: float) -> float:
    if hbar_eff <= 0:
        raise ValueError("hbar_eff must be positive.")
    if g3 <= 0:
        raise ValueError("g3 must be positive.")
    if qsd_kvisc2 <= 0:
        raise ValueError("qsd_kvisc2 must be positive.")
    dim_factor = d * (d**2 - 1) / 12.0
    return hbar_eff * g3 / sqrt(dim_factor * qsd_kvisc2)


def tau_from_hbar(m_gev: float, epsilon_c: float, hbar_eff: float) -> float:
    if m_gev <= 0:
        raise ValueError("m_gev must be positive.")
    if epsilon_c <= 0:
        raise ValueError("epsilon_c must be positive.")
    if hbar_eff <= 0:
        raise ValueError("hbar_eff must be positive.")
    return m_gev * (epsilon_c**2) / (2.0 * hbar_eff)


def rho_from_g2(m_gev: float, g2: float, hbar_eff: float) -> float:
    if m_gev <= 0:
        raise ValueError("m_gev must be positive.")
    if g2 <= 0:
        raise ValueError("g2 must be positive.")
    if hbar_eff <= 0:
        raise ValueError("hbar_eff must be positive.")
    return sqrt(2.0 * hbar_eff) * g2 / m_gev


def mass_scales(
    epsilon_c: float,
    rho: float,
    hbar_eff: float,
    lambda_gap: float | None,
    gamma: float | None,
) -> dict[str, float | None]:
    return {
        "m_clone": 1.0 / epsilon_c if epsilon_c > 0 else None,
        "m_mf": 1.0 / rho if rho > 0 else None,
        "m_gap": hbar_eff * lambda_gap if lambda_gap is not None else None,
        "m_friction": gamma,
    }


def dimensionless_ratios(
    epsilon_c: float,
    rho: float,
    hbar_eff: float,
    lambda_gap: float | None,
    tau: float | None,
) -> dict[str, float | None]:
    sigma_sep = epsilon_c / rho if rho > 0 else None
    eta_time = tau * lambda_gap if (lambda_gap is not None and tau is not None) else None
    kappa = None
    if lambda_gap is not None and rho > 0:
        kappa = 1.0 / (rho * hbar_eff * lambda_gap)
    return {"sigma_sep": sigma_sep, "eta_time": eta_time, "kappa": kappa}


def _history_param(history: RunHistory, keys: list[str], default: float) -> float:
    params = history.params or {}
    value: Any = params
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _select_qsd_indices(
    history: RunHistory,
    warmup_frac: float,
    start_step: int | None,
    stride: int,
    max_samples: int | None,
) -> list[int]:
    if history.n_recorded <= 1:
        return []

    indices = list(range(1, history.n_recorded))
    if start_step is not None:
        for idx, step in enumerate(history.recorded_steps):
            if step >= start_step and idx > 0:
                indices = list(range(idx, history.n_recorded))
                break
    elif warmup_frac > 0:
        warmup_count = int(len(indices) * warmup_frac)
        indices = indices[warmup_count:]

    if stride > 1:
        indices = indices[::stride]

    if max_samples is not None and max_samples > 0:
        indices = indices[:max_samples]

    return indices


def compute_qsd_n1(
    history: RunHistory,
    epsilon_d: float,
    lambda_alg: float,
    indices: list[int],
) -> float | None:
    if epsilon_d <= 0 or not indices:
        return None
    if history.pbc and history.bounds is None:
        raise ValueError("history.bounds required when pbc=True")

    total = 0.0
    n_samples = 0
    bounds = history.bounds
    pbc = history.pbc
    device = history.x_before_clone.device
    eye = torch.eye(history.N, device=device, dtype=torch.bool)

    for t_idx in indices:
        info_idx = t_idx - 1
        alive = history.alive_mask[info_idx]
        n_alive = int(alive.sum().item())
        if n_alive <= 1:
            continue
        x = history.x_before_clone[t_idx]
        v = history.v_before_clone[t_idx]
        dist_sq = compute_algorithmic_distance_matrix(x, v, lambda_alg, bounds, pbc)
        kernel = torch.exp(-dist_sq / (epsilon_d**2))
        alive_mask = alive.unsqueeze(0) & alive.unsqueeze(1)
        kernel = torch.where(alive_mask & ~eye, kernel, torch.zeros_like(kernel))
        denom = n_alive * (n_alive - 1)
        total += kernel.sum().item() / denom
        n_samples += 1

    if n_samples == 0:
        return None
    return total / n_samples


def compute_qsd_kvisc2(
    history: RunHistory,
    length_scale: float,
    indices: list[int],
) -> float | None:
    if length_scale <= 0 or not indices:
        return None
    if history.pbc and history.bounds is None:
        raise ValueError("history.bounds required when pbc=True")

    total = 0.0
    n_samples = 0
    bounds = history.bounds
    pbc = history.pbc
    device = history.x_before_clone.device
    l_sq = length_scale**2
    eye = torch.eye(history.N, device=device, dtype=torch.bool)

    for t_idx in indices:
        info_idx = t_idx - 1
        alive = history.alive_mask[info_idx]
        n_alive = int(alive.sum().item())
        if n_alive <= 1:
            continue
        x = history.x_before_clone[t_idx]
        dist = compute_periodic_distance_matrix(x, bounds=bounds, pbc=pbc)
        kernel = torch.exp(-(dist**2) / (2.0 * l_sq))
        alive_mask = alive.unsqueeze(0) & alive.unsqueeze(1)
        kernel = torch.where(alive_mask & ~eye, kernel, torch.zeros_like(kernel))
        denom = n_alive * (n_alive - 1)
        total += (kernel**2).sum().item() / denom
        n_samples += 1

    if n_samples == 0:
        return None
    return total / n_samples


def compute_stability_metrics(history: RunHistory) -> dict[str, Any]:
    finite_x = torch.isfinite(history.x_before_clone).all().item()
    finite_v = torch.isfinite(history.v_before_clone).all().item()
    finite_fit = torch.isfinite(history.fitness).all().item()
    n_alive = history.n_alive.cpu().numpy()
    min_alive = int(n_alive.min()) if n_alive.size else 0
    final_alive = int(n_alive[-1]) if n_alive.size else 0
    alive_fraction = float(final_alive / history.N) if history.N > 0 else 0.0

    bounds = history.bounds
    out_of_bounds_frac = None
    if bounds is not None and not history.pbc and history.n_recorded > 1:
        last_x = history.x_before_clone[-1]
        inside = bounds.contains(last_x)
        out_of_bounds_frac = float((~inside).float().mean().item())

    stable = (
        not history.terminated_early
        and finite_x
        and finite_v
        and finite_fit
        and min_alive > 0
    )
    issues = []
    if history.terminated_early:
        issues.append("terminated_early")
    if not finite_x:
        issues.append("non_finite_positions")
    if not finite_v:
        issues.append("non_finite_velocities")
    if not finite_fit:
        issues.append("non_finite_fitness")
    if min_alive <= 0:
        issues.append("no_alive_walkers")

    return {
        "stable": stable,
        "issues": issues,
        "finite_positions": bool(finite_x),
        "finite_velocities": bool(finite_v),
        "finite_fitness": bool(finite_fit),
        "min_alive": min_alive,
        "final_alive": final_alive,
        "alive_fraction": alive_fraction,
        "out_of_bounds_fraction": out_of_bounds_frac,
    }


def build_summary(data: dict[str, Any]) -> str:
    couplings = data["couplings"]
    alg = data["algorithmic_parameters"]
    derived = data["derived"]
    ratios = data["ratios"]

    lines = [
        "Fractal Gas QFT Calibration Summary",
        "",
        f"Scale: {data['inputs']['constants']['scale_label']}",
        f"alpha_em = {_fmt(data['inputs']['constants']['alpha_em'])}",
        f"sin2_theta_w = {_fmt(data['inputs']['constants']['sin2_theta_w'])}",
        f"alpha_s = {_fmt(data['inputs']['constants']['alpha_s'])}",
        "",
        "Couplings:",
        f"  e_em = {_fmt(couplings['e_em'])}",
        f"  g1 = {_fmt(couplings['g1'])}",
        f"  g2 = {_fmt(couplings['g2'])}",
        f"  g3 = {_fmt(couplings['g3'])}",
        "",
        "Algorithmic parameters:",
        f"  epsilon_c = {_fmt(alg['epsilon_c'])}",
        f"  epsilon_d = {_fmt(alg['epsilon_d'])}",
        f"  rho = {_fmt(alg['rho'])}",
        f"  tau = {_fmt(alg['tau'])}",
        f"  epsilon_F = {_fmt(alg['epsilon_F'])}",
        f"  nu = {_fmt(alg['nu'])}",
        "",
        "Derived scales:",
        f"  m_clone = {_fmt(derived['m_clone'])}",
        f"  m_mf = {_fmt(derived['m_mf'])}",
        f"  m_gap = {_fmt(derived['m_gap'])}",
        "",
        "Dimensionless ratios:",
        f"  sigma_sep = {_fmt(ratios['sigma_sep'])}",
        f"  eta_time = {_fmt(ratios['eta_time'])}",
        f"  kappa = {_fmt(ratios['kappa'])}",
    ]
    if data.get("qsd_estimates") is not None:
        qsd = data["qsd_estimates"]
        lines.extend(
            [
                "",
                "QSD estimates:",
                f"  qsd_n1 = {_fmt(qsd.get('qsd_n1'))}",
                f"  qsd_kvisc2 = {_fmt(qsd.get('qsd_kvisc2'))}",
                f"  samples = {qsd.get('samples')}",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "QSD inputs:",
                f"  qsd_n1 = {_fmt(data['inputs']['calibration']['qsd_n1'])}",
                f"  qsd_kvisc2 = {_fmt(data['inputs']['calibration']['qsd_kvisc2'])}",
            ]
        )
    if data.get("stability") is not None:
        stability = data["stability"]
        lines.extend(
            [
                "",
                "Stability checks:",
                f"  stable = {stability.get('stable')}",
                f"  issues = {', '.join(stability.get('issues', [])) or 'none'}",
                f"  min_alive = {stability.get('min_alive')}",
                f"  alive_fraction = {_fmt(stability.get('alive_fraction'), 4)}",
            ]
        )
    lines.extend(
        [
            "",
            "Notes:",
            "  - qsd_n1 and qsd_kvisc2 must be measured from QSD statistics.",
            "  - epsilon_d depends on qsd_n1, which depends on epsilon_d itself.",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate Fractal Gas QFT parameters from measured constants."
    )
    parser.add_argument("--alpha-em", type=float, default=DEFAULT_ALPHA_EM)
    parser.add_argument("--sin2-theta-w", type=float, default=DEFAULT_SIN2_THETA_W)
    parser.add_argument("--alpha-s", type=float, default=DEFAULT_ALPHA_S)
    parser.add_argument("--scale-label", type=str, default="M_Z")
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--hbar-eff", type=float, default=1.0)
    parser.add_argument("--m-gev", type=float, default=1.0)
    parser.add_argument("--qsd-n1", type=float, default=1.0)
    parser.add_argument("--qsd-kvisc2", type=float, default=1.0)
    parser.add_argument("--lambda-gap", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/qft_calibration")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--history-path", type=str, default=None)
    parser.add_argument("--qsd-warmup-frac", type=float, default=0.1)
    parser.add_argument("--qsd-start-step", type=int, default=None)
    parser.add_argument("--qsd-sample-stride", type=int, default=1)
    parser.add_argument("--qsd-max-samples", type=int, default=None)
    parser.add_argument("--qsd-iter", type=int, default=4)
    parser.add_argument("--qsd-rtol", type=float, default=1e-3)
    parser.add_argument("--no-history-qsd", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    constants = MeasuredConstants(
        alpha_em=args.alpha_em,
        sin2_theta_w=args.sin2_theta_w,
        alpha_s=args.alpha_s,
        scale_label=args.scale_label,
    )

    inputs = CalibrationInputs(
        d=args.d,
        hbar_eff=args.hbar_eff,
        m_gev=args.m_gev,
        qsd_n1=args.qsd_n1,
        qsd_kvisc2=args.qsd_kvisc2,
        lambda_gap=args.lambda_gap,
        gamma=args.gamma,
    )

    couplings = gauge_couplings(constants.alpha_em, constants.sin2_theta_w, constants.alpha_s)

    history = None
    qsd_estimates = None
    stability = None
    qsd_n1 = inputs.qsd_n1
    qsd_kvisc2 = inputs.qsd_kvisc2

    if args.history_path:
        history = RunHistory.load(args.history_path)
        stability = compute_stability_metrics(history)

        if not args.no_history_qsd:
            lambda_alg = _history_param(history, ["companion_selection", "lambda_alg"], 0.0)
            viscous_scale = _history_param(history, ["kinetic", "viscous_length_scale"], 1.0)
            indices = _select_qsd_indices(
                history,
                warmup_frac=max(0.0, args.qsd_warmup_frac),
                start_step=args.qsd_start_step,
                stride=max(1, args.qsd_sample_stride),
                max_samples=args.qsd_max_samples,
            )

            qsd_n1_iter = qsd_n1
            for _ in range(max(1, args.qsd_iter)):
                epsilon_d_iter = epsilon_d_from_g1(
                    inputs.hbar_eff, couplings["g1"], qsd_n1_iter
                )
                qsd_n1_new = compute_qsd_n1(history, epsilon_d_iter, lambda_alg, indices)
                if qsd_n1_new is None:
                    break
                if abs(qsd_n1_new - qsd_n1_iter) <= args.qsd_rtol * max(
                    qsd_n1_iter, 1e-12
                ):
                    qsd_n1_iter = qsd_n1_new
                    break
                qsd_n1_iter = qsd_n1_new

            qsd_n1 = qsd_n1_iter
            qsd_kvisc2_est = compute_qsd_kvisc2(history, viscous_scale, indices)
            if qsd_kvisc2_est is not None:
                qsd_kvisc2 = qsd_kvisc2_est

            qsd_estimates = {
                "qsd_n1": qsd_n1,
                "qsd_kvisc2": qsd_kvisc2,
                "samples": len(indices),
                "lambda_alg": lambda_alg,
                "viscous_length_scale": viscous_scale,
            }

    epsilon_c = epsilon_c_from_g2(inputs.hbar_eff, couplings["g2"], inputs.d)
    epsilon_d = epsilon_d_from_g1(inputs.hbar_eff, couplings["g1"], qsd_n1)
    tau = tau_from_hbar(inputs.m_gev, epsilon_c, inputs.hbar_eff)
    rho = rho_from_g2(inputs.m_gev, couplings["g2"], inputs.hbar_eff)
    epsilon_f = epsilon_f_from_em(inputs.m_gev, couplings["e_em"])
    nu = nu_from_g3(inputs.hbar_eff, couplings["g3"], inputs.d, qsd_kvisc2)

    derived = mass_scales(epsilon_c, rho, inputs.hbar_eff, inputs.lambda_gap, inputs.gamma)
    ratios = dimensionless_ratios(epsilon_c, rho, inputs.hbar_eff, inputs.lambda_gap, tau)

    result = {
        "inputs": {
            "constants": asdict(constants),
            "calibration": asdict(inputs),
        },
        "qsd_estimates": qsd_estimates,
        "qsd_used": {
            "qsd_n1": qsd_n1,
            "qsd_kvisc2": qsd_kvisc2,
        },
        "stability": stability,
        "couplings": couplings,
        "algorithmic_parameters": {
            "epsilon_c": epsilon_c,
            "epsilon_d": epsilon_d,
            "rho": rho,
            "tau": tau,
            "epsilon_F": epsilon_f,
            "nu": nu,
        },
        "derived": derived,
        "ratios": ratios,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{run_id}_calibration.json"
    summary_path = output_dir / f"{run_id}_summary.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(result), f, indent=2, sort_keys=True)

    summary = build_summary(result)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(summary)

    print(summary)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
