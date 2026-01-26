"""
Analyze Fractal Gas RunHistory for QFT observables and theory diagnostics.

This script loads a saved RunHistory (from fractal_gas_potential_well.py),
computes gauge/field observables, Lyapunov diagnostics, QSD variance metrics,
and optional FractalSet curvature summaries.

Usage:
    python src/experiments/analyze_fractal_gas_qft.py --history-path outputs/..._history.pt
    python src/experiments/analyze_fractal_gas_qft.py --build-fractal-set
    python src/experiments/analyze_fractal_gas_qft.py --compute-particles --build-fractal-set
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fragile.fractalai.core.companion_selection import compute_algorithmic_distance_matrix
from fragile.fractalai.core.fitness import compute_fitness
from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.lyapunov import compute_lyapunov_components_trajectory
from fragile.fractalai.qft.particle_observables import (
    compute_baryon_operator,
    compute_baryon_operator_knn,
    compute_color_state,
    compute_companion_distance,
    compute_effective_mass,
    compute_knn_indices,
    compute_meson_operator,
    compute_meson_operator_knn,
    compute_time_correlator,
    fit_mass_exponential,
    select_mass_plateau,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
    MPL_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    HAS_MPL = False
    MPL_ERROR = str(exc)

try:
    from fragile.fractalai.core.fractal_set import FractalSet
    from fragile.fractalai.geometry.curvature import (
        check_cheeger_consistency,
        compute_ricci_from_fractal_set_graph,
        compute_ricci_from_fractal_set_hessian,
    )

    HAS_FRACTAL_SET = True
    FRACTAL_SET_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    HAS_FRACTAL_SET = False
    FRACTAL_SET_ERROR = str(exc)


@dataclass
class AnalysisConfig:
    analysis_time_index: int | None = None
    analysis_step: int | None = None
    warmup_fraction: float = 0.1
    h_eff: float = 1.0
    correlation_r_max: float = 0.5
    correlation_bins: int = 50
    gradient_neighbors: int = 5
    build_fractal_set: bool = False
    fractal_set_stride: int = 10
    # New options for proper local field analysis
    use_local_fields: bool = False
    use_connected: bool = False
    density_sigma: float = 0.5
    compute_particles: bool = False
    particle_operators: tuple[str, ...] = ("baryon", "meson", "glueball")
    particle_max_lag: int | None = 80
    particle_fit_start: int = 7
    particle_fit_stop: int = 16
    particle_fit_mode: str = "window"
    particle_plateau_min_points: int = 3
    particle_plateau_max_points: int | None = None
    particle_plateau_max_cv: float | None = 0.2
    particle_mass: float = 1.0
    particle_ell0: float | None = None
    particle_use_connected: bool = True
    particle_neighbor_method: str = "knn"
    particle_knn_k: int = 4
    particle_knn_sample: int | None = 512
    particle_meson_reduce: str = "mean"
    particle_baryon_pairs: int | None = None
    compute_string_tension: bool = False
    string_tension_max_triangles: int = 20000
    string_tension_bins: int = 20


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.ndarray, torch.Tensor)):
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


def _write_progress(
    progress_path: Path,
    stage: str,
    payload: dict[str, Any],
    state: str = "in_progress",
) -> None:
    data: dict[str, Any] = {}
    if progress_path.exists():
        try:
            data = json.loads(progress_path.read_text())
        except json.JSONDecodeError:
            data = {}
    data.update(payload)
    data["status"] = {
        "state": state,
        "stage": stage,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    progress_path.write_text(json.dumps(_json_safe(data), indent=2, sort_keys=True))


def _get_param(params: dict[str, Any] | None, keys: list[str], default: Any) -> Any:
    current = params or {}
    for key in keys:
        if not isinstance(current, dict) or key not in current or current[key] is None:
            return default
        current = current[key]
    return current


def _find_latest_history(output_dir: Path) -> Path:
    candidates = list(output_dir.glob("*_history.pt"))
    if not candidates:
        raise FileNotFoundError(f"No history files found in {output_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _select_time_index(history: RunHistory, cfg: AnalysisConfig) -> int:
    if cfg.analysis_step is not None:
        return history.get_step_index(cfg.analysis_step)
    if cfg.analysis_time_index is not None:
        return cfg.analysis_time_index
    return history.n_recorded - 1


def _downsample_history(history: RunHistory, stride: int) -> RunHistory:
    if stride <= 1 or history.n_recorded <= 1:
        return history

    indices = list(range(0, history.n_recorded, stride))
    if indices[-1] != history.n_recorded - 1:
        indices.append(history.n_recorded - 1)

    info_indices = [idx - 1 for idx in indices if idx > 0]

    data = history.to_dict()
    for key, value in data.items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.shape[0] == history.n_recorded:
            data[key] = value[indices]
        elif value.shape[0] == history.n_recorded - 1:
            data[key] = value[info_indices]

    data["n_recorded"] = len(indices)
    data["record_every"] = history.record_every * stride
    data["recorded_steps"] = [history.recorded_steps[i] for i in indices]

    return RunHistory(**data)


def _bin_by_distance(
    positions: torch.Tensor,
    values: torch.Tensor,
    alive: torch.Tensor,
    r_max: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos_diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    distances = torch.sqrt((pos_diff**2).sum(dim=-1))
    products = values.unsqueeze(1) * values.unsqueeze(0)

    mask = (
        alive.unsqueeze(1)
        & alive.unsqueeze(0)
        & (torch.eye(len(alive), device=alive.device, dtype=torch.bool).logical_not())
        & (distances <= r_max)
    )

    distances_valid = distances[mask].cpu().numpy()
    products_valid = products[mask].cpu().numpy()

    bin_edges = np.linspace(0, r_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    binned_correlations = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        in_bin = (distances_valid >= bin_edges[i]) & (distances_valid < bin_edges[i + 1])
        if in_bin.sum() > 0:
            binned_correlations[i] = products_valid[in_bin].mean()
            bin_counts[i] = in_bin.sum()

    return bin_centers, binned_correlations, bin_counts


def _bin_by_distance_connected(
    positions: torch.Tensor,
    values: torch.Tensor,
    alive: torch.Tensor,
    r_max: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute connected two-point correlator G(r) = <φφ> - <φ>².

    The connected correlator subtracts the mean to measure pure fluctuations,
    giving a cleaner exponential decay signal for QFT-like systems.
    """
    # Subtract mean to get fluctuations (connected correlator)
    mean_val = values[alive].mean()
    fluctuations = values - mean_val

    pos_diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    distances = torch.sqrt((pos_diff**2).sum(dim=-1))
    products = fluctuations.unsqueeze(1) * fluctuations.unsqueeze(0)

    mask = (
        alive.unsqueeze(1)
        & alive.unsqueeze(0)
        & (torch.eye(len(alive), device=alive.device, dtype=torch.bool).logical_not())
        & (distances <= r_max)
    )

    distances_valid = distances[mask].cpu().numpy()
    products_valid = products[mask].cpu().numpy()

    bin_edges = np.linspace(0, r_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    binned_correlations = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        in_bin = (distances_valid >= bin_edges[i]) & (distances_valid < bin_edges[i + 1])
        if in_bin.sum() > 0:
            binned_correlations[i] = products_valid[in_bin].mean()
            bin_counts[i] = in_bin.sum()

    return bin_centers, binned_correlations, bin_counts


def _compute_local_fields(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    rewards: torch.Tensor,
    alive: torch.Tensor,
    sigma_density: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Compute proper LOCAL fields for QFT correlation analysis.

    These fields are deterministic functions of position, suitable for
    computing QFT-style correlation functions. Unlike d_prime which depends
    on random companion selection, these fields are purely local.

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        rewards: Raw reward values [N]
        alive: Alive mask [N]
        sigma_density: Kernel width for local density estimation

    Returns:
        Dictionary of local fields:
        - density: ρ(x) - local walker density via kernel density estimate
        - diversity_local: 1/ρ(x) - inverse density (proper local diversity)
        - radial: ||x|| - distance from origin
        - kinetic: 0.5||v||² - kinetic energy
        - reward_raw: r(x) = -U(x) - raw rewards (already local)
    """
    N = positions.shape[0]
    device = positions.device

    # 1. Local density field ρ(x_i) - kernel density estimate
    # ρ_i = Σ_j K(x_i, x_j) where K is Gaussian kernel
    dists_sq = ((positions.unsqueeze(1) - positions.unsqueeze(0)) ** 2).sum(dim=-1)
    kernel = torch.exp(-dists_sq / (2 * sigma_density**2))

    # Mask for alive walkers
    alive_2d = alive.unsqueeze(0).float() * alive.unsqueeze(1).float()
    kernel = kernel * alive_2d
    kernel.fill_diagonal_(0)  # Exclude self from density

    density = kernel.sum(dim=1)

    # Normalize density (avoid division by zero)
    alive_density = density[alive]
    if alive_density.numel() > 0 and alive_density.mean() > 0:
        density = density / alive_density.mean()
    else:
        density = torch.ones_like(density)

    # 2. Local diversity field = 1/density (inverse density)
    # High diversity = low local density = walker is "exploring"
    diversity_local = 1.0 / torch.clamp(density, min=1e-6)

    # 3. Radial distance field ||x|| (local function of position)
    radial = torch.sqrt((positions**2).sum(dim=-1))

    # 4. Kinetic energy field T(x) = 0.5||v||²
    kinetic = 0.5 * (velocities**2).sum(dim=-1)

    # 5. Raw rewards (already a local function: r = -U(x))
    # This is the cleanest QFT observable

    return {
        "density": torch.where(alive, density, torch.zeros_like(density)),
        "diversity_local": torch.where(alive, diversity_local, torch.zeros_like(diversity_local)),
        "radial": torch.where(alive, radial, torch.zeros_like(radial)),
        "kinetic": torch.where(alive, kinetic, torch.zeros_like(kinetic)),
        "reward_raw": torch.where(alive, rewards, torch.zeros_like(rewards)),
    }


def _fit_exponential_decay(
    r: np.ndarray, C: np.ndarray, counts: np.ndarray
) -> dict[str, float]:
    valid = (counts > 0) & (C > 0)
    if valid.sum() < 2:
        return {"C0": 0.0, "xi": 0.0, "r_squared": 0.0}

    r_valid = r[valid]
    C_valid = C[valid]
    weights = counts[valid]

    x = r_valid**2
    y = np.log(C_valid)

    coeffs = np.polyfit(x, y, 1, w=np.sqrt(weights))
    slope, intercept = coeffs

    if slope >= 0:
        return {"C0": float(np.exp(intercept)), "xi": 0.0, "r_squared": 0.0}

    xi = float(np.sqrt(-1.0 / slope))
    C0 = float(np.exp(intercept))

    y_pred = intercept + slope * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"C0": C0, "xi": xi, "r_squared": r_squared}


def _fit_connected_correlator(
    r: np.ndarray, C: np.ndarray, counts: np.ndarray
) -> dict[str, float]:
    """Fit connected correlator that may have negative values at large r.

    For connected correlators G(r) = <φφ> - <φ>², the correlation can go negative
    at large distances (anti-correlation). This function:
    1. Finds the zero-crossing distance r_zero
    2. Fits exponential decay only to the positive portion before zero-crossing
    3. Returns additional diagnostics about the correlation structure

    The correlation length ξ is extracted from G(r) ~ G₀ exp(-r²/ξ²) for r < r_zero.
    """
    valid = counts > 0
    if valid.sum() < 2:
        return {
            "C0": 0.0,
            "xi": 0.0,
            "r_squared": 0.0,
            "r_zero": 0.0,
            "has_zero_crossing": False,
            "n_positive": 0,
            "n_negative": 0,
        }

    r_valid = r[valid]
    C_valid = C[valid]
    counts_valid = counts[valid]

    # Count positive/negative values
    n_positive = int((C_valid > 0).sum())
    n_negative = int((C_valid < 0).sum())

    # Find zero-crossing: first index where C goes from positive to negative
    # (assuming C starts positive and decays)
    sign_changes = np.where(np.diff(np.sign(C_valid)) < 0)[0]
    has_zero_crossing = len(sign_changes) > 0

    if has_zero_crossing:
        # Interpolate to find zero-crossing distance
        idx = sign_changes[0]
        if idx + 1 < len(r_valid) and C_valid[idx] != C_valid[idx + 1]:
            # Linear interpolation
            r_zero = r_valid[idx] + (r_valid[idx + 1] - r_valid[idx]) * (
                -C_valid[idx] / (C_valid[idx + 1] - C_valid[idx])
            )
        else:
            r_zero = r_valid[idx]
        r_zero = float(r_zero)

        # Use only positive values before zero-crossing for fitting
        fit_mask = (C_valid > 0) & (r_valid < r_zero)
    else:
        r_zero = float(r_valid[-1])  # No zero crossing within measured range
        fit_mask = C_valid > 0

    # Need at least 2 points for fitting
    if fit_mask.sum() < 2:
        return {
            "C0": 0.0,
            "xi": 0.0,
            "r_squared": 0.0,
            "r_zero": r_zero,
            "has_zero_crossing": has_zero_crossing,
            "n_positive": n_positive,
            "n_negative": n_negative,
        }

    r_fit = r_valid[fit_mask]
    C_fit = C_valid[fit_mask]
    weights = counts_valid[fit_mask]

    # Fit: log(C) = log(C0) - r²/ξ²
    x = r_fit**2
    y = np.log(C_fit)

    try:
        coeffs = np.polyfit(x, y, 1, w=np.sqrt(weights))
        slope, intercept = coeffs
    except (np.linalg.LinAlgError, ValueError):
        return {
            "C0": 0.0,
            "xi": 0.0,
            "r_squared": 0.0,
            "r_zero": r_zero,
            "has_zero_crossing": has_zero_crossing,
            "n_positive": n_positive,
            "n_negative": n_negative,
        }

    if slope >= 0:
        # Correlation increasing (anti-QFT) - return zero correlation length
        return {
            "C0": float(np.exp(intercept)),
            "xi": 0.0,
            "r_squared": 0.0,
            "r_zero": r_zero,
            "has_zero_crossing": has_zero_crossing,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "slope_positive": True,
        }

    xi = float(np.sqrt(-1.0 / slope))
    C0 = float(np.exp(intercept))

    # Compute R² for the fitted region
    y_pred = intercept + slope * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "C0": C0,
        "xi": xi,
        "r_squared": r_squared,
        "r_zero": r_zero,
        "has_zero_crossing": has_zero_crossing,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_fit_points": int(fit_mask.sum()),
    }


def _compute_field_gradients(
    positions: torch.Tensor, field_values: torch.Tensor, alive: torch.Tensor, k_neighbors: int
) -> torch.Tensor:
    if positions.shape[0] < 2:
        return torch.zeros_like(field_values)
    k_neighbors = min(k_neighbors, max(1, positions.shape[0] - 1))
    distances = torch.cdist(positions, positions)
    distances_masked = distances.clone()
    distances_masked[~alive, :] = float("inf")
    distances_masked[:, ~alive] = float("inf")
    distances_masked.fill_diagonal_(float("inf"))

    _, nearest_idx = torch.topk(distances_masked, k=k_neighbors, dim=1, largest=False)
    gradients_list = []
    for k in range(k_neighbors):
        neighbor_idx = nearest_idx[:, k]
        field_diff = field_values[neighbor_idx] - field_values
        dist = distances[torch.arange(positions.shape[0]), neighbor_idx]
        gradient_magnitude = torch.abs(field_diff) / torch.clamp(dist, min=1e-10)
        gradients_list.append(gradient_magnitude)

    gradients = torch.stack(gradients_list, dim=1).mean(dim=1)
    return torch.where(alive, gradients, torch.zeros_like(gradients))


def _compute_collective_fields(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    rewards: torch.Tensor,
    alive: torch.Tensor,
    companions: torch.Tensor,
    params: dict[str, Any],
    bounds: Any,
    pbc: bool,
) -> dict[str, torch.Tensor]:
    fitness, info = compute_fitness(
        positions=positions,
        velocities=velocities,
        rewards=rewards,
        alive=alive,
        companions=companions,
        alpha=params["alpha"],
        beta=params["beta"],
        eta=params["eta"],
        lambda_alg=params["lambda_alg"],
        sigma_min=params["sigma_min"],
        A=params["A"],
        epsilon_dist=params["epsilon_dist"],
        rho=params["rho"],
        bounds=bounds,
        pbc=pbc,
    )

    return {
        "fitness": fitness,
        "d_prime": info["rescaled_distances"],
        "r_prime": info["rescaled_rewards"],
        "z_distances": info["z_distances"],
        "z_rewards": info["z_rewards"],
        "mu_distances": info["mu_distances"],
        "sigma_distances": info["sigma_distances"],
        "mu_rewards": info["mu_rewards"],
        "sigma_rewards": info["sigma_rewards"],
        "distances": info["distances"],
    }


def _compute_cloning_score(
    fitness: torch.Tensor,
    alive: torch.Tensor,
    clone_companions: torch.Tensor,
    epsilon_clone: float,
) -> torch.Tensor:
    fitness_companion = fitness[clone_companions]
    score = (fitness_companion - fitness) / (fitness + epsilon_clone)
    return torch.where(alive, score, torch.zeros_like(score))


def _compute_u1_phases(
    fitness: torch.Tensor, companions: torch.Tensor, alive: torch.Tensor, h_eff: float
) -> torch.Tensor:
    fitness_companion = fitness[companions]
    phases = -(fitness_companion - fitness) / h_eff
    return torch.where(alive, phases, torch.zeros_like(phases))


def _compute_u1_amplitude(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    alive: torch.Tensor,
    epsilon_d: float,
    lambda_alg: float,
    bounds: Any,
    pbc: bool,
) -> torch.Tensor:
    N = positions.shape[0]
    dist_sq = compute_algorithmic_distance_matrix(positions, velocities, lambda_alg, bounds, pbc)
    weights = torch.exp(-dist_sq / (2 * epsilon_d**2))

    alive_mask = alive.unsqueeze(1) & alive.unsqueeze(0)
    self_mask = ~torch.eye(N, device=alive.device, dtype=torch.bool)
    weights = weights * alive_mask.float() * self_mask.float()

    return weights / (weights.sum(dim=1, keepdim=True) + 1e-10)


def _compute_su2_phases(
    fitness: torch.Tensor,
    alive: torch.Tensor,
    clone_companions: torch.Tensor,
    epsilon_clone: float,
    h_eff: float,
) -> torch.Tensor:
    scores = _compute_cloning_score(fitness, alive, clone_companions, epsilon_clone)
    phases = scores / h_eff
    return torch.where(alive, phases, torch.zeros_like(phases))


def _compute_su2_pairing_probability(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    alive: torch.Tensor,
    epsilon_c: float,
    lambda_alg: float,
    bounds: Any,
    pbc: bool,
) -> torch.Tensor:
    N = positions.shape[0]
    dist_sq = compute_algorithmic_distance_matrix(positions, velocities, lambda_alg, bounds, pbc)
    weights = torch.exp(-dist_sq / (2 * epsilon_c**2))

    alive_mask = alive.unsqueeze(1) & alive.unsqueeze(0)
    self_mask = ~torch.eye(N, device=alive.device, dtype=torch.bool)
    weights = weights * alive_mask.float() * self_mask.float()

    return weights / (weights.sum(dim=1, keepdim=True) + 1e-10)


def _compute_dressed_state(
    phases: torch.Tensor, amplitudes: torch.Tensor, idx: int
) -> torch.Tensor:
    probs = amplitudes[idx]
    phase = phases[idx]
    return torch.sqrt(probs) * torch.exp(1j * phase)


def _compute_su2_doublet_state(
    phases: torch.Tensor, amplitudes: torch.Tensor, idx_i: int, idx_j: int
) -> torch.Tensor:
    probs_i = amplitudes[idx_i]
    probs_j = amplitudes[idx_j]
    phase_i = phases[idx_i]
    phase_j = phases[idx_j]

    up_component = torch.sqrt(probs_i) * torch.exp(1j * phase_i)
    down_component = torch.sqrt(probs_j) * torch.exp(1j * phase_j)
    return up_component + down_component


def _compute_norm_squared(state: torch.Tensor, alive: torch.Tensor) -> float:
    masked = torch.where(alive, state, torch.zeros_like(state))
    return torch.abs(torch.dot(masked.conj(), masked)).item()


def _fitness_stats(fitness: torch.Tensor, alive: torch.Tensor) -> dict[str, float]:
    fitness_alive = fitness[alive]
    if fitness_alive.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "mean": fitness_alive.mean().item(),
        "std": fitness_alive.std().item(),
        "min": fitness_alive.min().item(),
        "max": fitness_alive.max().item(),
        "median": fitness_alive.median().item(),
    }


def _edge_value(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _compute_wilson_loops(fractal_set, timestep: int) -> dict[str, Any] | None:
    tri = fractal_set.triangles
    if not tri["time_index"]:
        return None

    phases = []
    for idx, t_idx in enumerate(tri["time_index"]):
        if int(t_idx) != int(timestep):
            continue
        edge_cst = tri["edge_cst"][idx]
        edge_ig = tri["edge_ig"][idx]
        edge_ia = tri["edge_ia"][idx]

        phi_cst = _edge_value(fractal_set.edges["cst"]["phi_cst"][edge_cst])
        phi_ig = _edge_value(fractal_set.edges["ig"]["theta_ij"][edge_ig])
        phi_ia = _edge_value(fractal_set.edges["ia"]["phi_ia"][edge_ia])
        phases.append(phi_cst + phi_ig + phi_ia)

    if not phases:
        return None

    phases_np = np.array(phases, dtype=np.float64)
    wilson = np.cos(phases_np)
    action = 1.0 - wilson

    return {
        "timestep": int(timestep),
        "n_loops": int(len(phases)),
        "phase_mean": float(phases_np.mean()),
        "phase_std": float(phases_np.std()),
        "wilson_mean": float(wilson.mean()),
        "wilson_std": float(wilson.std()),
        "action_mean": float(action.mean()),
        "action_std": float(action.std()),
        "wilson_values": wilson,
        "action_values": action,
    }


def _compute_wilson_timeseries(fractal_set) -> dict[str, np.ndarray]:
    times = []
    means = []
    actions = []
    for t_idx in range(1, fractal_set.n_recorded):
        loops = _compute_wilson_loops(fractal_set, t_idx)
        if loops is None:
            continue
        times.append(int(t_idx))
        means.append(float(loops["wilson_mean"]))
        actions.append(float(loops["action_mean"]))
    return {
        "time_index": np.array(times, dtype=np.int64),
        "wilson_mean": np.array(means, dtype=np.float64),
        "action_mean": np.array(actions, dtype=np.float64),
    }


def _plot_wilson_histogram(values: np.ndarray, title: str, path: Path) -> Path | None:
    if not HAS_MPL:
        return None
    if values.size == 0:
        return None
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=50, alpha=0.8)
    plt.xlabel("Wilson loop (Re)")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _plot_wilson_timeseries(
    time_index: np.ndarray, action_mean: np.ndarray, path: Path
) -> Path | None:
    if not HAS_MPL:
        return None
    if time_index.size == 0:
        return None
    plt.figure(figsize=(6, 4))
    plt.plot(time_index, action_mean, "-o", markersize=3)
    plt.xlabel("time index")
    plt.ylabel("Wilson action (mean)")
    plt.title("Wilson Action Over Time")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _masked_mean_std(values: torch.Tensor, alive: torch.Tensor) -> tuple[float, float]:
    alive_values = values[alive]
    if alive_values.numel() == 0:
        return 0.0, 0.0
    mean_val = alive_values.mean().item()
    if alive_values.numel() < 2:
        return mean_val, 0.0
    return mean_val, alive_values.std().item()


def _compute_hypocoercive_variance(
    positions: torch.Tensor, velocities: torch.Tensor, lambda_v: float = 1.0
) -> dict[str, float]:
    x = positions
    v = velocities

    x_mean = x.mean(dim=0, keepdim=True)
    v_mean = v.mean(dim=0, keepdim=True)

    var_x = ((x - x_mean) ** 2).sum(dim=1).mean().item()
    var_v = ((v - v_mean) ** 2).sum(dim=1).mean().item()
    var_h = var_x + lambda_v * var_v

    x_diff = x.unsqueeze(0) - x.unsqueeze(1)
    v_diff = v.unsqueeze(0) - v.unsqueeze(1)

    d_x = (x_diff**2).sum(dim=2).sqrt().max().item()
    d_v = (v_diff**2).sum(dim=2).sqrt().max().item()

    d_max_h_sq = d_x**2 + lambda_v * d_v**2
    d_max_h = float(np.sqrt(d_max_h_sq))

    ratio_x = var_x / (d_x**2) if d_x > 0 else 0.0
    ratio_v = var_v / (d_v**2) if d_v > 0 else 0.0
    ratio_h = var_h / d_max_h_sq if d_max_h_sq > 0 else 0.0

    return {
        "var_x": var_x,
        "var_v": var_v,
        "var_h": var_h,
        "lambda_v": lambda_v,
        "d_max_x": d_x,
        "d_max_v": d_v,
        "d_max_h": d_max_h,
        "d_max_h_sq": d_max_h_sq,
        "ratio_x": ratio_x,
        "ratio_v": ratio_v,
        "ratio_h": ratio_h,
    }


def _estimate_edge_budget(var_h: float, d_max_sq: float, d_close: float, K: int) -> float:
    numerator = d_max_sq - 2 * var_h
    denominator = d_max_sq - d_close**2

    if denominator <= 0:
        return K * (K - 1) / 2
    if numerator <= 0:
        return 0.0

    fraction = numerator / denominator
    return (K * (K - 1) / 2) * fraction


def _compute_variance_from_history(
    history: RunHistory,
    warmup_samples: int,
    lambda_v: float = 1.0,
) -> dict[str, float]:
    n_recorded = history.n_recorded
    if warmup_samples >= n_recorded:
        msg = f"warmup_samples ({warmup_samples}) >= n_recorded ({n_recorded})"
        raise ValueError(msg)

    x_qsd = history.x_final[warmup_samples:]
    v_qsd = history.v_final[warmup_samples:]

    n_qsd_samples = x_qsd.shape[0]
    samples_metrics = []

    for i in range(n_qsd_samples):
        metrics = _compute_hypocoercive_variance(x_qsd[i], v_qsd[i], lambda_v=lambda_v)
        samples_metrics.append(metrics)

    ratio_h_samples = [m["ratio_h"] for m in samples_metrics]
    var_h_samples = [m["var_h"] for m in samples_metrics]
    d_max_h_sq_samples = [m["d_max_h_sq"] for m in samples_metrics]

    ratio_h_mean = float(np.mean(ratio_h_samples)) if ratio_h_samples else 0.0
    ratio_h_std = float(np.std(ratio_h_samples)) if ratio_h_samples else 0.0
    var_h_mean = float(np.mean(var_h_samples)) if var_h_samples else 0.0
    d_max_h_sq_mean = float(np.mean(d_max_h_sq_samples)) if d_max_h_sq_samples else 0.0

    N = history.N
    d_close_threshold = float(np.sqrt(d_max_h_sq_mean / N)) if N > 0 else 0.0
    n_close_estimate = _estimate_edge_budget(var_h_mean, d_max_h_sq_mean, d_close_threshold, N)

    scaling_exponent = (
        float(np.log(n_close_estimate) / np.log(N)) if N > 1 and n_close_estimate > 0 else 0.0
    )

    return {
        "ratio_h_mean": ratio_h_mean,
        "ratio_h_std": ratio_h_std,
        "var_h_mean": var_h_mean,
        "d_max_h_sq_mean": d_max_h_sq_mean,
        "n_close_estimate": n_close_estimate,
        "scaling_exponent": scaling_exponent,
        "n_qsd_samples": n_qsd_samples,
    }


def _plot_correlation_decay(
    r: np.ndarray,
    C: np.ndarray,
    counts: np.ndarray,
    fit: dict[str, float],
    title: str,
    path: Path,
) -> Path | None:
    if not HAS_MPL:
        return None
    mask = (counts > 0) & (C > 0)
    if mask.sum() < 2:
        return None
    r_plot = r[mask]
    c_plot = C[mask]

    plt.figure(figsize=(6, 4))
    plt.plot(r_plot, c_plot, "o", label="C(r)")
    if fit.get("xi", 0.0) > 0 and fit.get("C0", 0.0) > 0:
        r_line = np.linspace(float(r_plot.min()), float(r_plot.max()), 200)
        c_line = fit["C0"] * np.exp(-(r_line**2) / (fit["xi"] ** 2))
        plt.plot(r_line, c_line, "-", label="exp fit")
    plt.yscale("log")
    plt.xlabel("r")
    plt.ylabel("C(r)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _plot_lyapunov(
    time: np.ndarray,
    V_total: np.ndarray,
    V_var_x: np.ndarray,
    V_var_v: np.ndarray,
    path: Path,
) -> Path | None:
    if not HAS_MPL:
        return None
    plt.figure(figsize=(6, 4))
    plt.plot(time, V_total, label="V_total")
    plt.plot(time, V_var_x, label="V_var_x")
    plt.plot(time, V_var_v, label="V_var_v")
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("Lyapunov")
    plt.title("Lyapunov Components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _plot_phase_histograms(
    u1_phases: torch.Tensor,
    su2_phases: torch.Tensor,
    alive: torch.Tensor,
    path: Path,
) -> Path | None:
    if not HAS_MPL:
        return None
    u1_vals = u1_phases[alive].cpu().numpy()
    su2_vals = su2_phases[alive].cpu().numpy()
    if u1_vals.size == 0 or su2_vals.size == 0:
        return None

    plt.figure(figsize=(6, 4))
    plt.hist(u1_vals, bins=50, alpha=0.6, label="U1 phase")
    plt.hist(su2_vals, bins=50, alpha=0.6, label="SU2 phase")
    plt.xlabel("phase")
    plt.ylabel("count")
    plt.title("Gauge Phase Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _compute_triangle_areas(
    history: RunHistory,
    time_index: np.ndarray,
    source_walker: np.ndarray,
    influencer_walker: np.ndarray,
) -> np.ndarray:
    device = history.x_before_clone.device
    t_idx = torch.as_tensor(time_index, device=device, dtype=torch.long)
    i_idx = torch.as_tensor(source_walker, device=device, dtype=torch.long)
    j_idx = torch.as_tensor(influencer_walker, device=device, dtype=torch.long)

    x_i = history.x_before_clone[t_idx, i_idx]
    x_j = history.x_before_clone[t_idx, j_idx]
    x_f = history.x_final[t_idx, i_idx]

    a = x_j - x_i
    b = x_f - x_i
    a2 = (a * a).sum(dim=-1)
    b2 = (b * b).sum(dim=-1)
    dot = (a * b).sum(dim=-1)
    area_sq = torch.clamp(a2 * b2 - dot * dot, min=0.0)
    area = 0.5 * torch.sqrt(area_sq)
    return area.cpu().numpy()


def _compute_string_tension(
    history: RunHistory,
    fractal_set: FractalSet,
    max_triangles: int,
    n_bins: int,
) -> dict[str, Any] | None:
    tri = fractal_set.triangles
    n_tri = len(tri["time_index"])
    if n_tri == 0:
        return None

    idx = np.arange(n_tri)
    if n_tri > max_triangles:
        idx = np.linspace(0, n_tri - 1, max_triangles, dtype=int)

    time_index = np.asarray(tri["time_index"], dtype=int)[idx]
    source_walker = np.asarray(tri["source_walker"], dtype=int)[idx]
    influencer_walker = np.asarray(tri["influencer_walker"], dtype=int)[idx]
    edge_cst = np.asarray(tri["edge_cst"], dtype=int)[idx]
    edge_ig = np.asarray(tri["edge_ig"], dtype=int)[idx]
    edge_ia = np.asarray(tri["edge_ia"], dtype=int)[idx]

    areas = _compute_triangle_areas(history, time_index, source_walker, influencer_walker)

    phi_cst = np.asarray(fractal_set.edges["cst"]["phi_cst"], dtype=float)
    theta_ig = np.asarray(fractal_set.edges["ig"]["theta_ij"], dtype=float)
    phi_ia = np.asarray(fractal_set.edges["ia"]["phi_ia"], dtype=float)

    phases = phi_cst[edge_cst] + theta_ig[edge_ig] + phi_ia[edge_ia]
    wilson = np.cos(phases)
    valid = (wilson > 0) & np.isfinite(areas)
    if valid.sum() < 5:
        return None

    areas = areas[valid]
    wilson = wilson[valid]

    area_min = float(areas.min())
    area_max = float(areas.max())
    if area_max <= area_min:
        return None

    bin_edges = np.linspace(area_min, area_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    mean_w = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=float)

    for i in range(n_bins):
        mask = (areas >= bin_edges[i]) & (areas < bin_edges[i + 1])
        if mask.sum() > 0:
            mean_w[i] = float(wilson[mask].mean())
            counts[i] = float(mask.sum())

    fit_mask = (counts > 0) & (mean_w > 0)
    if fit_mask.sum() < 2:
        return None

    x = bin_centers[fit_mask]
    y = np.log(mean_w[fit_mask])
    coeffs = np.polyfit(x, y, 1, w=np.sqrt(counts[fit_mask]))
    slope, intercept = coeffs
    sigma = float(-slope)

    y_pred = intercept + slope * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "sigma": sigma,
        "r_squared": r_squared,
        "n_triangles": int(n_tri),
        "n_used": int(valid.sum()),
        "area_min": area_min,
        "area_max": area_max,
        "bin_centers": bin_centers,
        "mean_w": mean_w,
        "counts": counts,
    }


def _estimate_time_step(times: np.ndarray, fallback: float) -> float:
    if times.size < 2:
        return float(fallback)
    diffs = np.diff(times)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return float(fallback)
    return float(np.median(diffs))


def _fit_particle_mass(
    lag_times: np.ndarray,
    corr: np.ndarray,
    fit_start: int,
    fit_stop: int | None,
    fit_mode: str,
    plateau_min_points: int,
    plateau_max_points: int | None,
    plateau_max_cv: float | None,
) -> dict[str, Any]:
    fit_stop_idx = corr.size - 1 if fit_stop is None else min(fit_stop, corr.size - 1)
    fit_start_idx = max(0, fit_start)
    base_fit = fit_mass_exponential(
        lag_times,
        corr,
        fit_start=fit_start_idx,
        fit_stop=fit_stop_idx,
    )
    base_fit["fit_start"] = int(fit_start_idx)
    base_fit["fit_stop"] = int(fit_stop_idx)
    base_fit["fit_mode"] = "window"

    if fit_mode not in {"plateau", "auto"}:
        return base_fit

    plateau = select_mass_plateau(
        lag_times,
        corr,
        fit_start=fit_start_idx,
        fit_stop=fit_stop_idx,
        min_points=plateau_min_points,
        max_points=plateau_max_points,
        max_cv=plateau_max_cv,
    )
    if plateau is None:
        if fit_mode == "plateau":
            base_fit["fit_mode"] = "window_fallback"
            base_fit["plateau"] = None
        return base_fit

    plateau_fit = fit_mass_exponential(
        lag_times,
        corr,
        fit_start=int(plateau["fit_start"]),
        fit_stop=int(plateau["fit_stop"]),
    )
    plateau_fit["fit_start"] = int(plateau["fit_start"])
    plateau_fit["fit_stop"] = int(plateau["fit_stop"])
    plateau_fit["fit_mode"] = "plateau"
    plateau_fit["plateau"] = plateau
    return plateau_fit


def _compute_particle_observables(
    history: RunHistory,
    operators: tuple[str, ...],
    h_eff: float,
    mass: float,
    ell0: float | None,
    max_lag: int | None,
    fit_start: int,
    fit_stop: int,
    fit_mode: str,
    plateau_min_points: int,
    plateau_max_points: int | None,
    plateau_max_cv: float | None,
    use_connected: bool,
    neighbor_method: str,
    knn_k: int,
    knn_sample: int | None,
    meson_reduce: str,
    baryon_pairs: int | None,
    warmup_fraction: float,
    glueball_data: dict[str, np.ndarray] | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    supported = {"baryon", "meson", "glueball"}
    requested = {op.strip().lower() for op in operators if op}
    unknown = sorted(requested - supported)
    ops = set(requested & supported)
    errors: dict[str, str] = {}

    if unknown:
        errors["unknown"] = f"Unsupported operators: {', '.join(unknown)}"

    if "baryon" in ops and history.d != 3:
        errors["baryon"] = "Baryon operator requires d=3."
        ops.remove("baryon")

    if neighbor_method == "knn" and "baryon" in ops and knn_k < 2:
        errors["baryon"] = "particle_knn_k must be >= 2 for baryon operator."
        ops.remove("baryon")

    series_ops = sorted(op for op in ops if op != "glueball")
    series_lists = {op: [] for op in series_ops}
    time_index: list[int] = []
    time_tau: list[float] = []
    ell0_values: list[float] = []

    start_idx = max(1, int(history.n_recorded * warmup_fraction))
    fallback_dt = history.delta_t * history.record_every

    for t_idx in range(start_idx, history.n_recorded):
        info_idx = t_idx - 1
        alive = history.alive_mask[info_idx]
        if not alive.any():
            continue

        x_pre = history.x_before_clone[t_idx]
        v_pre = history.v_before_clone[t_idx]
        comp_dist = history.companions_distance[info_idx]
        comp_clone = history.companions_clone[info_idx]

        sample_indices = torch.where(alive)[0]
        if knn_sample is not None and sample_indices.numel() > knn_sample:
            sample_indices = sample_indices[:knn_sample]

        if ell0 is None:
            dist = compute_companion_distance(x_pre, comp_dist, history.pbc, history.bounds)
            if dist.numel() > 0 and alive.any():
                ell0_t = float(dist[alive].mean().item())
            else:
                ell0_t = 1.0
        else:
            ell0_t = ell0

        if ell0_t <= 0:
            errors.setdefault("ell0", "ell0 collapsed to 0; using 1.0")
            ell0_t = 1.0

        try:
            color, color_valid = compute_color_state(
                history.force_viscous[info_idx],
                v_pre,
                h_eff,
                mass,
                ell0_t,
            )
        except ValueError as exc:
            errors["color_state"] = str(exc)
            break

        neighbor_indices = None
        if neighbor_method == "knn" and series_ops:
            try:
                neighbor_indices = compute_knn_indices(
                    x_pre,
                    alive,
                    knn_k,
                    history.pbc,
                    history.bounds,
                    sample_indices=sample_indices,
                )
            except ValueError as exc:
                errors["knn"] = str(exc)
                neighbor_indices = None

        for op in series_ops:
            if op not in series_lists:
                continue
            if op == "meson":
                if neighbor_method == "knn":
                    if neighbor_indices is None or sample_indices.numel() == 0:
                        series_lists[op].append(0.0 + 0.0j)
                    else:
                        meson, valid = compute_meson_operator_knn(
                            color,
                            sample_indices,
                            neighbor_indices,
                            alive,
                            color_valid,
                            reduce=meson_reduce,
                        )
                        value = meson[valid].mean().item() if valid.any() else 0.0 + 0.0j
                        series_lists[op].append(value)
                else:
                    meson, valid = compute_meson_operator(color, comp_dist, alive, color_valid)
                    value = meson[valid].mean().item() if valid.any() else 0.0 + 0.0j
                    series_lists[op].append(value)
            elif op == "baryon":
                if neighbor_method == "knn":
                    if neighbor_indices is None or sample_indices.numel() == 0:
                        series_lists[op].append(0.0 + 0.0j)
                    else:
                        try:
                            baryon, valid = compute_baryon_operator_knn(
                                color,
                                sample_indices,
                                neighbor_indices,
                                alive,
                                color_valid,
                                max_pairs=baryon_pairs,
                            )
                        except ValueError as exc:
                            errors["baryon"] = str(exc)
                            ops.discard("baryon")
                            series_lists.pop("baryon", None)
                            continue
                        value = baryon[valid].mean().item() if valid.any() else 0.0 + 0.0j
                        series_lists[op].append(value)
                else:
                    try:
                        baryon, valid = compute_baryon_operator(
                            color, comp_dist, comp_clone, alive, color_valid
                        )
                    except ValueError as exc:
                        errors["baryon"] = str(exc)
                        ops.discard("baryon")
                        series_lists.pop("baryon", None)
                        continue
                    value = baryon[valid].mean().item() if valid.any() else 0.0 + 0.0j
                    series_lists[op].append(value)

        time_index.append(int(t_idx))
        time_tau.append(float(history.recorded_steps[t_idx] * history.delta_t))
        ell0_values.append(float(ell0_t))

    arrays: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {
        "operators": {},
        "errors": errors if errors else None,
        "neighbor_method": neighbor_method,
        "knn_k": int(knn_k),
        "knn_sample": int(knn_sample) if knn_sample is not None else None,
        "meson_reduce": meson_reduce,
        "baryon_pairs": int(baryon_pairs) if baryon_pairs is not None else None,
    }

    if time_tau:
        arrays["particle_time_index"] = np.array(time_index, dtype=np.int64)
        arrays["particle_time_tau"] = np.array(time_tau, dtype=np.float64)
        arrays["particle_ell0"] = np.array(ell0_values, dtype=np.float64)

    dt = _estimate_time_step(np.array(time_tau, dtype=np.float64), fallback_dt)

    for op, values in series_lists.items():
        series = np.array(values, dtype=np.complex128)
        arrays[f"particle_{op}_series"] = series

        lags, corr = compute_time_correlator(series, max_lag=max_lag, use_connected=use_connected)
        lag_times = lags.astype(np.float64) * dt
        arrays[f"particle_{op}_lags"] = lag_times
        arrays[f"particle_{op}_corr"] = np.real(corr)
        arrays[f"particle_{op}_eff_mass"] = compute_effective_mass(corr, dt)

        fit = _fit_particle_mass(
            lag_times,
            corr,
            fit_start=fit_start,
            fit_stop=fit_stop,
            fit_mode=fit_mode,
            plateau_min_points=plateau_min_points,
            plateau_max_points=plateau_max_points,
            plateau_max_cv=plateau_max_cv,
        )
        metrics["operators"][op] = {
            "n_samples": int(series.size),
            "dt": float(dt),
            "series_mean_real": float(np.real(series.mean())) if series.size else 0.0,
            "series_mean_abs": float(np.abs(series).mean()) if series.size else 0.0,
            "fit": fit,
        }

    if "glueball" in ops:
        if glueball_data is None:
            errors["glueball"] = "Glueball operator requires --build-fractal-set."
        else:
            glue_series = glueball_data["series"]
            glue_tau = glueball_data["tau"]
            glue_dt = _estimate_time_step(glue_tau, fallback_dt)

            arrays["particle_glueball_time_index"] = glueball_data["time_index"]
            arrays["particle_glueball_time_tau"] = glue_tau
            arrays["particle_glueball_series"] = glue_series

            lags, corr = compute_time_correlator(
                glue_series, max_lag=max_lag, use_connected=use_connected
            )
            lag_times = lags.astype(np.float64) * glue_dt
            arrays["particle_glueball_lags"] = lag_times
            arrays["particle_glueball_corr"] = np.real(corr)
            arrays["particle_glueball_eff_mass"] = compute_effective_mass(corr, glue_dt)

            fit = _fit_particle_mass(
                lag_times,
                corr,
                fit_start=fit_start,
                fit_stop=fit_stop,
                fit_mode=fit_mode,
                plateau_min_points=plateau_min_points,
                plateau_max_points=plateau_max_points,
                plateau_max_cv=plateau_max_cv,
            )
            metrics["operators"]["glueball"] = {
                "n_samples": int(glue_series.size),
                "dt": float(glue_dt),
                "series_mean": float(np.mean(glue_series)) if glue_series.size else 0.0,
                "fit": fit,
            }

    metrics["errors"] = errors if errors else None
    return metrics, arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-path", default=None)
    parser.add_argument("--output-dir", default="outputs/fractal_gas_potential_well_analysis")
    parser.add_argument("--analysis-id", default=None)
    parser.add_argument("--analysis-time-index", type=int, default=None)
    parser.add_argument("--analysis-step", type=int, default=None)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--h-eff", type=float, default=1.0)
    parser.add_argument("--correlation-r-max", type=float, default=0.5)
    parser.add_argument("--correlation-bins", type=int, default=50)
    parser.add_argument("--gradient-neighbors", type=int, default=5)
    parser.add_argument("--build-fractal-set", action="store_true")
    parser.add_argument("--fractal-set-stride", type=int, default=10)
    # New options for proper local field analysis
    parser.add_argument(
        "--use-local-fields",
        action="store_true",
        help="Use proper local fields (density-based diversity) instead of companion-based d_prime",
    )
    parser.add_argument(
        "--use-connected",
        action="store_true",
        help="Compute connected correlator G(r) = <φφ> - <φ>² instead of raw product",
    )
    parser.add_argument(
        "--density-sigma",
        type=float,
        default=0.5,
        help="Kernel width for local density estimation (default: 0.5)",
    )
    parser.add_argument(
        "--compute-particles",
        action="store_true",
        help="Compute particle observables (meson/baryon/glueball) and masses.",
    )
    parser.add_argument(
        "--particle-operators",
        default="baryon,meson,glueball",
        help="Comma-separated particle operators to compute (glueball needs --build-fractal-set).",
    )
    parser.add_argument("--particle-max-lag", type=int, default=80)
    parser.add_argument("--particle-fit-start", type=int, default=7)
    parser.add_argument("--particle-fit-stop", type=int, default=16)
    parser.add_argument(
        "--particle-fit-mode",
        choices=["window", "plateau", "auto"],
        default="window",
        help="Particle mass fit mode: fixed window, effective-mass plateau, or auto fallback.",
    )
    parser.add_argument(
        "--particle-plateau-min-points",
        type=int,
        default=3,
        help="Minimum effective-mass points for plateau selection.",
    )
    parser.add_argument(
        "--particle-plateau-max-points",
        type=int,
        default=None,
        help="Optional maximum effective-mass points for plateau selection.",
    )
    parser.add_argument(
        "--particle-plateau-max-cv",
        type=float,
        default=0.2,
        help="Max coefficient of variation for plateau selection (larger = looser).",
    )
    parser.add_argument(
        "--particle-mass",
        type=float,
        default=1.0,
        help="Walker mass scale used in color-state phases.",
    )
    parser.add_argument(
        "--particle-ell0",
        type=float,
        default=None,
        help="Characteristic length for momentum phase (defaults to mean IG distance).",
    )
    parser.add_argument(
        "--particle-use-connected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use connected correlator for particle fits.",
    )
    parser.add_argument(
        "--particle-neighbor-method",
        choices=["companion", "knn"],
        default="knn",
        help="Neighbor selection for baryon/meson operators.",
    )
    parser.add_argument("--particle-knn-k", type=int, default=4)
    parser.add_argument("--particle-knn-sample", type=int, default=512)
    parser.add_argument(
        "--particle-meson-reduce",
        choices=["mean", "first"],
        default="mean",
        help="Reduction over k-NN meson pairs.",
    )
    parser.add_argument(
        "--particle-baryon-pairs",
        type=int,
        default=None,
        help="Max number of k-NN baryon pairs to average (defaults to all).",
    )
    parser.add_argument("--compute-string-tension", action="store_true")
    parser.add_argument("--string-tension-max-triangles", type=int, default=20000)
    parser.add_argument("--string-tension-bins", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    particle_ops = tuple(
        op.strip().lower() for op in args.particle_operators.split(",") if op.strip()
    )

    analysis_cfg = AnalysisConfig(
        analysis_time_index=args.analysis_time_index,
        analysis_step=args.analysis_step,
        warmup_fraction=args.warmup_fraction,
        h_eff=args.h_eff,
        correlation_r_max=args.correlation_r_max,
        correlation_bins=args.correlation_bins,
        gradient_neighbors=args.gradient_neighbors,
        build_fractal_set=args.build_fractal_set,
        fractal_set_stride=args.fractal_set_stride,
        use_local_fields=args.use_local_fields,
        use_connected=args.use_connected,
        density_sigma=args.density_sigma,
        compute_particles=args.compute_particles,
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
        compute_string_tension=args.compute_string_tension,
        string_tension_max_triangles=args.string_tension_max_triangles,
        string_tension_bins=args.string_tension_bins,
    )
    if analysis_cfg.h_eff <= 0:
        raise ValueError("h_eff must be positive")
    if analysis_cfg.correlation_r_max <= 0:
        raise ValueError("correlation_r_max must be positive")
    if analysis_cfg.correlation_bins <= 0:
        raise ValueError("correlation_bins must be positive")
    if analysis_cfg.gradient_neighbors <= 0:
        raise ValueError("gradient_neighbors must be positive")
    if analysis_cfg.fractal_set_stride <= 0:
        raise ValueError("fractal_set_stride must be positive")
    if analysis_cfg.density_sigma <= 0:
        raise ValueError("density_sigma must be positive")
    if analysis_cfg.compute_particles:
        if analysis_cfg.particle_mass <= 0:
            raise ValueError("particle_mass must be positive")
        if analysis_cfg.particle_ell0 is not None and analysis_cfg.particle_ell0 <= 0:
            raise ValueError("particle_ell0 must be positive when set")
        if analysis_cfg.particle_fit_start < 0:
            raise ValueError("particle_fit_start must be >= 0")
        if analysis_cfg.particle_fit_stop < analysis_cfg.particle_fit_start:
            raise ValueError("particle_fit_stop must be >= particle_fit_start")
        if analysis_cfg.particle_fit_mode not in {"window", "plateau", "auto"}:
            raise ValueError("particle_fit_mode must be 'window', 'plateau', or 'auto'")
        if analysis_cfg.particle_plateau_min_points <= 0:
            raise ValueError("particle_plateau_min_points must be positive")
        if (
            analysis_cfg.particle_plateau_max_points is not None
            and analysis_cfg.particle_plateau_max_points <= 0
        ):
            raise ValueError("particle_plateau_max_points must be positive when set")
        if (
            analysis_cfg.particle_plateau_max_points is not None
            and analysis_cfg.particle_plateau_max_points
            < analysis_cfg.particle_plateau_min_points
        ):
            raise ValueError(
                "particle_plateau_max_points must be >= particle_plateau_min_points"
            )
        if (
            analysis_cfg.particle_plateau_max_cv is not None
            and analysis_cfg.particle_plateau_max_cv <= 0
        ):
            raise ValueError("particle_plateau_max_cv must be positive when set")
        if analysis_cfg.particle_max_lag is not None and analysis_cfg.particle_max_lag <= 0:
            raise ValueError("particle_max_lag must be positive when set")
        if analysis_cfg.particle_neighbor_method not in {"companion", "knn"}:
            raise ValueError("particle_neighbor_method must be 'companion' or 'knn'")
        if analysis_cfg.particle_knn_k <= 0:
            raise ValueError("particle_knn_k must be positive")
        if (
            analysis_cfg.particle_knn_sample is not None
            and analysis_cfg.particle_knn_sample <= 0
        ):
            raise ValueError("particle_knn_sample must be positive when set")
        if analysis_cfg.particle_meson_reduce not in {"mean", "first"}:
            raise ValueError("particle_meson_reduce must be 'mean' or 'first'")
        if (
            analysis_cfg.particle_baryon_pairs is not None
            and analysis_cfg.particle_baryon_pairs <= 0
        ):
            raise ValueError("particle_baryon_pairs must be positive when set")
    if analysis_cfg.compute_string_tension:
        if analysis_cfg.string_tension_max_triangles <= 0:
            raise ValueError("string_tension_max_triangles must be positive")
        if analysis_cfg.string_tension_bins <= 1:
            raise ValueError("string_tension_bins must be > 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_id = args.analysis_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    progress_path = output_dir / f"{analysis_id}_progress.json"

    if args.history_path is None:
        history_path = _find_latest_history(Path("outputs/fractal_gas_potential_well"))
    else:
        history_path = Path(args.history_path)

    history = RunHistory.load(str(history_path))
    if history.n_recorded <= 1:
        msg = "RunHistory has no recorded steps beyond t=0; rerun with record_every <= n_steps."
        raise ValueError(msg)

    analysis_time_idx = _select_time_index(history, analysis_cfg)
    if analysis_time_idx <= 0:
        analysis_time_idx = 1
    if analysis_time_idx >= history.n_recorded:
        msg = (
            f"analysis_time_index {analysis_time_idx} out of bounds "
            f"[0, {history.n_recorded - 1}]"
        )
        raise ValueError(msg)

    analysis_step = history.recorded_steps[analysis_time_idx]
    info_idx = analysis_time_idx - 1

    params = history.params or {}
    obs_params = {
        "alpha": _get_param(params, ["fitness", "alpha"], 1.0),
        "beta": _get_param(params, ["fitness", "beta"], 1.0),
        "eta": _get_param(params, ["fitness", "eta"], 0.1),
        "A": _get_param(params, ["fitness", "A"], 2.0),
        "lambda_alg": _get_param(params, ["fitness", "lambda_alg"], 0.0),
        "sigma_min": _get_param(params, ["fitness", "sigma_min"], 1e-8),
        "epsilon_dist": _get_param(params, ["fitness", "epsilon_dist"], 1e-8),
        "rho": _get_param(params, ["fitness", "rho"], None),
    }

    epsilon_clone = _get_param(params, ["cloning", "epsilon_clone"], 1e-8)
    epsilon_companion = _get_param(params, ["companion_selection", "epsilon"], 0.1)
    epsilon_companion_clone = _get_param(
        params, ["companion_selection_clone", "epsilon"], epsilon_companion
    )
    epsilon_clone = max(float(epsilon_clone), 1e-12)
    epsilon_companion = max(float(epsilon_companion), 1e-12)
    epsilon_companion_clone = max(float(epsilon_companion_clone), 1e-12)
    lambda_alg = _get_param(
        params, ["companion_selection", "lambda_alg"], obs_params["lambda_alg"]
    )

    _write_progress(
        progress_path,
        "loaded",
        {
            "analysis_id": analysis_id,
            "history_path": str(history_path),
            "analysis_time_index": analysis_time_idx,
            "analysis_step": analysis_step,
            "analysis_config": asdict(analysis_cfg),
            "obs_params": obs_params,
            "companion_epsilon": epsilon_companion,
            "companion_epsilon_clone": epsilon_companion_clone,
            "epsilon_clone": epsilon_clone,
            "lambda_alg": lambda_alg,
            "plotting": {"enabled": HAS_MPL, "error": MPL_ERROR},
        },
    )

    x_pre = history.x_before_clone[analysis_time_idx]
    v_pre = history.v_before_clone[analysis_time_idx]
    rewards = history.rewards[info_idx]
    alive = history.alive_mask[info_idx]
    companions_distance = history.companions_distance[info_idx]
    companions_clone = history.companions_clone[info_idx]

    fields = _compute_collective_fields(
        x_pre,
        v_pre,
        rewards,
        alive,
        companions_distance,
        obs_params,
        history.bounds,
        history.pbc,
    )

    d_prime = fields["d_prime"]
    r_prime = fields["r_prime"]
    d_prime_mean, d_prime_std = _masked_mean_std(d_prime, alive)
    r_prime_mean, r_prime_std = _masked_mean_std(r_prime, alive)

    d_bins, d_corr, d_counts = _bin_by_distance(
        x_pre, d_prime, alive, analysis_cfg.correlation_r_max, analysis_cfg.correlation_bins
    )
    r_bins, r_corr, r_counts = _bin_by_distance(
        x_pre, r_prime, alive, analysis_cfg.correlation_r_max, analysis_cfg.correlation_bins
    )

    d_fit = _fit_exponential_decay(d_bins, d_corr, d_counts)
    r_fit = _fit_exponential_decay(r_bins, r_corr, r_counts)

    # Compute proper local fields and their correlations when requested
    local_fields = None
    local_correlations = {}
    if analysis_cfg.use_local_fields:
        local_fields = _compute_local_fields(
            x_pre,
            v_pre,
            rewards,
            alive,
            sigma_density=analysis_cfg.density_sigma,
        )

        # Choose correlation function based on --use-connected flag
        bin_func = _bin_by_distance_connected if analysis_cfg.use_connected else _bin_by_distance
        # Use appropriate fitting function for connected correlators
        fit_func = _fit_connected_correlator if analysis_cfg.use_connected else _fit_exponential_decay

        for field_name, field_values in local_fields.items():
            bins, corr, counts = bin_func(
                x_pre,
                field_values,
                alive,
                analysis_cfg.correlation_r_max,
                analysis_cfg.correlation_bins,
            )
            fit = fit_func(bins, corr, counts)
            local_correlations[field_name] = {
                "bins": bins,
                "correlation": corr,
                "counts": counts,
                "fit": fit,
            }

    local_fields_summary = None
    if local_correlations:
        local_fields_summary = {
            field_name: {
                "fit": corr_data["fit"],
                "mean": _masked_mean_std(local_fields[field_name], alive)[0],
                "std": _masked_mean_std(local_fields[field_name], alive)[1],
            }
            for field_name, corr_data in local_correlations.items()
        }

    fitness_stats = _fitness_stats(fields["fitness"], alive)

    d_grad = _compute_field_gradients(
        x_pre, d_prime, alive, analysis_cfg.gradient_neighbors
    )
    r_grad = _compute_field_gradients(
        x_pre, r_prime, alive, analysis_cfg.gradient_neighbors
    )
    d_grad_mean, d_grad_std = _masked_mean_std(d_grad, alive)
    r_grad_mean, r_grad_std = _masked_mean_std(r_grad, alive)

    u1_phases = _compute_u1_phases(
        fields["fitness"], companions_distance, alive, analysis_cfg.h_eff
    )
    u1_amplitudes = _compute_u1_amplitude(
        x_pre,
        v_pre,
        alive,
        epsilon_companion,
        lambda_alg,
        history.bounds,
        history.pbc,
    )

    alive_indices = torch.where(alive)[0]
    if len(alive_indices) > 0:
        walker_i = int(alive_indices[0].item())
        u1_state = _compute_dressed_state(u1_phases, u1_amplitudes, walker_i)
        u1_norm = _compute_norm_squared(u1_state, alive)
    else:
        walker_i = 0
        u1_norm = 0.0

    su2_phases = _compute_su2_phases(
        fields["fitness"], alive, companions_clone, epsilon_clone, analysis_cfg.h_eff
    )
    su2_amplitudes = _compute_su2_pairing_probability(
        x_pre,
        v_pre,
        alive,
        epsilon_companion_clone,
        lambda_alg,
        history.bounds,
        history.pbc,
    )

    if len(alive_indices) > 0:
        walker_j = int(companions_clone[walker_i].item())
        su2_state = _compute_su2_doublet_state(su2_phases, su2_amplitudes, walker_i, walker_j)
        su2_norm = _compute_norm_squared(su2_state, alive)
    else:
        su2_norm = 0.0

    u1_phase_mean, u1_phase_std = _masked_mean_std(u1_phases, alive)
    su2_phase_mean, su2_phase_std = _masked_mean_std(su2_phases, alive)

    observables = {
        "fitness_stats": fitness_stats,
        "d_prime_mean": d_prime_mean,
        "d_prime_std": d_prime_std,
        "r_prime_mean": r_prime_mean,
        "r_prime_std": r_prime_std,
        "d_prime_gradient_mean": d_grad_mean,
        "d_prime_gradient_std": d_grad_std,
        "r_prime_gradient_mean": r_grad_mean,
        "r_prime_gradient_std": r_grad_std,
        "d_prime_correlation": d_fit,
        "r_prime_correlation": r_fit,
    }

    _write_progress(
        progress_path,
        "observables",
        {"observables": observables, "local_fields": local_fields_summary},
    )

    u1_summary = {
        "phase_mean": u1_phase_mean,
        "phase_std": u1_phase_std,
        "gauge_invariant_norm": u1_norm,
        "walker_index": walker_i,
    }
    su2_summary = {
        "phase_mean": su2_phase_mean,
        "phase_std": su2_phase_std,
        "gauge_invariant_norm": su2_norm,
        "walker_index": walker_i,
    }
    _write_progress(progress_path, "gauge_phases", {"u1": u1_summary, "su2": su2_summary})

    qsd_metrics = None
    if history.n_recorded > 1:
        warmup_samples = int(
            max(
                1,
                min(
                    history.n_recorded - 1,
                    history.n_recorded * analysis_cfg.warmup_fraction,
                ),
            )
        )
        qsd_metrics = _compute_variance_from_history(history, warmup_samples, lambda_v=1.0)
        _write_progress(progress_path, "qsd_variance", {"qsd_variance": qsd_metrics})

    lyapunov = compute_lyapunov_components_trajectory(history, stage="final")
    lyapunov_summary = {
        "initial_total": lyapunov["V_total"][0].item(),
        "final_total": lyapunov["V_total"][-1].item(),
        "initial_position_ratio": lyapunov["position_ratio"][0].item(),
        "final_position_ratio": lyapunov["position_ratio"][-1].item(),
    }
    _write_progress(progress_path, "lyapunov", {"lyapunov": lyapunov_summary})

    history_small = None
    fractal_set_metrics = None
    wilson_metrics = None
    wilson_timeseries = None
    string_tension = None
    if analysis_cfg.build_fractal_set:
        if HAS_FRACTAL_SET:
            history_small = _downsample_history(history, analysis_cfg.fractal_set_stride)
            fractal_set = FractalSet(history_small)
            timestep = max(0, history_small.n_recorded - 1)
            curvature = compute_ricci_from_fractal_set_graph(fractal_set, timestep=timestep)
            curvature_hessian = compute_ricci_from_fractal_set_hessian(
                fractal_set, timestep=timestep
            )
            cheeger = None
            if curvature_hessian.get("has_hessian_data"):
                cheeger = check_cheeger_consistency(
                    curvature_hessian.get("ricci_scalars", np.array([])),
                    curvature.get("eigenvalues", np.array([])),
                )
            fractal_set_metrics = {
                "downsample_stride": analysis_cfg.fractal_set_stride,
                "n_recorded": history_small.n_recorded,
                "total_nodes": fractal_set.total_nodes,
                "num_cst_edges": fractal_set.num_cst_edges,
                "num_ig_edges": fractal_set.num_ig_edges,
                "num_ia_edges": fractal_set.num_ia_edges,
                "num_clone_edges": fractal_set.num_clone_edges,
                "num_triangles": fractal_set.num_triangles,
                "curvature": {
                    "timestep": curvature.get("timestep"),
                    "spectral_gap": curvature.get("spectral_gap"),
                    "mean_ricci_estimate": curvature.get("mean_ricci_estimate"),
                    "n_walkers": curvature.get("n_walkers"),
                },
                "curvature_hessian": {
                    "has_hessian_data": curvature_hessian.get("has_hessian_data"),
                    "mean_ricci": curvature_hessian.get("mean_ricci"),
                    "std_ricci": curvature_hessian.get("std_ricci"),
                },
                "cheeger_consistency": cheeger,
            }
            wilson_metrics = _compute_wilson_loops(fractal_set, timestep=timestep)
            wilson_timeseries = _compute_wilson_timeseries(fractal_set)
            if analysis_cfg.compute_string_tension:
                string_tension = _compute_string_tension(
                    history_small,
                    fractal_set,
                    max_triangles=analysis_cfg.string_tension_max_triangles,
                    n_bins=analysis_cfg.string_tension_bins,
                )
        else:
            fractal_set_metrics = {"error": FRACTAL_SET_ERROR}

    glueball_data = None
    if wilson_timeseries is not None and history_small is not None:
        time_idx = wilson_timeseries["time_index"]
        if time_idx.size > 0:
            recorded_steps = np.array(history_small.recorded_steps, dtype=np.float64)
            tau = recorded_steps[time_idx] * history_small.delta_t
            glueball_data = {
                "time_index": time_idx,
                "tau": tau,
                "series": wilson_timeseries["action_mean"],
            }

    _write_progress(
        progress_path,
        "fractal_set",
        {
            "fractal_set": fractal_set_metrics,
            "wilson_loops": wilson_metrics,
            "wilson_timeseries": wilson_timeseries if analysis_cfg.build_fractal_set else None,
            "string_tension": string_tension,
        },
    )

    plot_paths = {}
    if HAS_MPL:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        d_plot = _plot_correlation_decay(
            d_bins,
            d_corr,
            d_counts,
            d_fit,
            "Diversity Correlation Decay",
            plots_dir / f"{analysis_id}_d_prime_corr.png",
        )
        r_plot = _plot_correlation_decay(
            r_bins,
            r_corr,
            r_counts,
            r_fit,
            "Reward Correlation Decay",
            plots_dir / f"{analysis_id}_r_prime_corr.png",
        )
        lyap_plot = _plot_lyapunov(
            lyapunov["time"].cpu().numpy(),
            lyapunov["V_total"].cpu().numpy(),
            lyapunov["V_var_x"].cpu().numpy(),
            lyapunov["V_var_v"].cpu().numpy(),
            plots_dir / f"{analysis_id}_lyapunov.png",
        )
        phase_plot = _plot_phase_histograms(
            u1_phases,
            su2_phases,
            alive,
            plots_dir / f"{analysis_id}_phase_hist.png",
        )
        wilson_hist_plot = None
        wilson_time_plot = None
        if wilson_metrics is not None:
            wilson_hist_plot = _plot_wilson_histogram(
                wilson_metrics["wilson_values"],
                "Wilson Loop Distribution",
                plots_dir / f"{analysis_id}_wilson_hist.png",
            )
        if analysis_cfg.build_fractal_set and wilson_timeseries is not None:
            wilson_time_plot = _plot_wilson_timeseries(
                wilson_timeseries["time_index"],
                wilson_timeseries["action_mean"],
                plots_dir / f"{analysis_id}_wilson_time.png",
            )

        # Plot local field correlations when requested
        local_field_plots = {}
        if analysis_cfg.use_local_fields and local_correlations:
            connected_suffix = "_connected" if analysis_cfg.use_connected else ""
            field_titles = {
                "density": "Local Density",
                "diversity_local": "Local Diversity (1/ρ)",
                "radial": "Radial Distance ||x||",
                "kinetic": "Kinetic Energy",
                "reward_raw": "Raw Reward",
            }
            for field_name, corr_data in local_correlations.items():
                title = field_titles.get(field_name, field_name)
                plot_path = _plot_correlation_decay(
                    corr_data["bins"],
                    corr_data["correlation"],
                    corr_data["counts"],
                    corr_data["fit"],
                    f"{title} Correlation{connected_suffix}",
                    plots_dir / f"{analysis_id}_{field_name}_corr{connected_suffix}.png",
                )
                if plot_path:
                    local_field_plots[field_name] = str(plot_path)

        plot_paths = {
            "d_prime_correlation": str(d_plot) if d_plot else None,
            "r_prime_correlation": str(r_plot) if r_plot else None,
            "lyapunov": str(lyap_plot) if lyap_plot else None,
            "phase_histograms": str(phase_plot) if phase_plot else None,
            "wilson_histogram": str(wilson_hist_plot) if wilson_hist_plot else None,
            "wilson_timeseries": str(wilson_time_plot) if wilson_time_plot else None,
            "local_fields": local_field_plots if local_field_plots else None,
        }

    _write_progress(progress_path, "plots", {"plots": plot_paths})

    particle_metrics = None
    particle_arrays: dict[str, np.ndarray] = {}
    if analysis_cfg.compute_particles:
        particle_metrics, particle_arrays = _compute_particle_observables(
            history=history,
            operators=analysis_cfg.particle_operators,
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

    _write_progress(progress_path, "particles", {"particle_observables": particle_metrics})

    summary = history.summary()
    analysis = {
        "history_path": str(history_path),
        "summary": summary,
        "analysis_time_index": analysis_time_idx,
        "analysis_step": analysis_step,
        "observables": observables,
        "local_fields": local_fields_summary,
        "u1": u1_summary,
        "su2": su2_summary,
        "qsd_variance": qsd_metrics,
        "lyapunov": lyapunov_summary,
        "fractal_set": fractal_set_metrics,
        "wilson_loops": wilson_metrics,
        "wilson_timeseries": wilson_timeseries if analysis_cfg.build_fractal_set else None,
        "string_tension": string_tension,
        "particle_observables": particle_metrics,
        "analysis_config": asdict(analysis_cfg),
        "obs_params": obs_params,
        "companion_epsilon": epsilon_companion,
        "companion_epsilon_clone": epsilon_companion_clone,
        "epsilon_clone": epsilon_clone,
        "lambda_alg": lambda_alg,
        "plots": plot_paths,
        "plotting": {"enabled": HAS_MPL, "error": MPL_ERROR},
    }

    metrics_path = output_dir / f"{analysis_id}_metrics.json"
    metrics_path.write_text(json.dumps(_json_safe(analysis), indent=2, sort_keys=True))

    arrays = {
        "d_prime_bins": d_bins,
        "d_prime_correlation": d_corr,
        "d_prime_counts": d_counts,
        "r_prime_bins": r_bins,
        "r_prime_correlation": r_corr,
        "r_prime_counts": r_counts,
        "lyapunov_time": lyapunov["time"].cpu().numpy(),
        "lyapunov_total": lyapunov["V_total"].cpu().numpy(),
        "lyapunov_var_x": lyapunov["V_var_x"].cpu().numpy(),
        "lyapunov_var_v": lyapunov["V_var_v"].cpu().numpy(),
    }
    if wilson_timeseries is not None:
        arrays["wilson_time_index"] = wilson_timeseries["time_index"]
        arrays["wilson_action_mean"] = wilson_timeseries["action_mean"]

    # Add local field correlation arrays
    if local_correlations:
        for field_name, corr_data in local_correlations.items():
            arrays[f"{field_name}_bins"] = corr_data["bins"]
            arrays[f"{field_name}_correlation"] = corr_data["correlation"]
            arrays[f"{field_name}_counts"] = corr_data["counts"]

    if particle_arrays:
        arrays.update(particle_arrays)
    if string_tension is not None:
        arrays["string_tension_bins"] = string_tension["bin_centers"]
        arrays["string_tension_mean_w"] = string_tension["mean_w"]
        arrays["string_tension_counts"] = string_tension["counts"]

    arrays_path = output_dir / f"{analysis_id}_arrays.npz"
    np.savez(arrays_path, **arrays)

    _write_progress(
        progress_path,
        "complete",
        {
            "summary": summary,
            "metrics_path": str(metrics_path),
            "arrays_path": str(arrays_path),
        },
        state="complete",
    )

    print(summary)
    print("Saved:")
    print(f"  metrics: {metrics_path}")
    print(f"  arrays: {arrays_path}")

    # Print local field analysis summary when enabled
    if analysis_cfg.use_local_fields and local_correlations:
        print("\nLocal Field Correlation Analysis:")
        print(f"  density_sigma: {analysis_cfg.density_sigma}")
        print(f"  use_connected: {analysis_cfg.use_connected}")

        if analysis_cfg.use_connected:
            # Extended output for connected correlators
            print("\n  Field               | ξ (corr)  | R²       | r_zero   | +/- pts")
            print("  --------------------|-----------|----------|----------|--------")
            for field_name, corr_data in local_correlations.items():
                fit = corr_data["fit"]
                xi = fit.get("xi", 0.0)
                r2 = fit.get("r_squared", 0.0)
                r_zero = fit.get("r_zero", 0.0)
                n_pos = fit.get("n_positive", 0)
                n_neg = fit.get("n_negative", 0)
                print(f"  {field_name:<18} | {xi:>9.4f} | {r2:>8.4f} | {r_zero:>8.4f} | {n_pos}/{n_neg}")
        else:
            print("\n  Field               | ξ (corr. length) | R² (fit quality)")
            print("  --------------------|------------------|------------------")
            for field_name, corr_data in local_correlations.items():
                fit = corr_data["fit"]
                xi = fit.get("xi", 0.0)
                r2 = fit.get("r_squared", 0.0)
                print(f"  {field_name:<18} | {xi:>16.4f} | {r2:>16.4f}")

        print("\n  Comparison with companion-based d_prime:")
        print(f"    d_prime ξ: {d_fit.get('xi', 0.0):.4f}, R²: {d_fit.get('r_squared', 0.0):.4f}")
        print(f"    r_prime ξ: {r_fit.get('xi', 0.0):.4f}, R²: {r_fit.get('r_squared', 0.0):.4f}")

    if analysis_cfg.compute_particles and particle_metrics is not None:
        operators = particle_metrics.get("operators") or {}
        if operators:
            print("\nParticle Mass Estimates (Euclidean time):")
            print("  Operator    | mass       | R²        | fit              | samples")
            print("  ------------|------------|-----------|------------------|--------")
            for name, data in operators.items():
                fit = data.get("fit", {})
                mass = fit.get("mass", 0.0)
                r2 = fit.get("r_squared", 0.0)
                fit_mode = fit.get("fit_mode", "window")
                fit_start = int(fit.get("fit_start", analysis_cfg.particle_fit_start))
                fit_stop = int(fit.get("fit_stop", analysis_cfg.particle_fit_stop))
                fit_label = f"{fit_mode} {fit_start}-{fit_stop}"
                n_samples = data.get("n_samples", 0)
                print(
                    f"  {name:<11} | {mass:>10.4f} | {r2:>9.4f} | "
                    f"{fit_label:<16} | {n_samples:>7}"
                )


if __name__ == "__main__":
    main()
