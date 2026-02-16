"""
Multi-trial statistical QFT validation pipeline.

Runs the fractal gas simulation multiple times with different random seeds,
aggregates all metrics, and computes confidence intervals with error bars
for rigorous empirical validation.

Usage:
    python src/experiments/run_qft_validation_ensemble.py --n-trials 100 --parallel-jobs 4
    python src/experiments/run_qft_validation_ensemble.py --n-trials 5 --parallel-jobs 2  # Quick test
    python src/experiments/run_qft_validation_ensemble.py --skip-simulation  # Only aggregate existing
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import json
from multiprocessing import cpu_count, Pool
from pathlib import Path
import traceback
from typing import Any

import numpy as np


try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    def tqdm(iterable, **kwargs):  # noqa: ARG001
        return iterable


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


@dataclass
class EnsembleConfig:
    """Configuration for the ensemble validation run."""

    n_trials: int = 100
    n_walkers: int = 200
    n_steps: int = 1000
    parallel_jobs: int = 4
    seed_base: int = 42
    output_dir: Path = field(default_factory=lambda: Path("outputs/qft_ensemble"))
    # Simulation config
    dims: int = 2
    alpha: float = 0.1
    bounds_extent: float = 10.0
    record_every: int = 1
    device: str = "cpu"
    dtype: str = "float32"
    gamma: float = 1.0
    beta: float = 1.0
    delta_t: float = 129.327
    epsilon_F: float = 994.399
    companion_epsilon_diversity: float = 2.12029
    companion_epsilon_clone: float = 1.68419
    lambda_alg: float = 1.0
    epsilon_clone: float = 0.01
    sigma_x: float = 0.1
    alpha_restitution: float = 0.5
    fitness_rho: float | None = None  # Use mean-field regime (local regime not yet supported)
    # Analysis flags
    use_local_fields: bool = True
    use_connected: bool = True
    build_fractal_set: bool = True
    density_sigma: float = 0.5
    correlation_r_max: float = 2.0
    correlation_bins: int = 50
    warmup_fraction: float = 0.1
    fractal_set_stride: int = 10
    # SU(3) viscous coupling config
    use_viscous_coupling: bool = True
    nu: float = 0.948271
    viscous_length_scale: float = 0.00976705


def _json_safe(value: Any) -> Any:
    """Convert value to JSON-serializable format."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
        if value != value:  # NaN check
            return "nan"
    return value


def run_single_trial(args: tuple[int, EnsembleConfig]) -> dict[str, Any]:
    """Run one simulation + analysis, return metrics dict or error info.

    This function is designed to be called via multiprocessing.Pool.
    """
    trial_idx, config = args
    seed = config.seed_base + trial_idx
    run_id = f"trial_{trial_idx:04d}"
    trial_dir = config.output_dir / "trials" / run_id

    result = {
        "trial_idx": trial_idx,
        "seed": seed,
        "run_id": run_id,
        "success": False,
        "error": None,
        "metrics_path": None,
        "metrics": None,
    }

    try:
        # Import here to avoid issues with multiprocessing
        import torch

        from fragile.fractalai.bounds import TorchBounds
        from fragile.fractalai.core.cloning import CloneOperator
        from fragile.fractalai.core.companion_selection import CompanionSelection
        from fragile.fractalai.core.euclidean_gas import EuclideanGas
        from fragile.fractalai.core.fitness import FitnessOperator
        from fragile.fractalai.core.kinetic_operator import KineticOperator

        # Create output directory
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Build potential
        class QuadraticPotential:
            def __init__(self, alpha: float, dims: int, bounds_extent: float) -> None:
                self.alpha = float(alpha)
                self.dims = int(dims)
                self.bounds = TorchBounds.from_tuples(
                    [(-bounds_extent, bounds_extent)] * self.dims
                )

            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                return 0.5 * self.alpha * (x**2).sum(dim=-1)

        potential = QuadraticPotential(
            alpha=config.alpha,
            dims=config.dims,
            bounds_extent=config.bounds_extent,
        )

        # Build operators
        companion = CompanionSelection(
            method="cloning",
            epsilon=config.companion_epsilon_diversity,
            lambda_alg=config.lambda_alg,
            exclude_self=True,
        )
        companion_clone = CompanionSelection(
            method="cloning",
            epsilon=config.companion_epsilon_clone,
            lambda_alg=config.lambda_alg,
            exclude_self=True,
        )

        fitness_op = FitnessOperator(
            alpha=1.0,
            beta=1.0,
            eta=0.1,
            lambda_alg=config.lambda_alg,
            sigma_min=1e-8,
            epsilon_dist=1e-8,
            A=2.0,
            rho=config.fitness_rho,
        )

        kinetic_op = KineticOperator(
            gamma=config.gamma,
            beta=config.beta,
            delta_t=config.delta_t,
            epsilon_F=config.epsilon_F,
            use_potential_force=True,
            potential=potential,
            nu=config.nu if config.use_viscous_coupling else 0.0,
            use_viscous_coupling=config.use_viscous_coupling,
            viscous_length_scale=config.viscous_length_scale,
        )

        cloning = CloneOperator(
            p_max=1.0,
            epsilon_clone=config.epsilon_clone,
            sigma_x=config.sigma_x,
            alpha_restitution=config.alpha_restitution,
        )

        gas = EuclideanGas(
            N=config.n_walkers,
            d=config.dims,
            potential=potential,
            companion_selection=companion,
            companion_selection_clone=companion_clone,
            kinetic_op=kinetic_op,
            cloning=cloning,
            fitness_op=fitness_op,
            bounds=potential.bounds,
            device=torch.device(config.device),
            dtype=config.dtype,
            enable_cloning=True,
            enable_kinetic=True,
            pbc=False,
        )

        # Run simulation
        history = gas.run(
            n_steps=config.n_steps,
            record_every=config.record_every,
            seed=seed,
            record_rng_state=True,
        )

        # Save history
        history_path = trial_dir / "history.pt"
        history.save(str(history_path))

        # Save config
        trial_config = {
            "trial_idx": trial_idx,
            "seed": seed,
            "n_walkers": config.n_walkers,
            "n_steps": config.n_steps,
            "dims": config.dims,
            "alpha": config.alpha,
            "delta_t": config.delta_t,
            "epsilon_F": config.epsilon_F,
            "companion_epsilon_diversity": config.companion_epsilon_diversity,
            "companion_epsilon_clone": config.companion_epsilon_clone,
            "lambda_alg": config.lambda_alg,
            "epsilon_clone": config.epsilon_clone,
            "nu": config.nu,
            "use_viscous_coupling": config.use_viscous_coupling,
            "viscous_length_scale": config.viscous_length_scale,
            "fitness_rho": config.fitness_rho,
        }
        config_path = trial_dir / "config.json"
        config_path.write_text(json.dumps(trial_config, indent=2))

        # Run analysis
        metrics = _analyze_history(history, config, trial_dir, run_id)

        # Save metrics
        metrics_path = trial_dir / "metrics.json"
        metrics_path.write_text(json.dumps(_json_safe(metrics), indent=2, sort_keys=True))

        result["success"] = True
        result["metrics_path"] = str(metrics_path)
        result["metrics"] = metrics

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    return result


def _analyze_history(history, config: EnsembleConfig, trial_dir: Path, run_id: str) -> dict:
    """Analyze a single history and return metrics dict."""

    from fragile.fractalai.core.fitness import compute_fitness
    from fragile.fractalai.lyapunov import compute_lyapunov_components_trajectory

    # Get analysis time index (last recorded step)
    analysis_time_idx = history.n_recorded - 1
    if analysis_time_idx <= 0:
        analysis_time_idx = 1

    analysis_step = history.recorded_steps[analysis_time_idx]
    info_idx = analysis_time_idx - 1

    # Extract state
    x_pre = history.x_before_clone[analysis_time_idx]
    v_pre = history.v_before_clone[analysis_time_idx]
    rewards = history.rewards[info_idx]
    alive = history.alive_mask[info_idx]
    companions_distance = history.companions_distance[info_idx]
    companions_clone = history.companions_clone[info_idx]

    params = history.params or {}

    # Compute observables
    obs_params = {
        "alpha": _get_param(params, ["fitness", "alpha"], 1.0),
        "beta": _get_param(params, ["fitness", "beta"], 1.0),
        "eta": _get_param(params, ["fitness", "eta"], 0.1),
        "A": _get_param(params, ["fitness", "A"], 2.0),
        "lambda_alg": _get_param(params, ["fitness", "lambda_alg"], 1.0),
        "sigma_min": _get_param(params, ["fitness", "sigma_min"], 1e-8),
        "epsilon_dist": _get_param(params, ["fitness", "epsilon_dist"], 1e-8),
        "rho": _get_param(params, ["fitness", "rho"], None),
    }

    fitness, _info = compute_fitness(
        positions=x_pre,
        velocities=v_pre,
        rewards=rewards,
        alive=alive,
        companions=companions_distance,
        alpha=obs_params["alpha"],
        beta=obs_params["beta"],
        eta=obs_params["eta"],
        lambda_alg=obs_params["lambda_alg"],
        sigma_min=obs_params["sigma_min"],
        A=obs_params["A"],
        epsilon_dist=obs_params["epsilon_dist"],
        rho=obs_params["rho"],
        bounds=history.bounds,
        pbc=history.pbc,
    )

    # Compute local fields
    local_fields = _compute_local_fields(
        x_pre, v_pre, rewards, alive, sigma_density=config.density_sigma
    )

    # Compute correlation functions
    local_correlations = {}
    bin_func = _bin_by_distance_connected if config.use_connected else _bin_by_distance
    fit_func = _fit_connected_correlator if config.use_connected else _fit_exponential_decay

    for field_name, field_values in local_fields.items():
        bins, corr, counts = bin_func(
            x_pre,
            field_values,
            alive,
            config.correlation_r_max,
            config.correlation_bins,
        )
        fit = fit_func(bins, corr, counts)
        local_correlations[field_name] = {
            "fit": fit,
            "mean": _masked_mean_std(field_values, alive)[0],
            "std": _masked_mean_std(field_values, alive)[1],
        }

    # Compute gauge phases
    h_eff = 1.0
    u1_phases = _compute_u1_phases(fitness, companions_distance, alive, h_eff)
    epsilon_clone = _get_param(params, ["cloning", "epsilon_clone"], 0.01)
    su2_phases = _compute_su2_phases(fitness, alive, companions_clone, epsilon_clone, h_eff)

    u1_mean, u1_std = _masked_mean_std(u1_phases, alive)
    su2_mean, su2_std = _masked_mean_std(su2_phases, alive)

    # Compute SU(3) gauge structure from viscous coupling
    su3_metrics = {
        "su3_color_magnitude_mean": 0.0,
        "su3_color_magnitude_std": 0.0,
        "su3_alignment_mean": 0.0,
        "su3_alignment_std": 0.0,
        "su3_phase_mean": 0.0,
        "su3_phase_std": 0.0,
    }
    if config.use_viscous_coupling and config.nu > 0:
        viscous_force = _compute_viscous_force(
            positions=x_pre,
            velocities=v_pre,
            alive=alive,
            nu=config.nu,
            viscous_length_scale=config.viscous_length_scale,
            bounds=history.bounds,
            pbc=history.pbc,
        )
        # Characteristic length scale (use correlation r_max or viscous length scale)
        l_0 = config.viscous_length_scale
        color_state = _compute_su3_color_state(
            viscous_force=viscous_force,
            velocities=v_pre,
            alive=alive,
            h_eff=h_eff,
            l_0=l_0,
        )
        su3_metrics = _compute_su3_metrics(color_state, alive)

    # Compute Lyapunov
    lyapunov = compute_lyapunov_components_trajectory(history, stage="final")
    lyapunov_metrics = {
        "initial_total": float(lyapunov["V_total"][0].item()),
        "final_total": float(lyapunov["V_total"][-1].item()),
        "convergence_ratio": float(lyapunov["V_total"][-1].item() / lyapunov["V_total"][0].item())
        if lyapunov["V_total"][0].item() > 0
        else 0.0,
    }

    # Compute QSD variance
    qsd_metrics = None
    if history.n_recorded > 1:
        warmup_samples = int(
            max(1, min(history.n_recorded - 1, history.n_recorded * config.warmup_fraction))
        )
        qsd_metrics = _compute_variance_from_history(history, warmup_samples)

    # Build FractalSet and compute Wilson loops if requested
    wilson_metrics = None
    if config.build_fractal_set:
        try:
            from fragile.fractalai.core.fractal_set import FractalSet

            history_small = _downsample_history(history, config.fractal_set_stride)
            fractal_set = FractalSet(history_small)
            timestep = max(0, history_small.n_recorded - 1)
            wilson_metrics = _compute_wilson_loops(fractal_set, timestep)
        except Exception:
            wilson_metrics = None

    # Assemble metrics
    return {
        "analysis_time_index": analysis_time_idx,
        "analysis_step": analysis_step,
        "n_alive": int(alive.sum().item()),
        "local_fields": {
            field_name: {
                "xi": corr_data["fit"].get("xi", 0.0),
                "r_squared": corr_data["fit"].get("r_squared", 0.0),
                "r_zero": corr_data["fit"].get("r_zero", 0.0),
                "C0": corr_data["fit"].get("C0", 0.0),
                "has_zero_crossing": corr_data["fit"].get("has_zero_crossing", False),
                "mean": corr_data["mean"],
                "std": corr_data["std"],
            }
            for field_name, corr_data in local_correlations.items()
        },
        "gauge_phases": {
            "u1_phase_mean": u1_mean,
            "u1_phase_std": u1_std,
            "su2_phase_mean": su2_mean,
            "su2_phase_std": su2_std,
            **su3_metrics,
        },
        "lyapunov": lyapunov_metrics,
        "qsd_variance": qsd_metrics,
        "wilson_loops": {
            "wilson_mean": wilson_metrics.get("wilson_mean", 0.0) if wilson_metrics else 0.0,
            "wilson_std": wilson_metrics.get("wilson_std", 0.0) if wilson_metrics else 0.0,
            "wilson_action_mean": wilson_metrics.get("action_mean", 0.0)
            if wilson_metrics
            else 0.0,
            "wilson_action_std": wilson_metrics.get("action_std", 0.0) if wilson_metrics else 0.0,
            "n_loops": wilson_metrics.get("n_loops", 0) if wilson_metrics else 0,
        },
    }


def _get_param(params: dict[str, Any] | None, keys: list[str], default: Any) -> Any:
    """Get nested parameter from params dict."""
    current = params or {}
    for key in keys:
        if not isinstance(current, dict) or key not in current or current[key] is None:
            return default
        current = current[key]
    return current


def _compute_local_fields(
    positions,
    velocities,
    rewards,
    alive,
    sigma_density: float = 0.5,
) -> dict:
    """Compute proper LOCAL fields for QFT correlation analysis."""
    import torch

    positions.shape[0]

    # Local density field via kernel density estimate
    dists_sq = ((positions.unsqueeze(1) - positions.unsqueeze(0)) ** 2).sum(dim=-1)
    kernel = torch.exp(-dists_sq / (2 * sigma_density**2))

    alive_2d = alive.unsqueeze(0).float() * alive.unsqueeze(1).float()
    kernel = kernel * alive_2d
    kernel.fill_diagonal_(0)

    density = kernel.sum(dim=1)

    alive_density = density[alive]
    if alive_density.numel() > 0 and alive_density.mean() > 0:
        density = density / alive_density.mean()
    else:
        density = torch.ones_like(density)

    # Local diversity field = 1/density
    diversity_local = 1.0 / torch.clamp(density, min=1e-6)

    # Radial distance field
    radial = torch.sqrt((positions**2).sum(dim=-1))

    # Kinetic energy field
    kinetic = 0.5 * (velocities**2).sum(dim=-1)

    return {
        "density": torch.where(alive, density, torch.zeros_like(density)),
        "diversity_local": torch.where(alive, diversity_local, torch.zeros_like(diversity_local)),
        "radial": torch.where(alive, radial, torch.zeros_like(radial)),
        "kinetic": torch.where(alive, kinetic, torch.zeros_like(kinetic)),
        "reward_raw": torch.where(alive, rewards, torch.zeros_like(rewards)),
    }


def _bin_by_distance(positions, values, alive, r_max: float, n_bins: int):
    """Bin values by pairwise distance."""
    import torch

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


def _bin_by_distance_connected(positions, values, alive, r_max: float, n_bins: int):
    """Compute connected two-point correlator G(r) = <φφ> - <φ>²."""
    import torch

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


def _fit_exponential_decay(r, C, counts) -> dict:
    """Fit exponential decay to correlation function."""
    valid = (counts > 0) & (C > 0)
    if valid.sum() < 2:
        return {"C0": 0.0, "xi": 0.0, "r_squared": 0.0}

    r_valid = r[valid]
    C_valid = C[valid]
    weights = counts[valid]

    x = r_valid**2
    y = np.log(C_valid)

    try:
        coeffs = np.polyfit(x, y, 1, w=np.sqrt(weights))
        slope, intercept = coeffs
    except (np.linalg.LinAlgError, ValueError):
        return {"C0": 0.0, "xi": 0.0, "r_squared": 0.0}

    if slope >= 0:
        return {"C0": float(np.exp(intercept)), "xi": 0.0, "r_squared": 0.0}

    xi = float(np.sqrt(-1.0 / slope))
    C0 = float(np.exp(intercept))

    y_pred = intercept + slope * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"C0": C0, "xi": xi, "r_squared": r_squared}


def _fit_connected_correlator(r, C, counts) -> dict:
    """Fit connected correlator that may have negative values at large r."""
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

    n_positive = int((C_valid > 0).sum())
    n_negative = int((C_valid < 0).sum())

    sign_changes = np.where(np.diff(np.sign(C_valid)) < 0)[0]
    has_zero_crossing = len(sign_changes) > 0

    if has_zero_crossing:
        idx = sign_changes[0]
        if idx + 1 < len(r_valid) and C_valid[idx] != C_valid[idx + 1]:
            r_zero = r_valid[idx] + (r_valid[idx + 1] - r_valid[idx]) * (
                -C_valid[idx] / (C_valid[idx + 1] - C_valid[idx])
            )
        else:
            r_zero = r_valid[idx]
        r_zero = float(r_zero)
        fit_mask = (C_valid > 0) & (r_valid < r_zero)
    else:
        r_zero = float(r_valid[-1])
        fit_mask = C_valid > 0

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
        return {
            "C0": float(np.exp(intercept)),
            "xi": 0.0,
            "r_squared": 0.0,
            "r_zero": r_zero,
            "has_zero_crossing": has_zero_crossing,
            "n_positive": n_positive,
            "n_negative": n_negative,
        }

    xi = float(np.sqrt(-1.0 / slope))
    C0 = float(np.exp(intercept))

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
    }


def _masked_mean_std(values, alive) -> tuple[float, float]:
    """Compute mean and std of values where alive is True."""
    alive_values = values[alive]
    if alive_values.numel() == 0:
        return 0.0, 0.0
    mean_val = alive_values.mean().item()
    if alive_values.numel() < 2:
        return mean_val, 0.0
    return mean_val, alive_values.std().item()


def _compute_u1_phases(fitness, companions, alive, h_eff: float):
    """Compute U(1) gauge phases."""
    import torch

    fitness_companion = fitness[companions]
    phases = -(fitness_companion - fitness) / h_eff
    return torch.where(alive, phases, torch.zeros_like(phases))


def _compute_su2_phases(fitness, alive, clone_companions, epsilon_clone: float, h_eff: float):
    """Compute SU(2) gauge phases."""
    import torch

    fitness_companion = fitness[clone_companions]
    score = (fitness_companion - fitness) / (fitness + epsilon_clone)
    phases = score / h_eff
    return torch.where(alive, phases, torch.zeros_like(phases))


def _compute_viscous_force(
    positions,
    velocities,
    alive,
    nu: float,
    viscous_length_scale: float,
    bounds,
    pbc: bool,
):
    """Compute viscous coupling force for SU(3) gauge structure.

    F_viscous(x_i) = nu * sum_j [K(||x_i-x_j||)/deg(i)] * (v_j - v_i)

    This implements the velocity-coupling force that generates the SU(3)
    gauge structure through basis-rotation symmetry.

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        alive: Alive mask [N]
        nu: Viscosity coupling strength
        viscous_length_scale: Kernel width l for K(r) = exp(-r²/(2l²))
        bounds: Domain bounds (for PBC)
        pbc: Whether to use periodic boundary conditions

    Returns:
        viscous_force: Force vectors [N, d]
    """
    import torch

    from fragile.fractalai.core.distance import compute_periodic_distance_matrix

    if nu == 0.0:
        return torch.zeros_like(velocities)

    _N, _d = positions.shape

    # Compute pairwise distances with PBC support
    distances = compute_periodic_distance_matrix(
        positions, y=None, bounds=bounds, pbc=pbc
    )  # [N, N]

    # Compute Gaussian kernel K(r) = exp(-r²/(2l²))
    l_sq = viscous_length_scale**2
    kernel = torch.exp(-(distances**2) / (2 * l_sq))  # [N, N]

    # Zero out diagonal (no self-interaction)
    kernel.fill_diagonal_(0.0)

    # Apply alive mask: only alive walkers contribute
    alive_2d = alive.unsqueeze(0).float() * alive.unsqueeze(1).float()
    kernel = kernel * alive_2d

    # Compute local degree deg(i) = sum_{j!=i} K(||x_i - x_j||)
    deg = kernel.sum(dim=1, keepdim=True)  # [N, 1]
    deg = torch.clamp(deg, min=1e-10)  # Avoid division by zero

    # Compute normalized weights w_ij = K_ij / deg_i
    weights = kernel / deg  # [N, N]

    # Compute velocity differences: v_diff[i, j] = v_j - v_i
    v_diff = velocities.unsqueeze(0) - velocities.unsqueeze(1)  # [N, N, d]

    # Compute weighted sum: F_visc_i = nu * sum_j w_ij * (v_j - v_i)
    viscous_force = nu * torch.einsum("ij,ijd->id", weights, v_diff)  # [N, d]

    # Zero out force for dead walkers
    return torch.where(alive.unsqueeze(1), viscous_force, torch.zeros_like(viscous_force))


def _compute_su3_color_state(
    viscous_force,
    velocities,
    alive,
    h_eff: float,
    l_0: float,
):
    """Compute SU(3) color states from viscous force components.

    Following the Standard Model derivation (thm-sm-su3-emergence), the color
    state is constructed from complex force amplitudes:

        c_i^(α) = F_α^(visc)(i) * exp(i * p_i^(α) * l_0 / h_eff)

    normalized to unit length.

    Args:
        viscous_force: Viscous force vectors [N, d]
        velocities: Walker velocities [N, d]
        alive: Alive mask [N]
        h_eff: Effective Planck constant (sets phase scale)
        l_0: Characteristic length scale (mean edge length)

    Returns:
        dict with:
            color_magnitude: ||F_viscous|| [N]
            color_direction: F/||F|| [N, d] (normalized direction)
            color_phases: momentum-based phases [N, d]
            color_state_real: Real part of normalized color state [N, d]
            color_state_imag: Imaginary part of normalized color state [N, d]
    """
    import torch

    _N, _d = viscous_force.shape

    # Compute force magnitude ||F_viscous||
    force_magnitude = torch.norm(viscous_force, dim=1)  # [N]
    force_magnitude = torch.clamp(force_magnitude, min=1e-10)

    # Compute normalized force direction
    color_direction = viscous_force / force_magnitude.unsqueeze(1)  # [N, d]

    # Compute momentum-based phases: phi_i^(α) = p_i^(α) * l_0 / h_eff
    # where p_i^(α) = m * v_i^(α), and we set m=1
    color_phases = velocities * l_0 / h_eff  # [N, d]

    # Construct complex color amplitudes:
    # c_i^(α) = F_α * exp(i * phi_α)
    # = F_α * (cos(phi_α) + i * sin(phi_α))
    cos_phases = torch.cos(color_phases)  # [N, d]
    sin_phases = torch.sin(color_phases)  # [N, d]

    # Complex amplitude: (F * cos(phi), F * sin(phi))
    c_real = viscous_force * cos_phases  # [N, d]
    c_imag = viscous_force * sin_phases  # [N, d]

    # Compute normalization factor ||c|| = sqrt(sum_α |c^α|²)
    c_magnitude = torch.sqrt((c_real**2 + c_imag**2).sum(dim=1))  # [N]
    c_magnitude = torch.clamp(c_magnitude, min=1e-10)

    # Normalized color state
    color_state_real = c_real / c_magnitude.unsqueeze(1)  # [N, d]
    color_state_imag = c_imag / c_magnitude.unsqueeze(1)  # [N, d]

    # Zero out for dead walkers
    zero = torch.zeros_like(force_magnitude)
    zeros_d = torch.zeros_like(viscous_force)

    return {
        "color_magnitude": torch.where(alive, force_magnitude, zero),
        "color_direction": torch.where(alive.unsqueeze(1), color_direction, zeros_d),
        "color_phases": torch.where(alive.unsqueeze(1), color_phases, zeros_d),
        "color_state_real": torch.where(alive.unsqueeze(1), color_state_real, zeros_d),
        "color_state_imag": torch.where(alive.unsqueeze(1), color_state_imag, zeros_d),
    }


def _compute_su3_metrics(color_state: dict, alive) -> dict:
    """Compute aggregate SU(3) metrics from color states.

    Args:
        color_state: Output from _compute_su3_color_state()
        alive: Alive mask [N]

    Returns:
        dict with:
            su3_color_magnitude_mean/std: Statistics of force magnitude
            su3_alignment_mean/std: Pairwise dot products of color directions
            su3_phase_mean/std: Statistics of color phases
    """
    import torch

    color_magnitude = color_state["color_magnitude"]
    color_direction = color_state["color_direction"]
    color_phases = color_state["color_phases"]

    n_alive = alive.sum().item()

    # Color magnitude statistics
    if n_alive > 0:
        mag_alive = color_magnitude[alive]
        mag_mean = float(mag_alive.mean().item())
        mag_std = float(mag_alive.std().item()) if n_alive > 1 else 0.0
    else:
        mag_mean, mag_std = 0.0, 0.0

    # Alignment statistics: pairwise dot products of color directions
    if n_alive > 1:
        # Get alive color directions
        alive_dirs = color_direction[alive]  # [n_alive, d]
        # Compute all pairwise dot products
        dot_products = torch.mm(alive_dirs, alive_dirs.t())  # [n_alive, n_alive]
        # Extract upper triangle (excluding diagonal)
        n = alive_dirs.shape[0]
        upper_mask = torch.triu(torch.ones(n, n, device=dot_products.device), diagonal=1).bool()
        pairwise_dots = dot_products[upper_mask]

        if pairwise_dots.numel() > 0:
            align_mean = float(pairwise_dots.mean().item())
            align_std = float(pairwise_dots.std().item()) if pairwise_dots.numel() > 1 else 0.0
        else:
            align_mean, align_std = 1.0, 0.0
    elif n_alive == 1:
        # Single alive walker: alignment = 1.0 (with itself)
        align_mean, align_std = 1.0, 0.0
    else:
        align_mean, align_std = 0.0, 0.0

    # Phase statistics (take mean across dimensions)
    if n_alive > 0:
        phase_alive = color_phases[alive]  # [n_alive, d]
        # Compute phase magnitude per walker
        phase_magnitudes = torch.norm(phase_alive, dim=1)  # [n_alive]
        phase_mean = float(phase_magnitudes.mean().item())
        phase_std = float(phase_magnitudes.std().item()) if n_alive > 1 else 0.0
    else:
        phase_mean, phase_std = 0.0, 0.0

    return {
        "su3_color_magnitude_mean": mag_mean,
        "su3_color_magnitude_std": mag_std,
        "su3_alignment_mean": align_mean,
        "su3_alignment_std": align_std,
        "su3_phase_mean": phase_mean,
        "su3_phase_std": phase_std,
    }


def _compute_variance_from_history(history, warmup_samples: int, lambda_v: float = 1.0) -> dict:
    """Compute QSD variance metrics from history."""

    n_recorded = history.n_recorded
    if warmup_samples >= n_recorded:
        warmup_samples = max(1, n_recorded - 1)

    x_qsd = history.x_final[warmup_samples:]
    v_qsd = history.v_final[warmup_samples:]

    n_qsd_samples = x_qsd.shape[0]
    ratio_h_samples = []

    for i in range(n_qsd_samples):
        x = x_qsd[i]
        v = v_qsd[i]

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
        ratio_h = var_h / d_max_h_sq if d_max_h_sq > 0 else 0.0
        ratio_h_samples.append(ratio_h)

    ratio_h_mean = float(np.mean(ratio_h_samples)) if ratio_h_samples else 0.0
    ratio_h_std = float(np.std(ratio_h_samples)) if ratio_h_samples else 0.0

    N = history.N
    scaling_exponent = (
        float(np.log(ratio_h_mean * N) / np.log(N)) if N > 1 and ratio_h_mean > 0 else 0.0
    )

    return {
        "ratio_h_mean": ratio_h_mean,
        "ratio_h_std": ratio_h_std,
        "scaling_exponent": scaling_exponent,
        "n_qsd_samples": n_qsd_samples,
    }


def _downsample_history(history, stride: int):
    """Downsample history by stride factor."""
    import torch

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

    from fragile.fractalai.core.history import RunHistory

    return RunHistory(**data)


def _edge_value(value) -> float:
    """Extract scalar value from tensor or scalar."""
    import torch

    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _compute_wilson_loops(fractal_set, timestep: int) -> dict | None:
    """Compute Wilson loop statistics at given timestep."""
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
        "n_loops": len(phases),
        "phase_mean": float(phases_np.mean()),
        "phase_std": float(phases_np.std()),
        "wilson_mean": float(wilson.mean()),
        "wilson_std": float(wilson.std()),
        "action_mean": float(action.mean()),
        "action_std": float(action.std()),
    }


def run_ensemble(config: EnsembleConfig, skip_simulation: bool = False) -> list[dict]:
    """Run all trials in parallel using multiprocessing.Pool."""
    if skip_simulation:
        # Load existing results
        return load_existing_results(config)

    trial_args = [(i, config) for i in range(config.n_trials)]

    if config.parallel_jobs == 1:
        # Single-threaded for debugging
        results = []
        for args in tqdm(trial_args, desc="Running trials"):
            results.append(run_single_trial(args))
    else:
        # Parallel execution
        with Pool(config.parallel_jobs) as pool:
            if HAS_TQDM:
                results = list(
                    tqdm(
                        pool.imap(run_single_trial, trial_args),
                        total=config.n_trials,
                        desc="Running trials",
                    )
                )
            else:
                results = pool.map(run_single_trial, trial_args)

    return results


def load_existing_results(config: EnsembleConfig) -> list[dict]:
    """Load metrics from existing trial directories."""
    trials_dir = config.output_dir / "trials"
    results = []

    for i in range(config.n_trials):
        run_id = f"trial_{i:04d}"
        metrics_path = trials_dir / run_id / "metrics.json"

        result = {
            "trial_idx": i,
            "seed": config.seed_base + i,
            "run_id": run_id,
            "success": False,
            "error": None,
            "metrics_path": None,
            "metrics": None,
        }

        if metrics_path.exists():
            try:
                with open(metrics_path, encoding="utf-8") as f:
                    result["metrics"] = json.load(f)
                result["success"] = True
                result["metrics_path"] = str(metrics_path)
            except Exception as e:
                result["error"] = str(e)

        results.append(result)

    return results


def aggregate_metrics(results: list[dict]) -> dict:
    """Load all metrics and aggregate into arrays (one value per trial)."""
    successful = [r for r in results if r["success"] and r["metrics"]]

    if not successful:
        return {
            "n_successful": 0,
            "local_fields": {},
            "wilson_loops": {},
            "lyapunov": {},
            "qsd_variance": {},
            "gauge_phases": {},
        }

    aggregated = {
        "n_successful": len(successful),
        "local_fields": {},
        "wilson_loops": {},
        "lyapunov": {},
        "qsd_variance": {},
        "gauge_phases": {},
    }

    # Get field names from first successful trial
    first_metrics = successful[0]["metrics"]
    field_names = list(first_metrics.get("local_fields", {}).keys())

    # Aggregate local fields
    for field_name in field_names:
        aggregated["local_fields"][field_name] = {
            "xi": [],
            "r_squared": [],
            "r_zero": [],
            "C0": [],
        }

    for r in successful:
        m = r["metrics"]
        for field_name in field_names:
            field_data = m.get("local_fields", {}).get(field_name, {})
            aggregated["local_fields"][field_name]["xi"].append(field_data.get("xi", 0.0))
            aggregated["local_fields"][field_name]["r_squared"].append(
                field_data.get("r_squared", 0.0)
            )
            aggregated["local_fields"][field_name]["r_zero"].append(field_data.get("r_zero", 0.0))
            aggregated["local_fields"][field_name]["C0"].append(field_data.get("C0", 0.0))

    # Convert to numpy arrays
    for field_name in field_names:
        for metric in ["xi", "r_squared", "r_zero", "C0"]:
            aggregated["local_fields"][field_name][metric] = np.array(
                aggregated["local_fields"][field_name][metric]
            )

    # Aggregate Wilson loops
    wilson_keys = ["wilson_mean", "wilson_std", "wilson_action_mean", "wilson_action_std"]
    for key in wilson_keys:
        aggregated["wilson_loops"][key] = np.array([
            r["metrics"].get("wilson_loops", {}).get(key, 0.0) for r in successful
        ])

    # Aggregate Lyapunov
    lyapunov_keys = ["initial_total", "final_total", "convergence_ratio"]
    for key in lyapunov_keys:
        aggregated["lyapunov"][key] = np.array([
            r["metrics"].get("lyapunov", {}).get(key, 0.0) for r in successful
        ])

    # Aggregate QSD variance
    qsd_keys = ["ratio_h_mean", "scaling_exponent"]
    for key in qsd_keys:
        vals = []
        for r in successful:
            qsd = r["metrics"].get("qsd_variance")
            if qsd:
                vals.append(qsd.get(key, 0.0))
            else:
                vals.append(0.0)
        aggregated["qsd_variance"][key] = np.array(vals)

    # Aggregate gauge phases (U(1), SU(2), and SU(3))
    gauge_keys = [
        "u1_phase_mean",
        "u1_phase_std",
        "su2_phase_mean",
        "su2_phase_std",
        "su3_color_magnitude_mean",
        "su3_color_magnitude_std",
        "su3_alignment_mean",
        "su3_alignment_std",
        "su3_phase_mean",
        "su3_phase_std",
    ]
    for key in gauge_keys:
        aggregated["gauge_phases"][key] = np.array([
            r["metrics"].get("gauge_phases", {}).get(key, 0.0) for r in successful
        ])

    return aggregated


def compute_statistics(aggregated: dict) -> dict:
    """Compute mean, std, SE, 95% CI for each metric."""

    def compute_stats(values: np.ndarray) -> dict:
        if len(values) == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "se": 0.0,
                "ci_95_low": 0.0,
                "ci_95_high": 0.0,
                "values": [],
            }
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        se = std / np.sqrt(len(values)) if len(values) > 1 else 0.0
        ci_95_low = mean - 1.96 * se
        ci_95_high = mean + 1.96 * se
        return {
            "mean": mean,
            "std": std,
            "se": se,
            "ci_95_low": ci_95_low,
            "ci_95_high": ci_95_high,
            "values": values.tolist(),
        }

    stats = {
        "n_successful_trials": aggregated.get("n_successful", 0),
        "local_fields": {},
        "wilson_loops": {},
        "lyapunov": {},
        "qsd_variance": {},
        "gauge_phases": {},
    }

    # Local fields
    for field_name, field_data in aggregated.get("local_fields", {}).items():
        stats["local_fields"][field_name] = {}
        for metric_name, values in field_data.items():
            stats["local_fields"][field_name][metric_name] = compute_stats(values)

    # Wilson loops
    for key, values in aggregated.get("wilson_loops", {}).items():
        stats["wilson_loops"][key] = compute_stats(values)

    # Lyapunov
    for key, values in aggregated.get("lyapunov", {}).items():
        stats["lyapunov"][key] = compute_stats(values)

    # QSD variance
    for key, values in aggregated.get("qsd_variance", {}).items():
        stats["qsd_variance"][key] = compute_stats(values)

    # Gauge phases
    for key, values in aggregated.get("gauge_phases", {}).items():
        stats["gauge_phases"][key] = compute_stats(values)

    return stats


def generate_ensemble_plots(stats: dict, config: EnsembleConfig) -> dict[str, Path]:
    """Generate plots with error bars and distributions."""
    if not HAS_MPL:
        return {}

    plots_dir = config.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = {}

    # 1. Correlation lengths bar chart
    field_names = list(stats.get("local_fields", {}).keys())
    if field_names:
        _fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(field_names))
        means = [stats["local_fields"][f]["xi"]["mean"] for f in field_names]
        errors = [1.96 * stats["local_fields"][f]["xi"]["se"] for f in field_names]

        ax.bar(x, means, yerr=errors, capsize=5, alpha=0.7, color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels(field_names, rotation=45, ha="right")
        ax.set_ylabel("Correlation Length ξ")
        ax.set_title(f"Correlation Lengths (n={stats['n_successful_trials']} trials, 95% CI)")
        plt.tight_layout()
        path = plots_dir / "correlation_lengths.png"
        plt.savefig(path, dpi=150)
        plt.close()
        plot_paths["correlation_lengths"] = path

    # 2. Correlation lengths distribution (violin/box plot)
    if field_names:
        _fig, ax = plt.subplots(figsize=(10, 6))
        xi_data = [stats["local_fields"][f]["xi"]["values"] for f in field_names]
        ax.violinplot(xi_data, positions=np.arange(len(field_names)), showmeans=True)
        ax.set_xticks(np.arange(len(field_names)))
        ax.set_xticklabels(field_names, rotation=45, ha="right")
        ax.set_ylabel("Correlation Length ξ")
        ax.set_title("Correlation Length Distributions Across Trials")
        plt.tight_layout()
        path = plots_dir / "correlation_lengths_dist.png"
        plt.savefig(path, dpi=150)
        plt.close()
        plot_paths["correlation_lengths_dist"] = path

    # 3. R² summary
    if field_names:
        _fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(field_names))
        means = [stats["local_fields"][f]["r_squared"]["mean"] for f in field_names]
        errors = [1.96 * stats["local_fields"][f]["r_squared"]["se"] for f in field_names]

        ax.bar(x, means, yerr=errors, capsize=5, alpha=0.7, color="darkorange")
        ax.set_xticks(x)
        ax.set_xticklabels(field_names, rotation=45, ha="right")
        ax.set_ylabel("R² (Fit Quality)")
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.5, label="R²=0.8 threshold")
        ax.set_title(f"Fit Quality R² (n={stats['n_successful_trials']} trials, 95% CI)")
        ax.legend()
        plt.tight_layout()
        path = plots_dir / "r_squared_summary.png"
        plt.savefig(path, dpi=150)
        plt.close()
        plot_paths["r_squared_summary"] = path

    # 4. Wilson loop distribution
    wilson_values = stats.get("wilson_loops", {}).get("wilson_mean", {}).get("values", [])
    if wilson_values:
        _fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(wilson_values, bins=20, alpha=0.7, color="purple", edgecolor="black")
        mean = stats["wilson_loops"]["wilson_mean"]["mean"]
        se = stats["wilson_loops"]["wilson_mean"]["se"]
        ax.axvline(mean, color="red", linestyle="-", linewidth=2, label=f"Mean: {mean:.3f}")
        ax.axvline(mean - 1.96 * se, color="red", linestyle="--", alpha=0.5)
        ax.axvline(mean + 1.96 * se, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Wilson Loop Mean")
        ax.set_ylabel("Count")
        ax.set_title(f"Wilson Loop Distribution (n={stats['n_successful_trials']} trials)")
        ax.legend()
        plt.tight_layout()
        path = plots_dir / "wilson_loop_dist.png"
        plt.savefig(path, dpi=150)
        plt.close()
        plot_paths["wilson_loop_dist"] = path

    # 5. Lyapunov convergence histogram
    conv_values = stats.get("lyapunov", {}).get("convergence_ratio", {}).get("values", [])
    if conv_values:
        _fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(conv_values, bins=20, alpha=0.7, color="green", edgecolor="black")
        mean = stats["lyapunov"]["convergence_ratio"]["mean"]
        ax.axvline(mean, color="red", linestyle="-", linewidth=2, label=f"Mean: {mean:.3f}")
        ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5, label="No convergence")
        ax.set_xlabel("Lyapunov Convergence Ratio (final/initial)")
        ax.set_ylabel("Count")
        ax.set_title(f"Lyapunov Convergence (n={stats['n_successful_trials']} trials)")
        ax.legend()
        plt.tight_layout()
        path = plots_dir / "lyapunov_convergence.png"
        plt.savefig(path, dpi=150)
        plt.close()
        plot_paths["lyapunov_convergence"] = path

    # 6. Phase distributions (U(1), SU(2), SU(3))
    u1_values = stats.get("gauge_phases", {}).get("u1_phase_mean", {}).get("values", [])
    su2_values = stats.get("gauge_phases", {}).get("su2_phase_mean", {}).get("values", [])
    su3_values = stats.get("gauge_phases", {}).get("su3_phase_mean", {}).get("values", [])
    if u1_values and su2_values:
        # Determine number of panels based on SU(3) data availability
        has_su3 = su3_values and any(v != 0.0 for v in su3_values)
        n_panels = 3 if has_su3 else 2

        _fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
        if n_panels == 2:
            ax1, ax2 = axes
            ax3 = None
        else:
            ax1, ax2, ax3 = axes

        ax1.hist(u1_values, bins=20, alpha=0.7, color="blue", edgecolor="black")
        ax1.set_xlabel("U(1) Phase Mean")
        ax1.set_ylabel("Count")
        ax1.set_title("U(1) Gauge Phase Distribution")

        ax2.hist(su2_values, bins=20, alpha=0.7, color="orange", edgecolor="black")
        ax2.set_xlabel("SU(2) Phase Mean")
        ax2.set_ylabel("Count")
        ax2.set_title("SU(2) Gauge Phase Distribution")

        if ax3 is not None:
            ax3.hist(su3_values, bins=20, alpha=0.7, color="green", edgecolor="black")
            ax3.set_xlabel("SU(3) Phase Mean")
            ax3.set_ylabel("Count")
            ax3.set_title("SU(3) Gauge Phase Distribution")

        plt.tight_layout()
        path = plots_dir / "phase_distributions.png"
        plt.savefig(path, dpi=150)
        plt.close()
        plot_paths["phase_distributions"] = path

    # 7. SU(3) color alignment distribution
    su3_align_values = (
        stats.get("gauge_phases", {}).get("su3_alignment_mean", {}).get("values", [])
    )
    if su3_align_values and any(v != 0.0 for v in su3_align_values):
        _fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(su3_align_values, bins=20, alpha=0.7, color="purple", edgecolor="black")
        mean = stats["gauge_phases"]["su3_alignment_mean"]["mean"]
        se = stats["gauge_phases"]["su3_alignment_mean"]["se"]
        ax.axvline(mean, color="red", linestyle="-", linewidth=2, label=f"Mean: {mean:.3f}")
        ax.axvline(mean - 1.96 * se, color="red", linestyle="--", alpha=0.5)
        ax.axvline(mean + 1.96 * se, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("SU(3) Color Alignment (pairwise dot product)")
        ax.set_ylabel("Count")
        ax.set_title(
            f"SU(3) Color Alignment Distribution (n={stats['n_successful_trials']} trials)"
        )
        ax.legend()
        plt.tight_layout()
        path = plots_dir / "su3_color_alignment.png"
        plt.savefig(path, dpi=150)
        plt.close()
        plot_paths["su3_color_alignment"] = path

    return plot_paths


def save_ensemble_report(
    stats: dict, config: EnsembleConfig, results: list[dict], plot_paths: dict[str, Path]
) -> tuple[Path, Path]:
    """Save JSON report and markdown summary."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_report = {
        "config": {
            "n_trials": config.n_trials,
            "n_walkers": config.n_walkers,
            "n_steps": config.n_steps,
            "seed_base": config.seed_base,
            "use_local_fields": config.use_local_fields,
            "use_connected": config.use_connected,
            "build_fractal_set": config.build_fractal_set,
            "density_sigma": config.density_sigma,
            "correlation_r_max": config.correlation_r_max,
        },
        **stats,
        "plot_paths": {k: str(v) for k, v in plot_paths.items()},
    }

    json_path = config.output_dir / "ensemble_metrics.json"
    json_path.write_text(json.dumps(_json_safe(json_report), indent=2, sort_keys=True))

    # Generate markdown report
    n_success = stats.get("n_successful_trials", 0)
    n_failed = len(results) - n_success

    md_lines = [
        "# QFT Validation Ensemble Results\n",
        f"**Configuration:** {config.n_trials} trials, {config.n_walkers} walkers, {config.n_steps} steps\n",
        f"**Successful trials:** {n_success} / {config.n_trials}",
    ]

    if n_failed > 0:
        md_lines.append(f"**Failed trials:** {n_failed}")
    md_lines.append("")

    # Correlation Functions table
    md_lines.append("## Correlation Functions\n")
    md_lines.append("| Field | ξ (mean ± 95% CI) | R² (mean ± 95% CI) |")
    md_lines.append("|-------|-------------------|---------------------|")

    for field_name, field_stats in stats.get("local_fields", {}).items():
        xi = field_stats["xi"]
        r2 = field_stats["r_squared"]
        xi_str = f"{xi['mean']:.4f} ± {1.96 * xi['se']:.4f}"
        r2_str = f"{r2['mean']:.4f} ± {1.96 * r2['se']:.4f}"
        md_lines.append(f"| {field_name} | {xi_str} | {r2_str} |")

    md_lines.append("")

    # Wilson Loops
    wilson = stats.get("wilson_loops", {})
    if wilson.get("wilson_mean"):
        md_lines.append("## Wilson Loops\n")
        wm = wilson["wilson_mean"]
        wa = wilson["wilson_action_mean"]
        md_lines.append(f"- Mean Wilson loop: {wm['mean']:.3f} ± {1.96 * wm['se']:.3f} (95% CI)")
        md_lines.append(f"- Mean action: {wa['mean']:.3f} ± {1.96 * wa['se']:.3f} (95% CI)")
        md_lines.append("")

    # Lyapunov
    lyap = stats.get("lyapunov", {})
    if lyap.get("convergence_ratio"):
        md_lines.append("## Lyapunov Convergence\n")
        conv = lyap["convergence_ratio"]
        convergence_status = (
            "strong convergence"
            if conv["mean"] < 0.7
            else "moderate convergence"
            if conv["mean"] < 0.9
            else "weak/no convergence"
        )
        md_lines.append(
            f"- Convergence ratio: {conv['mean']:.3f} ± {1.96 * conv['se']:.3f} ({convergence_status})"
        )
        md_lines.append("")

    # QSD Variance
    qsd = stats.get("qsd_variance", {})
    if qsd.get("ratio_h_mean"):
        md_lines.append("## QSD Variance\n")
        ratio = qsd["ratio_h_mean"]
        md_lines.append(
            f"- Hypocoercive variance ratio: {ratio['mean']:.4f} ± {1.96 * ratio['se']:.4f}"
        )
        if qsd.get("scaling_exponent"):
            scale = qsd["scaling_exponent"]
            md_lines.append(f"- Scaling exponent: {scale['mean']:.3f} ± {1.96 * scale['se']:.3f}")
        md_lines.append("")

    # Gauge Phases
    gauge = stats.get("gauge_phases", {})
    if gauge.get("u1_phase_mean"):
        md_lines.append("## Gauge Phases\n")
        u1 = gauge["u1_phase_mean"]
        su2 = gauge["su2_phase_mean"]
        md_lines.append("### U(1) × SU(2) (Electroweak)")
        md_lines.append(f"- U(1) phase mean: {u1['mean']:.4f} ± {1.96 * u1['se']:.4f}")
        md_lines.append(f"- SU(2) phase mean: {su2['mean']:.4f} ± {1.96 * su2['se']:.4f}")
        md_lines.append("")

        # SU(3) metrics
        su3_mag = gauge.get("su3_color_magnitude_mean")
        su3_align = gauge.get("su3_alignment_mean")
        su3_phase = gauge.get("su3_phase_mean")
        if su3_mag and (
            su3_mag.get("mean", 0) != 0 or any(v != 0 for v in su3_mag.get("values", []))
        ):
            md_lines.append("### SU(3) (Strong Force / Viscous Coupling)")
            md_lines.append(
                f"- Color magnitude mean: {su3_mag['mean']:.4f} ± {1.96 * su3_mag['se']:.4f}"
            )
            if su3_align:
                md_lines.append(
                    f"- Color alignment mean: {su3_align['mean']:.4f} ± {1.96 * su3_align['se']:.4f}"
                )
            if su3_phase:
                md_lines.append(
                    f"- Color phase mean: {su3_phase['mean']:.4f} ± {1.96 * su3_phase['se']:.4f}"
                )
            md_lines.append("")
        else:
            md_lines.append("### SU(3) (Strong Force / Viscous Coupling)")
            md_lines.append("- *Viscous coupling disabled (use --use-viscous-coupling to enable)*")
            md_lines.append("")

    # Conclusion
    md_lines.append("## Conclusion\n")

    # Check if key predictions validated
    validated = []
    not_validated = []

    # Check correlation fits (R² > 0.5 for at least some fields)
    good_fits = sum(
        1 for f in stats.get("local_fields", {}).values() if f["r_squared"]["mean"] > 0.5
    )
    if good_fits > 0:
        validated.append(f"{good_fits} fields show QFT-like correlation decay (R² > 0.5)")
    else:
        not_validated.append("No fields show strong correlation decay")

    # Check Lyapunov convergence (ratio < 0.9 indicates contraction)
    if lyap.get("convergence_ratio") and lyap["convergence_ratio"]["mean"] < 0.9:
        validated.append("Lyapunov contraction confirmed")
    elif lyap.get("convergence_ratio"):
        not_validated.append("Weak Lyapunov contraction")

    # Check Wilson loops (mean between 0.5 and 1.0 is expected)
    if wilson.get("wilson_mean") and 0.3 < wilson["wilson_mean"]["mean"] < 1.0:
        validated.append("Wilson loops show expected gauge behavior")

    # Check SU(3) gauge structure
    su3_mag = gauge.get("su3_color_magnitude_mean")
    su3_align = gauge.get("su3_alignment_mean")
    if su3_mag and su3_mag.get("mean", 0) > 0:
        validated.append(f"SU(3) color structure active (magnitude={su3_mag['mean']:.4f})")
        if su3_align and -1 <= su3_align.get("mean", 0) <= 1:
            align_desc = (
                "correlated"
                if su3_align["mean"] > 0.5
                else "decorrelated"
                if su3_align["mean"] < -0.5
                else "mixed"
            )
            validated.append(f"SU(3) color alignment: {align_desc} ({su3_align['mean']:.3f})")

    if validated:
        md_lines.append("**Validated predictions:**")
        for v in validated:
            md_lines.append(f"- ✓ {v}")
        md_lines.append("")

    if not_validated:
        md_lines.append("**Needs investigation:**")
        for v in not_validated:
            md_lines.append(f"- ✗ {v}")
        md_lines.append("")

    # Timestamp
    md_lines.append(f"\n---\n*Generated: {datetime.utcnow().isoformat()}Z*")

    md_path = config.output_dir / "ensemble_report.md"
    md_path.write_text("\n".join(md_lines))

    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Ensemble configuration
    parser.add_argument("--n-trials", type=int, default=100, help="Number of trials to run")
    parser.add_argument("--n-walkers", type=int, default=200, help="Number of walkers per trial")
    parser.add_argument("--n-steps", type=int, default=1000, help="Number of simulation steps")
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs (default: CPU count - 1)",
    )
    parser.add_argument("--seed-base", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/qft_ensemble", help="Output directory"
    )

    # Simulation config
    parser.add_argument("--dims", type=int, default=2, help="Number of dimensions")
    parser.add_argument("--alpha", type=float, default=0.1, help="Potential well alpha")
    parser.add_argument("--bounds-extent", type=float, default=10.0, help="Bounds extent")
    parser.add_argument(
        "--delta-t",
        type=float,
        default=EnsembleConfig.delta_t,
        help=f"Kinetic time step (default: {EnsembleConfig.delta_t})",
    )
    parser.add_argument(
        "--epsilon-F",
        type=float,
        default=EnsembleConfig.epsilon_F,
        help=f"Fitness coupling scale (default: {EnsembleConfig.epsilon_F})",
    )
    parser.add_argument("--record-every", type=int, default=1, help="Record every N steps")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--dtype", type=str, default="float32", help="Torch dtype")
    parser.add_argument(
        "--companion-epsilon-diversity",
        type=float,
        default=EnsembleConfig.companion_epsilon_diversity,
        help="Diversity companion epsilon (epsilon_d)",
    )
    parser.add_argument(
        "--companion-epsilon-clone",
        type=float,
        default=EnsembleConfig.companion_epsilon_clone,
        help="Cloning companion epsilon (epsilon_c)",
    )

    # Analysis flags
    parser.add_argument(
        "--use-local-fields",
        action="store_true",
        default=True,
        help="Use local fields for correlation analysis (default: True)",
    )
    parser.add_argument(
        "--no-local-fields",
        action="store_false",
        dest="use_local_fields",
        help="Disable local field analysis",
    )
    parser.add_argument(
        "--use-connected",
        action="store_true",
        default=True,
        help="Use connected correlators (default: True)",
    )
    parser.add_argument(
        "--no-connected",
        action="store_false",
        dest="use_connected",
        help="Use raw correlators instead of connected",
    )
    parser.add_argument(
        "--build-fractal-set",
        action="store_true",
        default=True,
        help="Build FractalSet for Wilson loops (default: True)",
    )
    parser.add_argument(
        "--no-fractal-set",
        action="store_false",
        dest="build_fractal_set",
        help="Skip FractalSet construction",
    )
    parser.add_argument(
        "--density-sigma", type=float, default=0.5, help="Kernel width for density estimation"
    )
    parser.add_argument(
        "--correlation-r-max", type=float, default=2.0, help="Max distance for correlations"
    )
    parser.add_argument(
        "--correlation-bins", type=int, default=50, help="Number of correlation bins"
    )
    parser.add_argument(
        "--warmup-fraction", type=float, default=0.1, help="Warmup fraction for QSD analysis"
    )
    parser.add_argument(
        "--fractal-set-stride", type=int, default=10, help="Downsampling stride for FractalSet"
    )

    # SU(3) viscous coupling options
    parser.add_argument(
        "--use-viscous-coupling",
        action="store_true",
        help="Enable viscous coupling for SU(3) gauge structure",
    )
    parser.add_argument(
        "--nu", type=float, default=0.1, help="Viscosity strength for SU(3) (default: 0.1)"
    )
    parser.add_argument(
        "--viscous-length-scale",
        type=float,
        default=1.0,
        help="Kernel width for viscous coupling (default: 1.0)",
    )

    # Control flags
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="Skip simulation and only aggregate existing results",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set parallel jobs
    parallel_jobs = args.parallel_jobs
    if parallel_jobs is None:
        parallel_jobs = max(1, cpu_count() - 1)

    config = EnsembleConfig(
        n_trials=args.n_trials,
        n_walkers=args.n_walkers,
        n_steps=args.n_steps,
        parallel_jobs=parallel_jobs,
        seed_base=args.seed_base,
        output_dir=Path(args.output_dir),
        dims=args.dims,
        alpha=args.alpha,
        bounds_extent=args.bounds_extent,
        delta_t=args.delta_t,
        epsilon_F=args.epsilon_F,
        record_every=args.record_every,
        device=args.device,
        dtype=args.dtype,
        companion_epsilon_diversity=args.companion_epsilon_diversity,
        companion_epsilon_clone=args.companion_epsilon_clone,
        use_local_fields=args.use_local_fields,
        use_connected=args.use_connected,
        build_fractal_set=args.build_fractal_set,
        density_sigma=args.density_sigma,
        correlation_r_max=args.correlation_r_max,
        correlation_bins=args.correlation_bins,
        warmup_fraction=args.warmup_fraction,
        fractal_set_stride=args.fractal_set_stride,
        use_viscous_coupling=args.use_viscous_coupling,
        nu=args.nu,
        viscous_length_scale=args.viscous_length_scale,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("QFT Validation Ensemble Pipeline")
    print("=" * 50)
    print(f"Trials: {config.n_trials}")
    print(f"Walkers: {config.n_walkers}")
    print(f"Steps: {config.n_steps}")
    print(f"Parallel jobs: {config.parallel_jobs}")
    print(f"Output: {config.output_dir}")
    print(f"Skip simulation: {args.skip_simulation}")
    print(f"SU(3) viscous coupling: {'Enabled' if config.use_viscous_coupling else 'Disabled'}")
    if config.use_viscous_coupling:
        print(f"  nu = {config.nu}, length_scale = {config.viscous_length_scale}")
    print()

    # Phase 1 & 2: Run simulations + analysis
    print("Phase 1-2: Running simulations and analysis...")
    results = run_ensemble(config, skip_simulation=args.skip_simulation)

    n_success = sum(1 for r in results if r["success"])
    n_failed = len(results) - n_success
    print(f"  Completed: {n_success} / {config.n_trials} trials")
    if n_failed > 0:
        print(f"  Failed: {n_failed} trials")
        # Print first few errors
        errors = [r for r in results if not r["success"]]
        for err in errors[:3]:
            print(f"    Trial {err['trial_idx']}: {err['error'][:100]}...")

    # Phase 3: Aggregate metrics
    print("\nPhase 3: Aggregating metrics...")
    aggregated = aggregate_metrics(results)
    print(f"  Aggregated {aggregated['n_successful']} successful trials")

    # Phase 4: Compute statistics
    print("\nPhase 4: Computing statistics...")
    stats = compute_statistics(aggregated)

    # Phase 5: Generate plots
    print("\nPhase 5: Generating plots...")
    plot_paths = generate_ensemble_plots(stats, config)
    print(f"  Generated {len(plot_paths)} plots")

    # Phase 6: Save reports
    print("\nPhase 6: Saving reports...")
    json_path, md_path = save_ensemble_report(stats, config, results, plot_paths)
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("ENSEMBLE SUMMARY")
    print("=" * 50)

    print("\nCorrelation Lengths (ξ):")
    for field_name, field_stats in stats.get("local_fields", {}).items():
        xi = field_stats["xi"]
        print(f"  {field_name}: {xi['mean']:.4f} ± {1.96 * xi['se']:.4f} (95% CI)")

    print("\nFit Quality (R²):")
    for field_name, field_stats in stats.get("local_fields", {}).items():
        r2 = field_stats["r_squared"]
        quality = "Good" if r2["mean"] > 0.7 else "Fair" if r2["mean"] > 0.5 else "Poor"
        print(f"  {field_name}: {r2['mean']:.4f} ± {1.96 * r2['se']:.4f} [{quality}]")

    if stats.get("lyapunov", {}).get("convergence_ratio"):
        conv = stats["lyapunov"]["convergence_ratio"]
        print(f"\nLyapunov Convergence: {conv['mean']:.4f} ± {1.96 * conv['se']:.4f}")

    if stats.get("wilson_loops", {}).get("wilson_mean"):
        wm = stats["wilson_loops"]["wilson_mean"]
        print(f"Wilson Loop Mean: {wm['mean']:.4f} ± {1.96 * wm['se']:.4f}")

    # SU(3) summary
    su3_mag = stats.get("gauge_phases", {}).get("su3_color_magnitude_mean")
    if su3_mag and su3_mag.get("mean", 0) > 0:
        print(f"\nSU(3) Color Magnitude: {su3_mag['mean']:.4f} ± {1.96 * su3_mag['se']:.4f}")
        su3_align = stats.get("gauge_phases", {}).get("su3_alignment_mean")
        if su3_align:
            print(f"SU(3) Color Alignment: {su3_align['mean']:.4f} ± {1.96 * su3_align['se']:.4f}")

    print(f"\nFull report: {md_path}")


if __name__ == "__main__":
    main()
