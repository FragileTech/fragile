"""Comprehensive QSD Variance Parameter Sweep.

This module implements a full parameter grid sweep to test whether the QSD achieves
near-maximal variance (Var_h ≈ D_max²/2) required for O(N^{3/2}) edge budget.

**Experiment Design**:
- 3 potential types: unimodal (quadratic), multimodal (Gaussian mixture), periodic (Rastrigin)
- 5×5 α/β grid: exploitation vs diversity balance
- 3 swarm sizes: N ∈ {50, 100, 200}
- Total: 225 experiments

**Key Question**: Does multimodal landscape + high diversity achieve variance ratio ≥ 0.45?

See: VARIANCE_REQUIREMENT_ANALYSIS.md for theoretical background.
"""

from pathlib import Path
from typing import Callable

import matplotlib


matplotlib.use("Agg")  # Non-interactive backend
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from fragile.fractalai.bounds import TorchBounds
from fragile.fractalai.core.cloning import CloneOperator
from fragile.fractalai.core.companion_selection import CompanionSelection
from fragile.fractalai.core.euclidean_gas import EuclideanGas
from fragile.fractalai.core.fitness import FitnessOperator
from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.core.kinetic_operator import KineticOperator

# Import utilities from qsd_variance.py
from fragile.theory.qsd_variance import compute_hypocoercive_variance, estimate_edge_budget


# ==============================================================================
# Multimodal Potential Functions
# ==============================================================================


def create_quadratic_potential(alpha: float = 0.1) -> Callable:
    """Create simple quadratic confining potential U(x) = α|x|²/2.

    Args:
        alpha: Confinement strength

    Returns:
        Potential function U: R^d → R
    """

    def potential(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * alpha * (x**2).sum(dim=-1)

    return potential


def create_gaussian_mixture_potential(
    centers: list[tuple[float, ...]],
    depths: list[float],
    widths: list[float],
) -> Callable:
    """Create multimodal Gaussian mixture potential.

    U(x) = -Σ_k depth_k · exp(-||x - center_k||² / (2·width_k²))

    Multiple wells at specified centers. Negative sign makes wells (not peaks).

    Args:
        centers: List of mode centers [(x1, y1), (x2, y2), ...]
        depths: List of well depths [d1, d2, ...] (positive values)
        widths: List of basin widths [w1, w2, ...] (Gaussian std dev)

    Returns:
        Potential function U: R^d → R
    """
    centers_tensor = torch.tensor(centers, dtype=torch.float32)  # [K, d]
    depths_tensor = torch.tensor(depths, dtype=torch.float32)  # [K]
    widths_tensor = torch.tensor(widths, dtype=torch.float32)  # [K]

    def potential(x: torch.Tensor) -> torch.Tensor:
        # x: [N, d] or [d]
        if x.ndim == 1:
            x = x.unsqueeze(0)  # [1, d]

        # Compute ||x - center_k||² for all centers
        # x: [N, d], centers: [K, d] → diff: [N, K, d]
        diff = x.unsqueeze(1) - centers_tensor.unsqueeze(0).to(x.device)
        dist_sq = (diff**2).sum(dim=2)  # [N, K]

        # Gaussian well: -depth_k · exp(-dist²/(2·width²))
        wells = -depths_tensor.to(x.device) * torch.exp(
            -dist_sq / (2 * widths_tensor.to(x.device) ** 2)
        )  # [N, K]

        # Sum over all wells
        U = wells.sum(dim=1)  # [N]

        return U.squeeze()

    return potential


def create_rastrigin_potential(
    A: float = 10.0,
    omega: float = 2.0 * np.pi,
    offset: float = 0.0,
) -> Callable:
    """Create Rastrigin-like periodic potential with regular mode lattice.

    U(x) = offset + A·d + Σ_i [x_i² - A·cos(ω·x_i)]

    Regular lattice of local minima. In domain [-5, 5]^d, has ~9 modes per dimension.

    Args:
        A: Amplitude of periodic modulation (default: 10.0)
        omega: Frequency of modulation (default: 2π, period = 1.0)
        offset: Baseline offset (default: 0.0)

    Returns:
        Potential function U: R^d → R
    """

    def potential(x: torch.Tensor) -> torch.Tensor:
        # x: [N, d] or [d]
        d = x.shape[-1]

        # Quadratic part: Σ x_i²
        quadratic = (x**2).sum(dim=-1)  # [N] or scalar

        # Periodic part: -Σ A·cos(ω·x_i)
        periodic = -(A * torch.cos(omega * x)).sum(dim=-1)  # [N] or scalar

        # Total: offset + A·d + Σ[x_i² - A·cos(ω·x_i)]
        return offset + A * d + quadratic + periodic

    return potential


# ==============================================================================
# Variance Analysis from RunHistory
# ==============================================================================


def compute_variance_from_history(
    history: RunHistory,
    warmup_samples: int,
    lambda_v: float = 1.0,
) -> dict[str, float]:
    """Compute QSD variance statistics from RunHistory.

    Extracts final states (after warmup), computes variance metrics averaged over
    QSD samples.

    Args:
        history: Complete run history from EuclideanGas.run()
        warmup_samples: Number of initial samples to skip (burn-in)
        lambda_v: Velocity weight for hypocoercive variance

    Returns:
        dict with averaged variance metrics:
            - ratio_h_mean, ratio_h_std
            - var_h_mean, d_max_h_sq_mean
            - n_close_estimate, scaling_exponent
    """
    # Extract final states after warmup
    # history.x_final: [n_recorded, N, d]
    # history.v_final: [n_recorded, N, d]

    n_recorded = history.n_recorded
    if warmup_samples >= n_recorded:
        msg = f"warmup_samples ({warmup_samples}) >= n_recorded ({n_recorded})"
        raise ValueError(msg)

    # QSD samples: from warmup_samples to end
    x_qsd = history.x_final[warmup_samples:]  # [n_qsd, N, d]
    v_qsd = history.v_final[warmup_samples:]  # [n_qsd, N, d]

    n_qsd_samples = x_qsd.shape[0]

    # Compute variance metrics for each QSD sample
    samples_metrics = []

    for i in range(n_qsd_samples):
        # Create temporary SwarmState-like object
        # (compute_hypocoercive_variance expects SwarmState with .x and .v attributes)
        class TempState:
            def __init__(self, x, v):
                self.x = x
                self.v = v

        state = TempState(x_qsd[i], v_qsd[i])
        metrics = compute_hypocoercive_variance(state, lambda_v=lambda_v)
        samples_metrics.append(metrics)

    # Average over QSD samples
    ratio_h_samples = [m["ratio_h"] for m in samples_metrics]
    var_h_samples = [m["var_h"] for m in samples_metrics]
    d_max_h_sq_samples = [m["d_max_h_sq"] for m in samples_metrics]

    ratio_h_mean = np.mean(ratio_h_samples)
    ratio_h_std = np.std(ratio_h_samples)
    var_h_mean = np.mean(var_h_samples)
    d_max_h_sq_mean = np.mean(d_max_h_sq_samples)

    # Estimate edge budget
    N = history.N
    d_close_threshold = np.sqrt(d_max_h_sq_mean / N)  # d_close = D_max/√N
    n_close_estimate = estimate_edge_budget(
        var_h_mean,
        d_max_h_sq_mean,
        d_close_threshold,
        N,
    )

    # Scaling exponent: log(N_close) / log(N)
    scaling_exponent = (
        np.log(n_close_estimate) / np.log(N) if N > 1 and n_close_estimate > 0 else 0.0
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


# ==============================================================================
# Single Experiment Runner
# ==============================================================================


def run_single_parameter_experiment(
    N: int,
    d: int,
    potential: Callable,
    potential_name: str,
    alpha_fit: float,
    beta_fit: float,
    n_steps_warmup: int = 3000,
    n_steps_sample: int = 500,
    record_every: int = 25,
    lambda_alg: float = 1.0,
    sigma_x: float = 0.1,
    eta: float = 0.1,
    device: str = "cpu",
) -> dict:
    """Run single experiment with specified parameters.

    Args:
        N: Number of walkers
        d: Spatial dimension
        potential: Potential function U(x)
        potential_name: Name for logging
        alpha_fit: Reward exponent (exploitation)
        beta_fit: Diversity exponent (exploration)
        n_steps_warmup: Warmup steps to reach QSD
        n_steps_sample: Sampling steps for QSD statistics
        record_every: Recording interval
        lambda_alg: Velocity weight in algorithmic distance
        sigma_x: Cloning perturbation scale
        eta: Fitness floor parameter
        device: "cpu" or "cuda"

    Returns:
        dict with experiment results
    """
    # Create bounds
    bounds = TorchBounds.from_tuples([(-10.0, 10.0) for _ in range(d)])

    # Create operators
    kinetic_op = KineticOperator(
        gamma=1.0,
        beta=2.0,
        delta_t=0.01,
        potential=potential,
        use_potential_force=False,  # Standard Euclidean Gas (no adaptive force)
    )

    companion_sel = CompanionSelection(
        method="cloning",
        epsilon=0.1,
        lambda_alg=lambda_alg,
    )

    cloning = CloneOperator(
        alpha_reward=1.0,
        sigma_x=sigma_x,
        lambda_alg=lambda_alg,
        alpha_rest=0.95,
    )

    fitness_op = FitnessOperator(
        alpha=alpha_fit,
        beta=beta_fit,
        eta=eta,
        lambda_alg=lambda_alg,
        sigma_min=1e-8,
        A=2.0,
        potential=potential,
        bounds=bounds,
    )

    # Create EuclideanGas
    gas = EuclideanGas(
        N=N,
        d=d,
        potential=potential,
        companion_selection=companion_sel,
        kinetic_op=kinetic_op,
        cloning=cloning,
        fitness_op=fitness_op,
        bounds=bounds,
        device=torch.device(device),
        dtype="float32",
        enable_cloning=True,
        enable_kinetic=True,
    )

    # Run simulation
    total_steps = n_steps_warmup + n_steps_sample
    history = gas.run(
        n_steps=total_steps,
        record_every=record_every,
    )

    # Analyze QSD variance
    warmup_samples = n_steps_warmup // record_every
    variance_metrics = compute_variance_from_history(
        history,
        warmup_samples=warmup_samples,
        lambda_v=1.0,
    )

    # Return results
    return {
        "potential": potential_name,
        "N": N,
        "d": d,
        "alpha_fit": alpha_fit,
        "beta_fit": beta_fit,
        "lambda_alg": lambda_alg,
        "sigma_x": sigma_x,
        "eta": eta,
        **variance_metrics,
    }


# ==============================================================================
# Full Parameter Grid Sweep
# ==============================================================================


def run_parameter_sweep(
    N_values: list[int] = [50, 100, 200],
    d: int = 2,
    alpha_values: list[float] = [0.1, 0.5, 1.0, 3.0, 10.0],
    beta_values: list[float] = [0.1, 0.5, 1.0, 3.0, 10.0],
    potential_configs: list[tuple[str, Callable]] | None = None,
    n_steps_warmup: int = 3000,
    n_steps_sample: int = 500,
    record_every: int = 25,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run full parameter grid sweep.

    Tests all combinations of:
    - Potentials (unimodal, multimodal, periodic)
    - Alpha/beta values (exploitation vs diversity)
    - Swarm sizes

    Args:
        N_values: Swarm sizes to test
        d: Spatial dimension
        alpha_values: Reward exponents (exploitation)
        beta_values: Diversity exponents (exploration)
        potential_configs: List of (name, potential_func) tuples
        n_steps_warmup: Warmup steps per experiment
        n_steps_sample: Sampling steps per experiment
        record_every: Recording interval
        output_dir: Directory for results (default: src/fragile/theory/)

    Returns:
        DataFrame with all experiment results
    """
    if output_dir is None:
        output_dir = Path(__file__).parent

    if potential_configs is None:
        # Default potential configurations
        potential_configs = [
            ("quadratic", create_quadratic_potential(alpha=0.1)),
            (
                "gaussian_4mode",
                create_gaussian_mixture_potential(
                    centers=[(-5.0, -5.0), (-5.0, 5.0), (5.0, -5.0), (5.0, 5.0)],
                    depths=[1.0, 1.0, 1.0, 1.0],
                    widths=[1.5, 1.5, 1.5, 1.5],
                ),
            ),
            ("rastrigin", create_rastrigin_potential(A=10.0, omega=2.0 * np.pi)),
        ]

    # Compute total experiments
    n_experiments = len(N_values) * len(alpha_values) * len(beta_values) * len(potential_configs)

    print("=" * 70)
    print("QSD Variance Parameter Sweep")
    print("=" * 70)
    print(f"Total experiments: {n_experiments}")
    print(f"  Potentials: {len(potential_configs)}")
    print(f"  N values: {N_values}")
    print(f"  Alpha values: {alpha_values}")
    print(f"  Beta values: {beta_values}")
    print(
        f"  Grid: {len(alpha_values)} × {len(beta_values)} = {len(alpha_values) * len(beta_values)}"
    )
    print(f"\nEstimated runtime: ~{n_experiments * 2.5 / 60:.1f} hours")
    print("=" * 70)

    results = []
    experiment_id = 0

    # Run grid sweep
    with tqdm(total=n_experiments, desc="Parameter sweep") as pbar:
        for potential_name, potential in potential_configs:
            for N in N_values:
                for alpha in alpha_values:
                    for beta in beta_values:
                        experiment_id += 1

                        # Update progress description
                        pbar.set_description(f"{potential_name} N={N} α={alpha:.1f} β={beta:.1f}")

                        # Run experiment
                        try:
                            result = run_single_parameter_experiment(
                                N=N,
                                d=d,
                                potential=potential,
                                potential_name=potential_name,
                                alpha_fit=alpha,
                                beta_fit=beta,
                                n_steps_warmup=n_steps_warmup,
                                n_steps_sample=n_steps_sample,
                                record_every=record_every,
                            )

                            results.append(result)

                        except Exception as e:
                            print(f"\nError in experiment {experiment_id}: {e}")
                            # Store failed experiment with NaN values
                            results.append({
                                "potential": potential_name,
                                "N": N,
                                "alpha_fit": alpha,
                                "beta_fit": beta,
                                "ratio_h_mean": np.nan,
                                "error": str(e),
                            })

                        pbar.update(1)

                        # Save intermediate results every 10 experiments
                        if experiment_id % 10 == 0:
                            df_intermediate = pd.DataFrame(results)
                            df_intermediate.to_csv(
                                output_dir / "qsd_variance_sweep_partial.csv",
                                index=False,
                            )

    # Create final DataFrame
    df_results = pd.DataFrame(results)

    # Save complete results
    output_file = output_dir / "qsd_variance_sweep_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")

    return df_results


# ==============================================================================
# Main Entry Point
# ==============================================================================


def main():
    """Run comprehensive parameter sweep."""
    results = run_parameter_sweep(
        N_values=[50, 100, 200],
        d=2,
        alpha_values=[0.1, 0.5, 1.0, 3.0, 10.0],
        beta_values=[0.1, 0.5, 1.0, 3.0, 10.0],
        n_steps_warmup=3000,
        n_steps_sample=500,
        record_every=25,
    )

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    for potential in results["potential"].unique():
        df_pot = results[results["potential"] == potential]
        print(f"\n{potential.upper()}:")
        print(f"  Mean variance ratio: {df_pot['ratio_h_mean'].mean():.4f}")
        print(f"  Max variance ratio:  {df_pot['ratio_h_mean'].max():.4f}")
        print(f"  Min variance ratio:  {df_pot['ratio_h_mean'].min():.4f}")

        # Find best (α, β) combination
        best_idx = df_pot["ratio_h_mean"].idxmax()
        best_row = df_pot.loc[best_idx]
        print(
            f"  Best (α={best_row['alpha_fit']:.1f}, β={best_row['beta_fit']:.1f}): "
            f"ratio={best_row['ratio_h_mean']:.4f}"
        )

    # Check if any achieved high variance
    high_variance = results[results["ratio_h_mean"] >= 0.45]
    if len(high_variance) > 0:
        print("\n✅ HIGH VARIANCE ACHIEVED (ratio ≥ 0.45):")
        print(high_variance[["potential", "N", "alpha_fit", "beta_fit", "ratio_h_mean"]])
        print("\n→ O(N^{3/2}) edge budget IS achievable!")
    else:
        print("\n❌ NO HIGH VARIANCE (ratio < 0.45) in any configuration")
        print("→ O(N^{3/2}) edge budget likely UNPROVABLE")

    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
