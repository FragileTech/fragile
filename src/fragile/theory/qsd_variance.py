"""QSD Variance Analysis Experiments.

This module implements experiments to measure the quasi-stationary distribution's
variance level and determine whether the hierarchical clustering proof's edge-counting
strategy is feasible.

**Research Question**: Does the QSD achieve near-maximal variance (Var_h ≈ D_max²/2)?

**Context**: The Phase-Space Packing Lemma (lem-phase-space-packing in 03_cloning.md)
provides an edge budget:

    N_close ≤ C(K,2) · (D_max² - 2·Var_h) / (D_max² - d_close²)

For O(N^{3/2}) edges with d_close = D_max/√N, this requires:

    Var_h ≥ D_max²/2 - O(D_max²/√N)

This is "near-maximal" variance. Framework analysis (06_convergence.md) shows QSD
has equilibrium variance Var_x^QSD ≤ C_x/κ_x, which is a balance point, not a maximum.

**This experiment measures**: Var_h^QSD / D_max² ratio empirically.

**Decision criteria**:
- If ratio ≈ 0.45-0.5 → High variance mechanism exists, O(N^{3/2}) feasible
- If ratio ≈ 0.1-0.3 → Moderate equilibrium, O(N^{3/2}) unprovable, edge budget is O(N²)

See: VARIANCE_REQUIREMENT_ANALYSIS.md for detailed mathematical analysis.
"""

from pathlib import Path

import matplotlib


matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from fragile.bounds import TorchBounds
from fragile.core.cloning import CloneOperator
from fragile.core.companion_selection import CompanionSelection
from fragile.core.euclidean_gas import EuclideanGas, SwarmState
from fragile.core.fitness import FitnessOperator
from fragile.core.kinetic_operator import KineticOperator


def create_quadratic_potential(alpha: float = 0.1):
    """Create simple quadratic confining potential U(x) = α|x|²/2."""

    def potential(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * alpha * (x**2).sum(dim=-1)

    return potential


def compute_hypocoercive_variance(state: SwarmState, lambda_v: float = 1.0) -> dict[str, float]:
    """Compute hypocoercive variance and related metrics.

    Following 03_cloning.md § 6.4.1 (lem-phase-space-packing):

        Var_h(S_k) := Var_x(S_k) + λ_v · Var_v(S_k)

    where:
        - Var_x = (1/k) Σ ||x_i - x_mean||²
        - Var_v = (1/k) Σ ||v_i - v_mean||²
        - λ_v is the velocity weight (from Langevin parameters)

    Args:
        state: Current swarm state (positions, velocities)
        lambda_v: Velocity weight parameter

    Returns:
        dict with variance metrics and ratios
    """
    # Extract positions and velocities
    x = state.x  # [N, d]
    v = state.v  # [N, d]

    # Compute means
    x_mean = x.mean(dim=0, keepdim=True)
    v_mean = v.mean(dim=0, keepdim=True)

    # Compute variances (per-walker squared deviation, then mean)
    var_x = ((x - x_mean) ** 2).sum(dim=1).mean().item()
    var_v = ((v - v_mean) ** 2).sum(dim=1).mean().item()

    # Hypocoercive variance
    var_h = var_x + lambda_v * var_v

    # Compute domain diameters
    # D_x = max_i,j ||x_i - x_j||
    # D_v = max_i,j ||v_i - v_j||

    # Pairwise distances
    x_diff = x.unsqueeze(0) - x.unsqueeze(1)  # [N, N, d]
    v_diff = v.unsqueeze(0) - v.unsqueeze(1)  # [N, N, d]

    d_x = (x_diff**2).sum(dim=2).sqrt().max().item()
    d_v = (v_diff**2).sum(dim=2).sqrt().max().item()

    # Hypocoercive diameter: D_h² = D_x² + λ_alg · D_v²
    # Using λ_alg = λ_v for simplicity (can be different in general)
    d_max_h_sq = d_x**2 + lambda_v * d_v**2
    d_max_h = np.sqrt(d_max_h_sq)

    # Compute ratios
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


def estimate_edge_budget(var_h: float, d_max_sq: float, d_close: float, K: int) -> float:
    """Estimate N_close from Phase-Space Packing Lemma.

    Following lem-phase-space-packing (03_cloning.md lines 2420-2550):

        N_close ≤ C(K,2) · (D_max² - 2·Var_h) / (D_max² - d_close²)

    Args:
        var_h: Measured hypocoercive variance
        d_max_sq: Squared hypocoercive diameter
        d_close: Proximity threshold
        K: Number of walkers

    Returns:
        Estimated number of close pairs (upper bound)
    """
    numerator = d_max_sq - 2 * var_h
    denominator = d_max_sq - d_close**2

    if denominator <= 0:
        # d_close ≥ d_max: all pairs are close
        return K * (K - 1) / 2

    if numerator <= 0:
        # Var_h ≥ d_max²/2: very few close pairs
        return 0.0

    fraction = numerator / denominator
    return (K * (K - 1) / 2) * fraction


def run_single_experiment(
    N: int,
    d: int,
    n_steps_warmup: int = 5000,
    n_steps_sample: int = 1000,
    sample_interval: int = 50,
    device: str = "cpu",
) -> dict:
    """Run single QSD variance measurement experiment.

    Args:
        N: Number of walkers
        d: Dimensionality
        n_steps_warmup: Steps to reach QSD (burn-in)
        n_steps_sample: Steps for sampling QSD
        sample_interval: Sample every N steps
        device: "cpu" or "cuda"

    Returns:
        dict with statistics over QSD samples
    """
    print(f"\n{'=' * 60}")
    print(f"Experiment: N={N}, d={d}")
    print(f"{'=' * 60}")

    # Create components
    potential = create_quadratic_potential(alpha=0.1)
    bounds = TorchBounds.from_tuples([(-10.0, 10.0) for _ in range(d)])

    kinetic_op = KineticOperator(
        gamma=1.0,
        beta=2.0,  # Inverse temperature
        delta_t=0.01,  # Time step
        potential=potential,  # Required for base Langevin dynamics
        use_potential_force=False,  # Disable adaptive force (standard Euclidean Gas)
    )

    companion_sel = CompanionSelection(
        method="cloning",  # Standard cloning companion selection
        epsilon=0.1,  # Interaction range
        lambda_alg=1.0,  # Phase-space aware (balanced position+velocity)
    )

    cloning = CloneOperator(
        alpha_reward=1.0,
        sigma_x=0.1,
        lambda_alg=1.0,
        alpha_rest=0.95,
    )

    # Create fitness operator (required for EuclideanGas)
    fitness_op = FitnessOperator(
        potential=potential,
        bounds=bounds,
    )

    # Create EuclideanGas instance
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

    # Initialize
    state = gas.initialize_state()

    # Warmup: evolve to QSD
    print(f"\nWarmup: {n_steps_warmup} steps to reach QSD...")
    for _ in tqdm(range(n_steps_warmup), desc="Warmup"):
        _, state = gas.step(state)  # Returns (state_after_cloning, state_after_kinetic)

    print("Warmup complete. Now sampling QSD...")

    # Sample QSD
    samples = []
    n_steps_sample // sample_interval

    for step in tqdm(range(n_steps_sample), desc="Sampling"):
        _, state = gas.step(state)  # Returns (state_after_cloning, state_after_kinetic)

        if step % sample_interval == 0:
            metrics = compute_hypocoercive_variance(state, lambda_v=1.0)
            samples.append(metrics)

    # Compute statistics over samples
    ratio_h_samples = [s["ratio_h"] for s in samples]
    var_h_samples = [s["var_h"] for s in samples]
    d_max_h_sq_samples = [s["d_max_h_sq"] for s in samples]

    # Mean and std
    ratio_h_mean = np.mean(ratio_h_samples)
    ratio_h_std = np.std(ratio_h_samples)

    var_h_mean = np.mean(var_h_samples)
    d_max_h_sq_mean = np.mean(d_max_h_sq_samples)

    # Estimate edge budget with d_close = D_max/√N
    d_close_threshold = np.sqrt(d_max_h_sq_mean / N)
    n_close_estimate = estimate_edge_budget(
        var_h_mean,
        d_max_h_sq_mean,
        d_close_threshold,
        N,
    )

    # Theoretical bounds
    n_close_theoretical_1_5 = N**1.5  # O(N^{3/2}) target
    n_close_theoretical_2 = N**2 / 2  # O(N²) upper bound

    # Scaling analysis
    scaling_exponent = (
        np.log(n_close_estimate) / np.log(N) if N > 1 and n_close_estimate > 0 else 0
    )

    print(f"\n{'=' * 60}")
    print("Results:")
    print(f"{'=' * 60}")
    print(f"Variance Ratio (Var_h / D_max²): {ratio_h_mean:.4f} ± {ratio_h_std:.4f}")
    print("  - Critical threshold for O(N^{3/2}): 0.50")
    print("  - Typical equilibrium range: 0.10-0.30")
    print("\nEdge Budget Estimate:")
    print(f"  N_close (measured): {n_close_estimate:.2e}")
    print(f"  O(N^{{3/2}}) bound: {n_close_theoretical_1_5:.2e}")
    print(f"  O(N²) bound: {n_close_theoretical_2:.2e}")
    print(f"  Scaling exponent: {scaling_exponent:.3f}")
    print("    - Target for O(N^{3/2}): 1.5")
    print("    - Fallback O(N²): 2.0")

    # Decision
    if ratio_h_mean >= 0.45:
        decision = "✅ HIGH VARIANCE - O(N^{3/2}) feasible"
    elif ratio_h_mean >= 0.30:
        decision = "⚠️  MODERATE-HIGH - Borderline, needs investigation"
    else:
        decision = "❌ MODERATE VARIANCE - O(N^{3/2}) unlikely, expect O(N²)"

    print(f"\n{decision}")
    print(f"{'=' * 60}\n")

    return {
        "N": N,
        "d": d,
        "ratio_h_mean": ratio_h_mean,
        "ratio_h_std": ratio_h_std,
        "var_h_mean": var_h_mean,
        "d_max_h_sq_mean": d_max_h_sq_mean,
        "n_close_estimate": n_close_estimate,
        "scaling_exponent": scaling_exponent,
        "decision": decision,
        "samples": samples,
    }


def run_scaling_study(
    N_values: list[int],
    d: int = 2,
    n_steps_warmup: int = 5000,
    n_steps_sample: int = 1000,
    sample_interval: int = 50,
    device: str = "cpu",
) -> list[dict]:
    """Run QSD variance experiments across multiple N values.

    Args:
        N_values: List of walker counts to test
        d: Dimensionality
        n_steps_warmup: Warmup steps per experiment
        n_steps_sample: Sampling steps per experiment
        sample_interval: Sampling interval
        device: "cpu" or "cuda"

    Returns:
        List of result dicts
    """
    results = []

    for N in N_values:
        result = run_single_experiment(
            N=N,
            d=d,
            n_steps_warmup=n_steps_warmup,
            n_steps_sample=n_steps_sample,
            sample_interval=sample_interval,
            device=device,
        )
        results.append(result)

    return results


def plot_results(results: list[dict], save_path: str | None = None):
    """Plot variance ratio and edge budget scaling.

    Args:
        results: List of experiment results
        save_path: Optional path to save figure
    """
    N_values = [r["N"] for r in results]
    ratio_h_means = [r["ratio_h_mean"] for r in results]
    ratio_h_stds = [r["ratio_h_std"] for r in results]
    n_close_estimates = [r["n_close_estimate"] for r in results]

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Variance Ratio
    ax1.errorbar(
        N_values, ratio_h_means, yerr=ratio_h_stds, marker="o", capsize=5, label="Measured"
    )
    ax1.axhline(y=0.5, color="r", linestyle="--", label="Critical (0.50 for O(N^{3/2}))")
    ax1.axhspan(0.1, 0.3, alpha=0.2, color="orange", label="Typical Equilibrium")
    ax1.set_xlabel("Number of Walkers (N)")
    ax1.set_ylabel("Variance Ratio (Var_h / D_max²)")
    ax1.set_title("QSD Variance Level vs. Swarm Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Edge Budget Scaling
    ax2.loglog(N_values, n_close_estimates, "o-", label="Measured N_close")

    # Reference lines
    N_ref = np.array(N_values)
    ax2.loglog(N_ref, N_ref**1.5, "--", alpha=0.5, label="O(N^{3/2})")
    ax2.loglog(N_ref, N_ref**2 / 2, "--", alpha=0.5, label="O(N²)/2")

    ax2.set_xlabel("Number of Walkers (N)")
    ax2.set_ylabel("Number of Close Pairs (N_close)")
    ax2.set_title("Edge Budget Scaling")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to: {save_path}")

    plt.close()


def main():
    """Main experiment runner."""
    print("\n" + "=" * 70)
    print("QSD Variance Analysis Experiment")
    print("=" * 70)
    print("\nResearch Question:")
    print("  Does the QSD achieve near-maximal variance (Var_h ≈ D_max²/2)?")
    print("\nContext:")
    print("  - Phase-Space Packing Lemma requires Var_h ≥ D_max²/2 for O(N^{3/2}) edges")
    print("  - Framework suggests equilibrium variance (balance, not maximum)")
    print("  - This experiment measures Var_h / D_max² empirically")
    print("\nDecision Criteria:")
    print("  - Ratio ≈ 0.45-0.50 → High variance, O(N^{3/2}) feasible")
    print("  - Ratio ≈ 0.10-0.30 → Moderate equilibrium, O(N^{3/2}) unprovable")
    print("=" * 70)

    # Experiment parameters
    N_values = [50, 100, 200]  # Test multiple scales (reduced for speed)
    d = 2  # 2D for visualization
    n_steps_warmup = 3000  # Reach QSD (reduced for speed)
    n_steps_sample = 500  # Sample QSD (reduced for speed)
    sample_interval = 25  # Sample every 25 steps

    print("\nExperimental Setup:")
    print(f"  N values: {N_values}")
    print(f"  Dimension: {d}")
    print(f"  Warmup steps: {n_steps_warmup}")
    print(f"  Sample steps: {n_steps_sample}")
    print(f"  Sample interval: {sample_interval}")

    # Run experiments
    results = run_scaling_study(
        N_values=N_values,
        d=d,
        n_steps_warmup=n_steps_warmup,
        n_steps_sample=n_steps_sample,
        sample_interval=sample_interval,
        device="cpu",
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("=" * 70)
    print(f"\n{'N':<8} {'Ratio':<12} {'Decision'}")
    print("-" * 70)
    for r in results:
        print(f"{r['N']:<8} {r['ratio_h_mean']:.4f} ± {r['ratio_h_std']:.4f}  {r['decision']}")

    # Overall conclusion
    avg_ratio = np.mean([r["ratio_h_mean"] for r in results])
    print("\n" + "=" * 70)
    print("OVERALL CONCLUSION")
    print("=" * 70)
    print(f"Average variance ratio across all N: {avg_ratio:.4f}")

    if avg_ratio >= 0.45:
        print("\n✅ RESULT: HIGH VARIANCE REGIME")
        print("   → QSD achieves near-maximal variance")
        print("   → O(N^{3/2}) edge budget is achievable")
        print("   → Hierarchical clustering proof via edge-counting is FEASIBLE")
        print("   → Next step: Prove variance maximization mechanism")
    elif avg_ratio >= 0.30:
        print("\n⚠️  RESULT: BORDERLINE REGIME")
        print("   → QSD has moderate-high variance")
        print("   → Edge budget scaling unclear (between N^{3/2} and N²)")
        print("   → Hierarchical clustering proof UNCERTAIN")
        print("   → Next step: Investigate N-scaling trend, test larger N")
    else:
        print("\n❌ RESULT: MODERATE EQUILIBRIUM REGIME")
        print("   → QSD has moderate variance (typical for confined systems)")
        print("   → Edge budget is O(N²), NOT O(N^{3/2})")
        print("   → Hierarchical clustering proof via edge-counting is INFEASIBLE")
        print("   → Next step: Accept O(N²) budget or explore alternative strategies")

    print("=" * 70)

    # Plot results
    save_path = Path(__file__).parent / "qsd_variance_results.png"
    plot_results(results, save_path=str(save_path))

    # Save numerical results
    results_path = Path(__file__).parent / "qsd_variance_results.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("QSD Variance Analysis Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'N':<8} {'Ratio':<12} {'Std':<12} {'N_close':<15} {'Exponent':<10}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(
                f"{r['N']:<8} {r['ratio_h_mean']:<12.4f} {r['ratio_h_std']:<12.4f} "
                f"{r['n_close_estimate']:<15.2e} {r['scaling_exponent']:<10.3f}\n"
            )
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Average ratio: {avg_ratio:.4f}\n")

    print(f"\nNumerical results saved to: {results_path}")

    return results


if __name__ == "__main__":
    results = main()
