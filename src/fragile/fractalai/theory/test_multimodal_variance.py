"""Quick test of multimodal variance experiment.

Tests a single configuration with Gaussian 4-mode potential and high diversity
to verify the setup works before running full grid.
"""

from fragile.theory.qsd_variance_sweep import (
    create_gaussian_mixture_potential,
    run_single_parameter_experiment,
)


def test_multimodal_high_diversity():
    """Test single experiment: 4-mode Gaussian with high diversity."""
    print("=" * 70)
    print("Testing Multimodal Variance Experiment")
    print("=" * 70)
    print("\nConfiguration:")
    print("  Potential: Gaussian 4-mode (centers at ±5, ±5)")
    print("  N: 50 walkers")
    print("  Alpha (exploitation): 0.5")
    print("  Beta (diversity): 5.0  [HIGH DIVERSITY]")
    print("  Warmup: 2000 steps")
    print("  Sample: 300 steps")
    print("\nExpectation: High diversity should spread walkers across modes")
    print("  → High variance (ratio ≈ 0.3-0.5 if hypothesis correct)")
    print("=" * 70)

    # Create 4-mode Gaussian potential
    potential = create_gaussian_mixture_potential(
        centers=[(-5.0, -5.0), (-5.0, 5.0), (5.0, -5.0), (5.0, 5.0)],
        depths=[1.0, 1.0, 1.0, 1.0],
        widths=[1.5, 1.5, 1.5, 1.5],
    )

    # Run single experiment
    result = run_single_parameter_experiment(
        N=50,
        d=2,
        potential=potential,
        potential_name="gaussian_4mode_test",
        alpha_fit=0.5,  # Moderate exploitation
        beta_fit=5.0,  # High diversity (strong repulsion)
        n_steps_warmup=2000,
        n_steps_sample=300,
        record_every=25,
    )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(
        f"Variance ratio (Var_h / D²_max): {result['ratio_h_mean']:.4f} ± {result['ratio_h_std']:.4f}"
    )
    print("  Critical threshold for O(N^{3/2}): 0.45")
    print("\nEdge budget:")
    print(f"  N_close (measured): {result['n_close_estimate']:.2e}")
    print(f"  Scaling exponent: {result['scaling_exponent']:.3f}")
    print("    Target for O(N^{3/2}): 1.5")
    print("    O(N²) scaling: 2.0")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if result["ratio_h_mean"] >= 0.45:
        print("✅ HIGH VARIANCE ACHIEVED!")
        print("  → Multimodal + high diversity DOES produce near-maximal variance")
        print("  → O(N^{3/2}) edge budget IS feasible")
        print("  → Hierarchical clustering proof via edge-counting CAN work")
    elif result["ratio_h_mean"] >= 0.30:
        print("⚠️ MODERATE-HIGH VARIANCE")
        print(f"  → Variance ratio {result['ratio_h_mean']:.3f} is higher than unimodal (~0.07)")
        print("  → But still below critical threshold (0.45)")
        print("  → Need to test more parameter combinations")
    else:
        print("❌ STILL LOW VARIANCE")
        print(f"  → Even with multimodal + high diversity: {result['ratio_h_mean']:.3f}")
        print("  → Similar to unimodal baseline (~0.07)")
        print("  → O(N^{3/2}) likely unprovable")

    print("=" * 70)

    return result


if __name__ == "__main__":
    result = test_multimodal_high_diversity()
