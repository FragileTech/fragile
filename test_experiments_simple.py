"""Simplified experiment runner for Fragile Gas algorithms.

This script runs key experiments using the existing GasConfig infrastructure.
"""

import torch
import numpy as np
import holoviews as hv

# Initialize holoviews extension (required for benchmarks)
hv.extension('bokeh')

from fragile.fractalai.experiments.gas_config_panel import GasConfigPanel
from fragile.fractalai.experiments.gauge.gauge_covariance import (
    test_gauge_covariance,
    generate_gauge_covariance_report,
)
from fragile.fractalai.experiments.gauge.locality_tests import (
    test_spatial_correlation,
    test_field_gradients,
)
from fragile.fractalai.convergence_bounds import (
    validate_ellipticity,
    validate_hypocoercivity,
)

print("=" * 80)
print("FRAGILE GAS EXPERIMENTS - COMPREHENSIVE TEST SUITE")
print("=" * 80)
print()

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# SETUP: Run Gas simulation using GasConfigPanel
# ============================================================================
print("SETUP: Running Gas simulation")
print("-" * 80)
print("Using Rastrigin benchmark (2D), 200 walkers, 100 steps")
print()

# Create Gas configuration with default parameters
config = GasConfigPanel(dims=2)
config.benchmark_name = "Rastrigin"
config.n_steps = 100

# Run simulation
print("Running simulation (this may take a minute)...")
history = config.run_simulation()

# Get final state from history
positions = history.x_final[-1]  # Last recorded positions
velocities = history.v_final[-1]  # Last recorded velocities
rewards = history.rewards[-1]  # Last recorded rewards
companions = history.companions_distance[-1]  # Last recorded companions
alive = history.alive_mask[-1]  # Last recorded alive mask

print(f"Simulation complete: {alive.sum()}/{len(alive)} walkers alive at end")
print(f"Best reward: {rewards[alive].max().item():.6f}")
print(f"Mean reward: {rewards[alive].mean().item():.6f}")
print()

# ============================================================================
# EXPERIMENT 1: GAUGE COVARIANCE TEST (CRITICAL)
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 1: GAUGE COVARIANCE TEST")
print("=" * 80)
print()
print("**PURPOSE**: Determine whether collective fields support LOCAL GAUGE THEORY")
print("or operate as gauge-invariant mean-field variables.")
print()
print("**THEORETICAL BACKGROUND**:")
print("  - If d'_i transforms non-trivially under local U(1) phase transformation")
print("    → Gauge-covariant fields → Local gauge theory viable")
print("  - If d'_i remains invariant")
print("    → Gauge-invariant observables → Mean-field interpretation")
print()

# Run gauge covariance test with different rho values
rho_values = [None, 0.5, 0.15, 0.05]
print("Testing with different localization scales (ρ):")
print()

results_by_rho = {}
for rho in rho_values:
    rho_str = "∞ (mean-field)" if rho is None else f"{rho:.3f}"
    print(f"ρ = {rho_str}:")

    try:
        results = test_gauge_covariance(
            positions, velocities, rewards, companions, alive,
            rho=rho,
            num_trials=10,
        )

        results_by_rho[rho] = results

        print(f"  Verdict: {results['verdict'].upper()}")
        print(f"  Δd' inside:  {results['delta_inside']:.6f}")
        print(f"  Δd' outside: {results['delta_outside']:.6f}")
        ratio = results['delta_inside'] / (results['delta_outside'] + 1e-10)
        print(f"  Response ratio (in/out): {ratio:.2f}")

        if results['verdict'] == 'covariant':
            print("  ✓ GAUGE COVARIANT - Local gauge theory viable!")
        elif results['verdict'] == 'invariant':
            print("  ✗ GAUGE INVARIANT - Mean-field interpretation")
        else:
            print("  ? INCONCLUSIVE - Need more data or different parameters")
        print()

    except Exception as e:
        print(f"  Error: {e}")
        print()

# Generate detailed report for intermediate regime
if 0.15 in results_by_rho:
    print("\nDetailed report for ρ = 0.15 (intermediate regime):")
    print("-" * 80)
    report = generate_gauge_covariance_report(results_by_rho[0.15])
    print(report)

# ============================================================================
# EXPERIMENT 2: LOCALITY TESTS
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 2: LOCALITY TESTS")
print("=" * 80)
print()
print("**PURPOSE**: Verify whether collective fields are truly local (ρ-dependent)")
print("or mean-field (ρ-independent).")
print()

print("Test 1A: Spatial Correlation C(r)")
print("-" * 80)
print("Expected: Local regime → C(r) ~ exp(-r²/ξ²) with ξ ≈ ρ")
print("          Mean-field  → C(r) ≈ constant")
print()

rho_test_values = [None, 0.3, 0.15, 0.05]
for rho_test in rho_test_values:
    rho_str = "∞ (mean-field)" if rho_test is None else f"{rho_test:.3f}"
    print(f"ρ = {rho_str}:")

    try:
        corr_results = test_spatial_correlation(
            positions, velocities, rewards, companions, alive,
            rho=rho_test,
        )

        xi = corr_results.get('xi', 0)
        C0 = corr_results.get('C0', 0)
        verdict = corr_results.get('verdict', 'unknown')

        print(f"  Correlation length ξ: {xi:.4f}")
        if rho_test is not None:
            print(f"  Expected ξ ≈ ρ:      {rho_test:.4f}")
            print(f"  Ratio ξ/ρ:          {xi/rho_test:.2f}")
        print(f"  Amplitude C₀:        {C0:.4f}")
        print(f"  Verdict: {verdict.upper()}")

        if verdict == 'local':
            print("  ✓ LOCAL FIELDS confirmed")
        else:
            print("  ✗ MEAN-FIELD behavior")
        print()

    except Exception as e:
        print(f"  Error: {e}")
        print()

print("\nTest 1B: Field Gradients |∇d'|")
print("-" * 80)
print("Expected: Local regime → |∇d'| ~ O(1/ρ)")
print("          Mean-field  → |∇d'| ≈ 0")
print()

try:
    grad_results = test_field_gradients(
        positions, velocities, rewards, companions, alive,
        rho=0.15,
    )

    mean_grad = grad_results.get('mean_gradient', 0)
    max_grad = grad_results.get('max_gradient', 0)
    verdict = grad_results.get('verdict', 'unknown')

    print(f"  Mean gradient: {mean_grad:.4f}")
    print(f"  Max gradient:  {max_grad:.4f}")
    print(f"  Expected ~1/ρ: {1/0.15:.4f}")
    print(f"  Verdict: {verdict.upper()}")

    if verdict == 'local':
        print("  ✓ SHARP GRADIENTS - Local fields confirmed")
    else:
        print("  → SMOOTH FIELDS - Mean-field behavior")
    print()

except Exception as e:
    print(f"  Error: {e}")
    print()

# ============================================================================
# EXPERIMENT 3: ELLIPTICITY VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 3: ELLIPTICITY VALIDATION")
print("=" * 80)
print()
print("**PURPOSE**: Validate uniform ellipticity condition ε_Σ > H_max")
print("This condition is CRITICAL for convergence guarantees.")
print()

# Rastrigin Hessian max eigenvalue
H_max_rastrigin = 4.0 * (2 * np.pi)**2  # ≈ 157.91
print(f"Rastrigin H_max (estimated): {H_max_rastrigin:.2f}")
print()

epsilon_Sigma_values = [1.0, 10.0, 50.0, 100.0, 200.0]

print("Testing ellipticity for different ε_Σ values:")
print("-" * 80)
for eps_Sigma in epsilon_Sigma_values:
    is_elliptic = validate_ellipticity(
        epsilon_Sigma=eps_Sigma,
        H_max=H_max_rastrigin,
    )

    if is_elliptic:
        c_min_val = 1.0 / (H_max_rastrigin + eps_Sigma)
        c_max_val = 1.0 / abs(eps_Sigma - H_max_rastrigin)
        status = "✓ ELLIPTIC"
        if eps_Sigma < 2 * H_max_rastrigin:
            status += " (but close to boundary!)"
    else:
        c_min_val = c_max_val = float('nan')
        status = "✗ DEGENERATE (violates ε_Σ > H_max)"

    print(f"  ε_Σ = {eps_Sigma:6.1f}: {status}")
    if is_elliptic:
        print(f"               c_min = {c_min_val:.6f}, c_max = {c_max_val:.6f}")

print()

# Test hypocoercivity
print("Testing Hypocoercivity (exponential convergence):")
print("-" * 80)
print("Hypocoercivity ensures exponential convergence to equilibrium.")
print()

epsilon_F_values = [0.1, 0.5, 1.0, 2.0]

for eps_F in epsilon_F_values:
    try:
        is_hypocoercive = validate_hypocoercivity(
            epsilon_F=eps_F,
            epsilon_Sigma=50.0,
            H_max=H_max_rastrigin,
            L=1.0,  # Lipschitz constant (approximate)
        )

        if is_hypocoercive:
            print(f"  ε_F = {eps_F:.2f}: ✓ HYPOCOERCIVE (exponential convergence)")
        else:
            print(f"  ε_F = {eps_F:.2f}: ✗ NOT HYPOCOERCIVE")
    except Exception as e:
        print(f"  ε_F = {eps_F:.2f}: Error - {e}")

print()

# ============================================================================
# EXPERIMENT 4: CONVERGENCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 4: CONVERGENCE ANALYSIS")
print("=" * 80)
print()

# Analyze convergence from the history we already have
n_recorded = history.n_recorded
best_rewards = []
mean_rewards = []
alive_counts = []

for step_idx in range(n_recorded):
    # Get alive walkers at this step
    if step_idx < len(history.alive_mask):
        alive_mask = history.alive_mask[step_idx]
        rewards_step = history.rewards[step_idx]
        rewards_alive = rewards_step[alive_mask]

        if len(rewards_alive) > 0:
            best_rewards.append(rewards_alive.max().item())
            mean_rewards.append(rewards_alive.mean().item())
            alive_counts.append(alive_mask.sum().item())

best_rewards = np.array(best_rewards)
mean_rewards = np.array(mean_rewards)

print(f"Convergence Metrics (100 steps):")
print("-" * 80)
print(f"  Initial best reward:  {best_rewards[0]:.6f}")
print(f"  Final best reward:    {best_rewards[-1]:.6f}")
print(f"  Improvement:          {best_rewards[-1] - best_rewards[0]:.6f}")
print()
print(f"  Global optimum (Rastrigin): 0.0")
print(f"  Gap to optimum:       {abs(best_rewards[-1]):.6f}")
print()

# Convergence assessment
if best_rewards[-1] > -1.0:
    convergence_status = "✓ CONVERGED to near-optimal solution"
elif best_rewards[-1] - best_rewards[0] > 10.0:
    convergence_status = "↗ IMPROVING significantly"
elif best_rewards[-1] - best_rewards[0] > 0:
    convergence_status = "→ IMPROVING slowly"
else:
    convergence_status = "✗ STAGNANT or diverging"

print(f"  Status: {convergence_status}")
print()
print(f"  Walker survival: {alive_counts[-1]}/{alive_counts[0]} ({100*alive_counts[-1]/alive_counts[0]:.1f}%)")
print()

# ============================================================================
# SUMMARY AND CONCLUSIONS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF FINDINGS")
print("=" * 80)
print()

print("1. GAUGE STRUCTURE (Experiment 1):")
print("   Critical for determining theoretical framework:")
print("   - COVARIANT → Local gauge theory (Yang-Mills-like)")
print("   - INVARIANT → Mean-field theory (Hartree-Fock-like)")
print()

# Determine overall gauge verdict
gauge_verdicts = [r['verdict'] for r in results_by_rho.values()]
if 'covariant' in gauge_verdicts:
    print("   Result: GAUGE COVARIANT behavior detected in at least one regime")
    print("   → Local gauge theory interpretation is VIABLE")
elif all(v == 'invariant' for v in gauge_verdicts):
    print("   Result: GAUGE INVARIANT across all tested regimes")
    print("   → Mean-field interpretation applies")
else:
    print("   Result: REGIME-DEPENDENT or INCONCLUSIVE")
    print("   → Requires further investigation")

print()

print("2. LOCALITY (Experiment 2):")
print("   Determines spatial structure of collective fields:")
print("   - LOCAL → ρ-dependent correlation, sharp gradients")
print("   - MEAN-FIELD → ρ-independent, smooth")
print("   Result: See detailed test results above")
print()

print("3. ELLIPTICITY (Experiment 3):")
print("   CRITICAL for convergence: ε_Σ > H_max MUST be satisfied")
print(f"   - Rastrigin H_max ≈ {H_max_rastrigin:.0f}")
# Get epsilon_Sigma from kinetic op
epsilon_Sigma_used = config.kinetic_op.epsilon_Sigma if hasattr(config.kinetic_op, 'epsilon_Sigma') else "unknown"
print(f"   - Current ε_Σ = {epsilon_Sigma_used}")
if isinstance(epsilon_Sigma_used, (int, float)) and epsilon_Sigma_used > H_max_rastrigin:
    print("   → ✓ CONDITION SATISFIED - Convergence guaranteed")
elif isinstance(epsilon_Sigma_used, (int, float)):
    print("   → ✗ WARNING: Degenerate diffusion - convergence NOT guaranteed")
else:
    print("   → Check kinetic operator parameters for actual ε_Σ value")
print()

print("4. CONVERGENCE (Experiment 4):")
print(f"   - {convergence_status}")
print(f"   - Best fitness reached: {best_rewards[-1]:.4f}")
print()

print("=" * 80)
print("END OF EXPERIMENTS")
print("=" * 80)
print()
print("For detailed analysis, see:")
print("  - docs/source/13_fractal_set_new/04c_test_cases.md (Theory)")
print("  - docs/source/2_geometric_gas/ (Geometric foundations)")
print("  - docs/source/3_brascamp_lieb/ (Convergence theory)")
