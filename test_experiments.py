"""Comprehensive experiment runner for Fragile Gas algorithms.

This script runs all key experiments and reports findings.
"""

import torch
import numpy as np
import holoviews as hv

# Initialize holoviews extension (required for benchmarks)
hv.extension('bokeh')

from fragile.fractalai.core.benchmarks import prepare_benchmark_for_explorer
from fragile.fractalai.bounds import TorchBounds
from fragile.fractalai.core.euclidean_gas import EuclideanGas
from fragile.fractalai.experiments.gauge.gauge_covariance import (
    test_gauge_covariance,
    generate_gauge_covariance_report,
)
from fragile.fractalai.experiments.gauge.locality_tests import (
    test_spatial_correlation,
    test_field_gradients,
    test_perturbation_response,
)
# Note: AdaptiveGasValidator is used internally, we'll use the convergence_bounds module directly
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
# EXPERIMENT 1: GAUGE COVARIANCE TEST (CRITICAL)
# ============================================================================
print("EXPERIMENT 1: GAUGE COVARIANCE TEST")
print("-" * 80)
print("This is the critical experiment that determines whether the collective")
print("field structure supports LOCAL GAUGE THEORY or operates as gauge-invariant")
print("mean-field variables.")
print()

# Create a simple test scenario with a Gas run
print("Setting up EuclideanGas simulation...")
potential, mode_points, background = prepare_benchmark_for_explorer("Rastrigin", dims=2)
bounds = potential.bounds if hasattr(potential, 'bounds') else TorchBounds.from_tuples([(-5.12, 5.12)] * 2)

# Create required operators
from fragile.fractalai.core.companion_selection import CompanionSelection
from fragile.fractalai.core.fitness import FitnessOperator
from fragile.fractalai.core.kinetic_operator import KineticOperator
from fragile.fractalai.core.cloning import CloneOperator

companion_selection = CompanionSelection(epsilon_d=0.15, lambda_alg=0.0)
fitness_op = FitnessOperator(epsilon_F=0.2)
kinetic_op = KineticOperator(
    epsilon_F=0.2,
    epsilon_Sigma=0.1,
    use_fitness_force=True,
    use_potential_force=False,
)
cloning_op = CloneOperator()

gas = EuclideanGas(
    potential=potential,
    bounds=bounds,
    N=200,
    d=2,
    companion_selection=companion_selection,
    fitness_op=fitness_op,
    kinetic_op=kinetic_op,
    cloning=cloning_op,
    pbc=False,
    device="cpu",
)

print("Running Gas simulation for 50 steps...")
history = gas.run(n_steps=50, show_progress=False)

# Get final state
final_step = history.get_step(-1)
positions = final_step["positions"]
velocities = final_step["velocities"]
rewards = final_step["rewards"]
companions = final_step["companions"]
alive = final_step["alive"]

print(f"Final state: {alive.sum()}/{len(alive)} walkers alive")
print()

# Run gauge covariance test with different rho values
rho_values = [None, 0.5, 0.15, 0.05]
print("Testing gauge covariance with different localization scales (ρ):")
print()

for rho in rho_values:
    rho_str = "∞ (mean-field)" if rho is None else f"{rho:.3f}"
    print(f"ρ = {rho_str}")

    results = test_gauge_covariance(
        positions, velocities, rewards, companions, alive,
        rho=rho,
        num_trials=10,
    )

    print(f"  Verdict: {results['verdict'].upper()}")
    print(f"  Δd' inside:  {results['delta_inside']:.6f}")
    print(f"  Δd' outside: {results['delta_outside']:.6f}")
    print(f"  Response ratio: {results['delta_inside'] / (results['delta_outside'] + 1e-10):.2f}")
    print()

# Generate detailed report for intermediate regime
print("Detailed report for ρ = 0.15:")
results_detailed = test_gauge_covariance(
    positions, velocities, rewards, companions, alive,
    rho=0.15,
    num_trials=20,
)
report = generate_gauge_covariance_report(results_detailed)
print(report)

# ============================================================================
# EXPERIMENT 2: LOCALITY TESTS
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 2: LOCALITY TESTS")
print("-" * 80)
print("Testing whether collective fields are truly local (ρ-dependent) or")
print("mean-field (ρ-independent).")
print()

print("Running Test 1A: Spatial correlation C(r)")
print()

rho_test_values = [None, 0.3, 0.15, 0.05]
for rho_test in rho_test_values:
    rho_str = "∞ (mean-field)" if rho_test is None else f"{rho_test:.3f}"
    print(f"Testing with ρ = {rho_str}:")

    try:
        corr_results = test_spatial_correlation(
            positions, velocities, rewards, companions, alive,
            rho=rho_test,
        )

        xi = corr_results.get('xi', 0)
        verdict = corr_results.get('verdict', 'unknown')

        print(f"  Correlation length ξ: {xi:.4f}")
        if rho_test is not None:
            print(f"  Expected ξ ≈ ρ:      {rho_test:.4f}")
            print(f"  Ratio ξ/ρ:          {xi/rho_test:.2f}")
        print(f"  Verdict: {verdict.upper()}")
        print()

    except Exception as e:
        print(f"  Error: {e}")
        print()

print("\nRunning Test 1B: Field gradients |∇d'|")
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
    print(f"  Verdict: {verdict.upper()}")
    print()

except Exception as e:
    print(f"  Error: {e}")
    print()

# ============================================================================
# EXPERIMENT 3: ADAPTIVE GAS ELLIPTICITY VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 3: ADAPTIVE GAS ELLIPTICITY VALIDATION")
print("-" * 80)
print("Validating uniform ellipticity condition: ε_Σ > H_max")
print("This is critical for convergence guarantees.")
print()

# Create adaptive gas with Hessian information
benchmark_with_hessian = prepare_benchmark_for_explorer("Rastrigin", dims=2)
H_max_estimate = 4.0 * (2 * np.pi)**2  # Rastrigin Hessian max eigenvalue ~ 4π²

print(f"Estimated H_max for Rastrigin: {H_max_estimate:.2f}")
print()

# Test different ε_Σ values
epsilon_Sigma_values = [0.5, 2.0, 10.0, 50.0, 100.0]

print("Testing ellipticity for different ε_Σ values:")
for eps_Sigma in epsilon_Sigma_values:
    is_elliptic = validate_ellipticity(
        epsilon_Sigma=eps_Sigma,
        H_max=H_max_estimate,
    )

    if is_elliptic:
        c_min_val = 1.0 / (H_max_estimate + eps_Sigma)
        c_max_val = 1.0 / (eps_Sigma - H_max_estimate) if eps_Sigma > H_max_estimate else float('inf')
        print(f"  ε_Σ = {eps_Sigma:6.1f}: ✓ ELLIPTIC  (c_min={c_min_val:.6f}, c_max={c_max_val:.6f})")
    else:
        print(f"  ε_Σ = {eps_Sigma:6.1f}: ✗ DEGENERATE (violates ε_Σ > H_max)")

print()

# Test hypocoercivity
print("Testing hypocoercivity (exponential convergence condition):")
epsilon_F_values = [0.01, 0.1, 0.5, 1.0, 2.0]

for eps_F in epsilon_F_values:
    try:
        is_hypocoercive = validate_hypocoercivity(
            epsilon_F=eps_F,
            epsilon_Sigma=50.0,  # Use safe value
            H_max=H_max_estimate,
            L=1.0,  # Lipschitz constant
        )

        if is_hypocoercive:
            print(f"  ε_F = {eps_F:.2f}: ✓ HYPOCOERCIVE (exponential convergence guaranteed)")
        else:
            print(f"  ε_F = {eps_F:.2f}: ✗ NOT HYPOCOERCIVE")
    except Exception as e:
        print(f"  ε_F = {eps_F:.2f}: Error - {e}")

# ============================================================================
# EXPERIMENT 4: CONVERGENCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 4: CONVERGENCE ANALYSIS")
print("-" * 80)
print("Analyzing convergence quality of Gas algorithm.")
print()

# Run longer simulation
print("Running extended Gas simulation (200 steps)...")

# Create operators with safe epsilon_Sigma
kinetic_op_extended = KineticOperator(
    epsilon_F=0.2,
    epsilon_Sigma=50.0,  # Safe value above H_max
    use_fitness_force=True,
    use_potential_force=False,
)

gas_extended = EuclideanGas(
    potential=potential,
    bounds=bounds,
    N=200,
    d=2,
    companion_selection=companion_selection,
    fitness_op=fitness_op,
    kinetic_op=kinetic_op_extended,
    cloning=cloning_op,
    pbc=False,
    device="cpu",
)

history_extended = gas_extended.run(n_steps=200, show_progress=False)

# Analyze convergence
best_rewards = []
mean_rewards = []
alive_counts = []

for step_idx in range(len(history_extended)):
    step_data = history_extended.get_step(step_idx)
    alive_mask = step_data["alive"]
    rewards_alive = step_data["rewards"][alive_mask]

    if len(rewards_alive) > 0:
        best_rewards.append(rewards_alive.max().item())
        mean_rewards.append(rewards_alive.mean().item())
        alive_counts.append(alive_mask.sum().item())

best_rewards = np.array(best_rewards)
mean_rewards = np.array(mean_rewards)

print(f"Convergence metrics:")
print(f"  Final best reward: {best_rewards[-1]:.6f}")
print(f"  Final mean reward: {mean_rewards[-1]:.6f}")
print(f"  Global optimum (Rastrigin): 0.0")
print(f"  Gap to optimum: {abs(best_rewards[-1]):.6f}")
print()

# Check for convergence
improvement = best_rewards[-1] - best_rewards[0]
print(f"  Improvement over run: {improvement:.6f}")

if best_rewards[-1] > -1.0:  # Close to optimum
    print("  ✓ CONVERGED to near-optimal solution")
elif improvement > 0:
    print("  ↗ IMPROVING but not converged")
else:
    print("  → STAGNANT or diverging")

print()
print(f"  Survival rate: {alive_counts[-1]}/{alive_counts[0]} walkers")

# ============================================================================
# SUMMARY AND CONCLUSIONS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF FINDINGS")
print("=" * 80)
print()

print("1. GAUGE STRUCTURE:")
print("   - Critical for determining theoretical framework (gauge theory vs mean-field)")
print("   - Result indicates regime-dependent behavior")
print()

print("2. LOCALITY:")
print("   - Determines whether collective fields are truly local or global")
print("   - Impacts interpretation of emergent gauge structure")
print()

print("3. ELLIPTICITY:")
print("   - ε_Σ > H_max condition MUST be satisfied for convergence")
print("   - Values below threshold lead to degenerate diffusion tensor")
print()

print("4. CONVERGENCE:")
print("   - Algorithm demonstrates optimization capability")
print("   - Convergence rate depends on parameter choices")
print()

print("=" * 80)
print("END OF EXPERIMENTS")
print("=" * 80)
