"""
Quick test script to verify notebook 05 setup works correctly.
Tests the Euclidean Gas experiment module and parameter initialization.
"""

import sys

sys.path.insert(0, "../src")

import numpy as np
import torch

from fragile.bounds import TorchBounds
from fragile.companion_selection import CompanionSelection
from fragile.euclidean_gas import (
    CloningParams,
    EuclideanGas,
    EuclideanGasParams,
    LangevinParams,
)
from fragile.euclidean_gas_experiments import (
    ConvergenceExperiment,
    create_multimodal_potential,
)


print("=" * 60)
print("Testing Notebook 05 Setup: Euclidean Gas Convergence")
print("=" * 60)

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Create multimodal potential
dims = 2
n_gaussians = 3

potential, target_mixture = create_multimodal_potential(
    dims=dims,
    n_gaussians=n_gaussians,
    bounds_range=(-8.0, 8.0),
    seed=42
)

print(f"\n✓ Created multimodal potential with {n_gaussians} modes")
print(f"  Centers: {target_mixture.centers.tolist()}")
print(f"  Weights: {target_mixture.weights.tolist()}")

# Test potential evaluation
test_points = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
values = potential.evaluate(test_points)
print(f"✓ Potential evaluation works: {values}")

# Create Euclidean Gas parameters
N = 20  # Small number for testing

# Define bounds
bounds = TorchBounds(
    low=torch.tensor([-6.0, -6.0]),
    high=torch.tensor([6.0, 6.0])
)

# Create companion selection strategy
companion_selection = CompanionSelection(
    method="uniform",
    lambda_alg=0.0,
)

params = EuclideanGasParams(
    N=N,
    d=dims,
    potential=potential,
    langevin=LangevinParams(gamma=1.0, beta=1.0, delta_t=0.05),
    cloning=CloningParams(
        sigma_x=0.1,
        lambda_alg=0.0,
        alpha_restitution=0.5,
        # Fitness computation parameters
        alpha=1.0,
        beta=1.0,
        eta=0.1,
        A=2.0,
        sigma_min=1e-8,
        # Cloning decision parameters
        p_max=1.0,
        epsilon_clone=0.01,
        # Companion selection
        companion_selection=companion_selection,
    ),
    bounds=bounds,
    device="cpu",
    dtype="float32",
)

print("✓ Created EuclideanGasParams")

# Create Euclidean Gas instance
gas = EuclideanGas(params)
print("✓ Created EuclideanGas instance")

# Initialize state
x_init = torch.rand(N, dims) * 2.0 + 4.0
v_init = torch.randn(N, dims) * 0.1
state = gas.initialize_state(x_init, v_init)
print(f"✓ Initialized state with {state.N} walkers")

# Test one step
print("\nTesting single step...")
try:
    _, new_state = gas.step(state)
    print("✓ Single step successful")
    print(f"  Position range: [{new_state.x.min():.2f}, {new_state.x.max():.2f}]")
    print(f"  Velocity range: [{new_state.v.min():.2f}, {new_state.v.max():.2f}]")
except Exception as e:
    print(f"✗ Step failed: {e}")
    raise

# Test convergence experiment
print("\nTesting convergence experiment...")
try:
    snapshot_times = [0, 5, 10]
    experiment = ConvergenceExperiment(
        gas=gas,
        save_snapshots_at=snapshot_times
    )
    print("✓ Created ConvergenceExperiment")

    # Run short experiment
    metrics, snapshots = experiment.run(
        n_steps=10,
        x_init=x_init,
        v_init=v_init,
        measure_every=2,
        verbose=True
    )

    print(f"✓ Experiment run successful")
    print(f"  Measurements: {len(metrics.time)}")
    print(f"  Snapshots: {len(snapshots)}")
    print(f"  Initial V_total: {metrics.V_total[0]:.6f}")
    print(f"  Final V_total: {metrics.V_total[-1]:.6f}")

    # Test exponential fit
    fit_result = metrics.fit_exponential_decay('V_total', fit_start_time=0)
    if fit_result is not None:
        kappa, C = fit_result
        print(f"✓ Exponential fit successful: κ = {kappa:.4f}, C = {C:.4f}")
    else:
        print("  (Exponential fit skipped - not enough data)")

except Exception as e:
    print(f"✗ Experiment failed: {e}")
    raise

print("\n" + "=" * 60)
print("All tests passed! Notebook 05 setup is working correctly.")
print("=" * 60)
