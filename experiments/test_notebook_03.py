"""
Quick test script to verify the notebook setup works correctly.
Tests the potential creation and parameter initialization without running full simulation.
"""

import sys


sys.path.insert(0, "../src")

import numpy as np
import torch

from fragile.benchmarks import MixtureOfGaussians
from fragile.euclidean_gas import (
    LangevinParams,
    PotentialParams,
)
from fragile.geometric_gas import (
    AdaptiveParams,
    GeometricGas,
    GeometricGasParams,
    LocalizationKernelParams,
)


print("=" * 60)
print("Testing Notebook 03 Setup")
print("=" * 60)

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Create multimodal potential
dims = 2
n_gaussians = 3

centers = torch.tensor([
    [0.0, 0.0],
    [4.0, 3.0],
    [-3.0, 2.5],
])

stds = torch.tensor([
    [0.8, 0.8],
    [1.0, 1.0],
    [1.2, 1.2],
])

weights = torch.tensor([0.5, 0.3, 0.2])

target_mixture = MixtureOfGaussians(
    dims=dims,
    n_gaussians=n_gaussians,
    centers=centers,
    stds=stds,
    weights=weights,
    bounds_range=(-8.0, 8.0),
)

print(f"\n✓ Created mixture with {n_gaussians} modes")


# Create potential wrapper
class MixtureBasedPotential(PotentialParams):
    """Potential derived from Mixture of Gaussians."""

    mixture: object  # Field to store the mixture

    model_config = {"arbitrary_types_allowed": True}

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixture(x)


potential = MixtureBasedPotential(mixture=target_mixture)
print("✓ Created potential wrapper")

# Test potential evaluation
test_points = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
values = potential.evaluate(test_points)
print(f"✓ Potential evaluation works: {values}")

# Create Geometric Gas parameters
N = 20  # Small number for testing

params = GeometricGasParams(
    N=N,
    d=dims,
    potential=potential,
    langevin=LangevinParams(gamma=1.0, beta=1.0, delta_t=0.05),
    localization=LocalizationKernelParams(rho=2.0, kernel_type="gaussian"),
    adaptive=AdaptiveParams(
        epsilon_F=0.05,
        nu=0.02,
        epsilon_Sigma=0.01,
        rescale_amplitude=1.0,
        sigma_var_min=0.1,
        viscous_length_scale=2.0,
    ),
    device="cpu",
    dtype="float32",
)

print("✓ Created GeometricGasParams")


# Create measurement function
def measurement_fn(x):
    return -potential.evaluate(x)


# Create Geometric Gas instance
gas = GeometricGas(params, measurement_fn=measurement_fn)
print("✓ Created GeometricGas instance")

# Initialize state
x_init = torch.rand(N, dims) * 2.0 + 5.0
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

print("\n" + "=" * 60)
print("All tests passed! Notebook 03 setup is working correctly.")
print("=" * 60)
