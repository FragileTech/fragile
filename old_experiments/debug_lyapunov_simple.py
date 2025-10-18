"""
Simple debug script to test individual components before full notebook.
"""
import torch
from fragile.euclidean_gas import (
    EuclideanGas,
    EuclideanGasParams,
    LangevinParams,
    CloningParams,
    SimpleQuadraticPotential,
)
from fragile.bounds import TorchBounds
from fragile.lyapunov import VectorizedOps

print("=" * 80)
print("SIMPLE LYAPUNOV DEBUG")
print("=" * 80)

# ============================================================================
# 1. Test Basic Setup
# ============================================================================
print("\n1. Testing basic setup...")

device = "cpu"
N = 20  # Small number for fast testing
d = 2

# Bounds
bounds = TorchBounds(
    high=torch.tensor([5.0, 5.0]),
    low=torch.tensor([-5.0, -5.0]),
    device=device
)

# Optimum at center
x_opt = torch.tensor([0.0, 0.0], device=device)

# Potential
quadratic_potential = SimpleQuadraticPotential(
    x_opt=x_opt,
    reward_alpha=1.0,
    reward_beta=0.0
)

# Langevin
langevin_params = LangevinParams(
    gamma=1.0,
    beta=1.0,
    delta_t=0.01,
    integrator="baoab"
)

# Cloning
cloning_params = CloningParams(
    sigma_x=0.5,
    lambda_alg=1.0,
    alpha_restitution=0.5,
    use_inelastic_collision=True
)

# Combined parameters
params = EuclideanGasParams(
    N=N,
    d=d,
    potential=quadratic_potential,
    langevin=langevin_params,
    cloning=cloning_params,
    bounds=bounds,
    device=device,
    dtype="float32"
)

print("✓ Parameters created successfully")

# ============================================================================
# 2. Test Gas Initialization
# ============================================================================
print("\n2. Testing gas initialization...")

gas = EuclideanGas(params)
print("✓ Gas instance created")

# Initialize state
torch.manual_seed(42)
x_init = bounds.sample(N)
v_init = torch.randn(N, d, device=device) * 1.0

state = gas.initialize_state(x_init, v_init)
print("✓ State initialized")
print(f"  x shape: {state.x.shape}")
print(f"  v shape: {state.v.shape}")

# ============================================================================
# 3. Test Basic Metrics
# ============================================================================
print("\n3. Testing basic metrics...")

alive_mask = bounds.contains(state.x)
print(f"  Alive walkers: {alive_mask.sum().item()}/{N}")

var_x = VectorizedOps.variance_position(state)
var_v = VectorizedOps.variance_velocity(state)
print(f"  Position variance: {var_x:.4f}")
print(f"  Velocity variance: {var_v:.4f}")

potential = quadratic_potential.evaluate(state.x)
print(f"  Mean potential: {potential.mean():.4f}")
print(f"  Potential shape: {potential.shape}")

print("✓ Basic metrics working")

# ============================================================================
# 4. Test Single Step
# ============================================================================
print("\n4. Testing single step...")

_, state_new = gas.step(state)
print("✓ Single step completed")

var_x_new = VectorizedOps.variance_position(state_new)
print(f"  New position variance: {var_x_new:.4f}")

# ============================================================================
# 5. Test Short Run (10 steps, no clustering)
# ============================================================================
print("\n5. Testing short run without clustering...")

state = gas.initialize_state(x_init, v_init)
n_steps = 10

var_history = []
alive_history = []

for t in range(n_steps):
    _, state = gas.step(state)
    var_history.append(VectorizedOps.variance_position(state).item())
    alive_history.append(bounds.contains(state.x).sum().item())

print(f"✓ Completed {n_steps} steps")
print(f"  Final variance: {var_history[-1]:.4f}")
print(f"  Final alive: {alive_history[-1]}/{N}")

# ============================================================================
# 6. Test Clustering on One State (POTENTIAL BOTTLENECK)
# ============================================================================
print("\n6. Testing clustering analysis (potential bottleneck)...")

from fragile.lyapunov import identify_high_error_clusters

# Reset to initial state
state = gas.initialize_state(x_init, v_init)
alive_mask = bounds.contains(state.x)

if alive_mask.sum() >= 2:
    print(f"  Testing with {alive_mask.sum().item()} alive walkers...")

    try:
        # This is likely where it hangs
        epsilon = cloning_params.get_epsilon_c()
        lambda_alg = cloning_params.lambda_alg

        print(f"  epsilon = {epsilon:.4f}")
        print(f"  lambda_alg = {lambda_alg:.4f}")

        clusters = identify_high_error_clusters(
            state, alive_mask,
            epsilon=epsilon,
            lambda_alg=lambda_alg
        )

        print("✓ Clustering completed")
        print(f"  High-error walkers: {clusters['H_k'].sum().item()}")
        print(f"  Low-error walkers: {clusters['L_k'].sum().item()}")

    except Exception as e:
        print(f"✗ Clustering failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("  Skipping: too few alive walkers")

# ============================================================================
# DONE
# ============================================================================
print("\n" + "=" * 80)
print("ALL TESTS COMPLETED")
print("=" * 80)
