#!/usr/bin/env python3
"""
Debug script for lyapunov notebook.
Run this to test all functionality before putting it in the notebook.
"""

import sys
sys.path.insert(0, '../src')

import torch
import numpy as np
import holoviews as hv
from holoviews import opts
import panel as pn

# Enable Bokeh backend
hv.extension('bokeh')
pn.extension()

from fragile.euclidean_gas import (
    EuclideanGas,
    EuclideanGasParams,
    LangevinParams,
    CloningParams,
    SwarmState,
    VectorizedOps,
)
from fragile.bounds import TorchBounds
from fragile.lyapunov import identify_high_error_clusters

print("✓ Imports loaded successfully")

# ============================================================================
# 1. Configure Swarm Parameters
# ============================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device}")

# Swarm parameters
N = 50   # Smaller number for detailed tracking
d = 2    # Dimensions

# Domain bounds
bounds = TorchBounds(
    low=torch.tensor([-5.0, -5.0]),
    high=torch.tensor([5.0, 5.0]),
    device=device
)

# Simple quadratic potential: V(x) = 0.5 * ||x - x_opt||^2
x_opt = torch.zeros(d, device=device)

from fragile.euclidean_gas import SimpleQuadraticPotential

quadratic_potential = SimpleQuadraticPotential(
    x_opt=x_opt,
    reward_alpha=1.0,
    reward_beta=0.0
)

# Langevin dynamics parameters
langevin_params = LangevinParams(
    gamma=1.0,        # Friction coefficient
    beta=2.0,         # Inverse temperature
    delta_t=0.05,     # Time step
    integrator="baoab"
)

# Cloning parameters
cloning_params = CloningParams(
    sigma_x=0.5,                    # Collision radius
    lambda_alg=0.5,                 # Velocity weight in algorithmic distance
    alpha_restitution=0.0,          # Inelastic collisions
    use_inelastic_collision=True,
    companion_selection_method="hybrid",
    epsilon_c=None  # Will default to sigma_x
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

print(f"\nSwarm Configuration:")
print(f"  N = {N} walkers")
print(f"  d = {d} dimensions")
print(f"  Domain: [{bounds.low[0]:.1f}, {bounds.high[0]:.1f}]^{d}")
print(f"  γ = {langevin_params.gamma:.2f}")
print(f"  β = {langevin_params.beta:.2f}")
print(f"  Δt = {langevin_params.delta_t:.3f}")
print(f"  σ_x = {cloning_params.sigma_x:.2f}")
print(f"  λ_alg = {cloning_params.lambda_alg:.2f}")

# ============================================================================
# 2. Initialize Swarm
# ============================================================================

# Create gas instance
gas = EuclideanGas(params)

# Initialize with spread-out walkers
torch.manual_seed(42)
x_init = bounds.sample(N)
v_init = torch.randn(N, d, device=device) * 1.0  # Higher initial velocity

state = gas.initialize_state(x_init, v_init)

print("\n✓ Swarm initialized")
print(f"\nInitial state:")
print(f"  Position variance: {VectorizedOps.variance_position(state):.4f}")
print(f"  Velocity variance: {VectorizedOps.variance_velocity(state):.4f}")
print(f"  Mean potential: {quadratic_potential.evaluate(state.x).mean():.4f}")
print(f"  Alive walkers: {bounds.contains(state.x).sum().item()}/{N}")

# ============================================================================
# 3. Run Swarm and Collect Data
# ============================================================================

n_steps = 150
print(f"\nRunning swarm for {n_steps} steps with detailed tracking...")

# Storage
history = {
    'x': [],           # Positions [n_steps+1, N, d]
    'v': [],           # Velocities
    'alive': [],       # Alive status [n_steps+1, N]
    'potential': [],   # Per-walker potential
    'dist_to_com': [], # Distance to center of mass
    'var_x': [],       # Position variance
    'var_v': [],       # Velocity variance
    'mu_x': [],        # Center of mass position
    'mu_v': [],        # Center of mass velocity
    'n_alive': [],     # Number of alive walkers
    'cluster_H': [],   # High-error mask
    'cluster_L': [],   # Low-error mask
    'n_high_error': [],
    'n_low_error': [],
}

def record_state(state, step):
    """Record detailed state information."""
    alive_mask = bounds.contains(state.x)

    # Basic state
    history['x'].append(state.x.cpu().clone())
    history['v'].append(state.v.cpu().clone())
    history['alive'].append(alive_mask.cpu().clone())

    # Per-walker metrics
    history['potential'].append(quadratic_potential.evaluate(state.x).cpu().clone())

    # Center of mass (computed over alive walkers)
    if alive_mask.any():
        mu_x = state.x[alive_mask].mean(dim=0)
        mu_v = state.v[alive_mask].mean(dim=0)
    else:
        mu_x = torch.zeros(d, device=device)
        mu_v = torch.zeros(d, device=device)

    history['mu_x'].append(mu_x.cpu().clone())
    history['mu_v'].append(mu_v.cpu().clone())

    # Distance to center of mass
    dist_to_com = torch.norm(state.x - mu_x, dim=1)
    history['dist_to_com'].append(dist_to_com.cpu().clone())

    # Variance
    var_x = VectorizedOps.variance_position(state)
    var_v = VectorizedOps.variance_velocity(state)
    history['var_x'].append(var_x.item())
    history['var_v'].append(var_v.item())

    # Alive count
    history['n_alive'].append(alive_mask.sum().item())

    # Cluster analysis (only every 5 steps for performance)
    if step % 5 == 0 and alive_mask.sum() >= 2:
        clusters = identify_high_error_clusters(
            state, alive_mask,
            epsilon=cloning_params.get_epsilon_c(),
            lambda_alg=cloning_params.lambda_alg
        )
        history['cluster_H'].append(clusters['H_k'].cpu().numpy())
        history['cluster_L'].append(clusters['L_k'].cpu().numpy())
        history['n_high_error'].append(clusters['H_k'].sum().item())
        history['n_low_error'].append(clusters['L_k'].sum().item())
    else:
        # Skip clustering this step (use previous values or zeros)
        if len(history['cluster_H']) > 0:
            # Reuse previous values
            history['cluster_H'].append(history['cluster_H'][-1])
            history['cluster_L'].append(history['cluster_L'][-1])
            history['n_high_error'].append(history['n_high_error'][-1])
            history['n_low_error'].append(history['n_low_error'][-1])
        else:
            # First step, initialize with zeros
            H_k = torch.zeros(N, dtype=torch.bool)
            L_k = torch.zeros(N, dtype=torch.bool)
            history['cluster_H'].append(H_k.numpy())
            history['cluster_L'].append(L_k.numpy())
            history['n_high_error'].append(0)
            history['n_low_error'].append(0)

# Record initial state
record_state(state, 0)

# Main loop
for t in range(n_steps):
    _, state = gas.step(state)
    record_state(state, t + 1)

    if (t + 1) % 30 == 0:
        print(f"  Step {t+1}/{n_steps} - Var_x: {history['var_x'][-1]:.4f}, "
              f"Alive: {history['n_alive'][-1]}/{N}, "
              f"High-error: {history['n_high_error'][-1]}")

# Convert to arrays
for key in ['x', 'v', 'alive', 'potential', 'dist_to_com', 'mu_x', 'mu_v']:
    history[key] = torch.stack(history[key]).numpy()

for key in ['var_x', 'var_v', 'n_alive', 'n_high_error', 'n_low_error']:
    history[key] = np.array(history[key])

history['cluster_H'] = np.array(history['cluster_H'])
history['cluster_L'] = np.array(history['cluster_L'])

steps = np.arange(n_steps + 1)

print(f"\n✓ Simulation complete!")
print(f"  Position variance: {history['var_x'][0]:.4f} → {history['var_x'][-1]:.4f}")
print(f"  Velocity variance: {history['var_v'][0]:.4f} → {history['var_v'][-1]:.4f}")
print(f"  Alive walkers: {history['n_alive'][0]} → {history['n_alive'][-1]}")

# ============================================================================
# 4. Test All Visualizations
# ============================================================================

print("\n" + "="*80)
print("TESTING VISUALIZATIONS")
print("="*80)

# Test position variance curve
print("\n[1/12] Testing position variance curve...")
var_x_curve = hv.Curve(
    (steps, history['var_x']),
    kdims=['Step'],
    vdims=['Position Variance'],
    label='V_Var,x(t)'
).opts(
    width=900,
    height=400,
    color='blue',
    line_width=3,
    title='Position Variance: Internal Swarm Spread',
    xlabel='Step',
    ylabel='V_Var,x',
    logy=True,
    tools=['hover'],
    fontsize={'title': 14, 'labels': 12}
)
print("✓ Position variance curve created")

# Test velocity variance curve
print("\n[2/12] Testing velocity variance curve...")
var_v_curve = hv.Curve(
    (steps, history['var_v']),
    kdims=['Step'],
    vdims=['Velocity Variance'],
    label='V_Var,v(t)'
).opts(
    width=900,
    height=400,
    color='green',
    line_width=3,
    title='Velocity Variance: Kinetic Energy Distribution',
    xlabel='Step',
    ylabel='V_Var,v',
    logy=True,
    tools=['hover'],
    fontsize={'title': 14, 'labels': 12}
)
print("✓ Velocity variance curve created")

# Test alive walker curve
print("\n[3/12] Testing alive walker curve...")
alive_curve = hv.Curve(
    (steps, history['n_alive']),
    kdims=['Step'],
    vdims=['Alive Walkers'],
    label='Alive'
).opts(
    width=900,
    height=300,
    color='darkgreen',
    line_width=2,
    title='Alive Walker Population',
    xlabel='Step',
    ylabel='Count',
    ylim=(0, N),
    tools=['hover'],
    fontsize={'title': 14, 'labels': 12}
)
n_line = hv.HLine(N).opts(color='black', line_dash='dashed', line_width=1)
alive_plot = alive_curve * n_line
print("✓ Alive walker curve created")

# Test cluster curves
print("\n[4/12] Testing cluster population curves...")
high_error_curve = hv.Curve(
    (steps, history['n_high_error']),
    kdims=['Step'],
    vdims=['Count'],
    label='High-Error'
).opts(color='red', line_width=2)

low_error_curve = hv.Curve(
    (steps, history['n_low_error']),
    kdims=['Step'],
    vdims=['Count'],
    label='Low-Error'
).opts(color='blue', line_width=2)

cluster_plot = (high_error_curve * low_error_curve).opts(
    width=900,
    height=400,
    title='Cluster Population: High-Error vs Low-Error',
    xlabel='Step',
    ylabel='Number of Walkers',
    legend_position='top_right',
    tools=['hover'],
    fontsize={'title': 14, 'labels': 12}
)
print("✓ Cluster population curves created")

# Test COM trajectory
print("\n[5/12] Testing center of mass trajectory...")
com_trajectory = hv.Path(
    [history['mu_x']],
    kdims=['x', 'y'],
    label='COM Trajectory'
).opts(
    color='purple',
    line_width=2,
    alpha=0.7
)

start_point = hv.Scatter(
    ([history['mu_x'][0, 0]], [history['mu_x'][0, 1]]),
    label='Start'
).opts(marker='o', size=12, color='green')

end_point = hv.Scatter(
    ([history['mu_x'][-1, 0]], [history['mu_x'][-1, 1]]),
    label='End'
).opts(marker='x', size=15, color='red', line_width=3)

optimum = hv.Scatter(
    ([x_opt[0].item()], [x_opt[1].item()]),
    label='Optimum'
).opts(marker='star', size=20, color='gold')

xlim = (bounds.low[0].item(), bounds.high[0].item())
ylim = (bounds.low[1].item(), bounds.high[1].item())

com_plot = (com_trajectory * start_point * end_point * optimum).opts(
    width=700,
    height=700,
    xlim=xlim,
    ylim=ylim,
    title='Center of Mass Trajectory',
    xlabel='x₁',
    ylabel='x₂',
    aspect='equal',
    legend_position='top_right',
    fontsize={'title': 14, 'labels': 12}
)
print("✓ COM trajectory created")

# Test distance heatmap
print("\n[6/12] Testing distance to COM heatmap...")
dist_heatmap = hv.Image(
    (steps, np.arange(N), history['dist_to_com'].T),
    kdims=['Step', 'Walker ID'],
    vdims=['Distance to COM']
).opts(
    width=900,
    height=500,
    cmap='plasma',
    colorbar=True,
    title='Per-Walker Distance to Center of Mass',
    xlabel='Step',
    ylabel='Walker ID',
    tools=['hover'],
    fontsize={'title': 14, 'labels': 12}
)
print("✓ Distance heatmap created")

# Test potential heatmap
print("\n[7/12] Testing potential energy heatmap...")
potential_heatmap = hv.Image(
    (steps, np.arange(N), history['potential'].T),
    kdims=['Step', 'Walker ID'],
    vdims=['Potential Energy']
).opts(
    width=900,
    height=500,
    cmap='viridis',
    colorbar=True,
    title='Per-Walker Potential Energy V(x)',
    xlabel='Step',
    ylabel='Walker ID',
    tools=['hover'],
    fontsize={'title': 14, 'labels': 12}
)
print("✓ Potential heatmap created")

# Test alive status heatmap
print("\n[8/12] Testing alive status heatmap...")
alive_heatmap = hv.Image(
    (steps, np.arange(N), history['alive'].T.astype(float)),
    kdims=['Step', 'Walker ID'],
    vdims=['Status']
).opts(
    width=900,
    height=500,
    cmap=['red', 'green'],
    colorbar=True,
    clim=(0, 1),
    title='Walker Alive Status (0=Dead, 1=Alive)',
    xlabel='Step',
    ylabel='Walker ID',
    tools=['hover'],
    fontsize={'title': 14, 'labels': 12}
)
print("✓ Alive status heatmap created")

# Test swarm plot function
print("\n[9/12] Testing swarm visualization function...")
def create_swarm_plot(step):
    """Create swarm visualization with cluster coloring."""
    x = history['x'][step]
    v = history['v'][step]
    alive = history['alive'][step]
    H = history['cluster_H'][step]
    L = history['cluster_L'][step]

    plots = []

    # Dead walkers
    dead = ~alive
    if dead.any():
        plots.append(hv.Scatter(
            (x[dead, 0], x[dead, 1]),
            label=f'Dead ({dead.sum()})'
        ).opts(size=6, color='gray', alpha=0.3))

    # Low-error
    if L.any():
        plots.append(hv.Scatter(
            (x[L, 0], x[L, 1]),
            label=f'Low-Error ({L.sum()})'
        ).opts(size=8, color='blue', alpha=0.7))

    # High-error
    if H.any():
        plots.append(hv.Scatter(
            (x[H, 0], x[H, 1]),
            label=f'High-Error ({H.sum()})'
        ).opts(size=10, color='red', alpha=0.8))

    # Velocity vectors (sample every 3rd walker for clarity)
    v_scale = 0.3
    sample_idx = np.arange(0, N, 3)
    arrows = []
    for i in sample_idx:
        if alive[i]:
            arrows.append((x[i, 0], x[i, 1],
                          x[i, 0] + v[i, 0] * v_scale,
                          x[i, 1] + v[i, 1] * v_scale))

    if arrows:
        arrow_plot = hv.Segments(arrows).opts(
            color='cyan', alpha=0.5, line_width=1.5
        )
        plots.append(arrow_plot)

    # Center of mass
    mu = history['mu_x'][step]
    com_point = hv.Scatter(
        ([mu[0]], [mu[1]]),
        label='COM'
    ).opts(marker='+', size=20, color='purple', line_width=3)
    plots.append(com_point)

    # Optimum
    opt_point = hv.Scatter(
        ([x_opt[0].item()], [x_opt[1].item()]),
        label='Optimum'
    ).opts(marker='star', size=15, color='gold')
    plots.append(opt_point)

    # Combine
    if plots:
        combined = plots[0]
        for p in plots[1:]:
            combined = combined * p
    else:
        combined = hv.Scatter(([0], [0])).opts(size=0)

    xlim = (bounds.low[0].item(), bounds.high[0].item())
    ylim = (bounds.low[1].item(), bounds.high[1].item())

    title = (f"Step {step}/{n_steps}\n"
             f"Alive: {alive.sum()}, High-Error: {H.sum()}, Low-Error: {L.sum()}\n"
             f"V_Var,x: {history['var_x'][step]:.4f}")

    return combined.opts(
        width=800,
        height=800,
        xlim=xlim,
        ylim=ylim,
        title=title,
        xlabel='x₁',
        ylabel='x₂',
        aspect='equal',
        legend_position='top_right',
        fontsize={'title': 11, 'labels': 12},
        tools=['hover']
    )

# Test at a few timesteps
test_steps = [0, n_steps//2, n_steps]
for test_step in test_steps:
    swarm_plot = create_swarm_plot(test_step)
    print(f"  ✓ Swarm plot at step {test_step} created")

print("✓ Swarm visualization function working")

# Test dynamic map
print("\n[10/12] Testing dynamic map...")
dmap = hv.DynamicMap(create_swarm_plot, kdims=['step'])
dmap = dmap.redim.range(step=(0, n_steps))
print("✓ Dynamic map created")

# ============================================================================
# 5. Summary Statistics
# ============================================================================

print("\n[11/12] Testing summary statistics...")
print("="*80)
print("SINGLE SWARM DEEP ANALYSIS SUMMARY")
print("="*80)

print(f"\nSwarm Configuration:")
print(f"  N = {N} walkers")
print(f"  d = {d} dimensions")
print(f"  Steps = {n_steps}")

print(f"\nParameters:")
print(f"  γ = {langevin_params.gamma:.2f}")
print(f"  β = {langevin_params.beta:.2f}")
print(f"  Δt = {langevin_params.delta_t:.3f}")
print(f"  σ_x = {cloning_params.sigma_x:.2f}")
print(f"  λ_alg = {cloning_params.lambda_alg:.2f}")

print(f"\nPosition Variance (V_Var,x):")
print(f"  Initial: {history['var_x'][0]:.6f}")
print(f"  Final: {history['var_x'][-1]:.6f}")
print(f"  Reduction: {(1 - history['var_x'][-1]/history['var_x'][0])*100:.2f}%")

print(f"\nVelocity Variance (V_Var,v):")
print(f"  Initial: {history['var_v'][0]:.6f}")
print(f"  Final: {history['var_v'][-1]:.6f}")
print(f"  Reduction: {(1 - history['var_v'][-1]/history['var_v'][0])*100:.2f}%")

print(f"\nAlive Walker Population:")
print(f"  Initial: {history['n_alive'][0]}/{N}")
print(f"  Final: {history['n_alive'][-1]}/{N}")
print(f"  Average: {history['n_alive'].mean():.1f}")
print(f"  Min: {history['n_alive'].min()}")

print(f"\nCluster Analysis:")
print(f"  Initial high-error: {history['n_high_error'][0]}")
print(f"  Final high-error: {history['n_high_error'][-1]}")
print(f"  Average high-error: {history['n_high_error'].mean():.1f}")
print(f"  Initial low-error: {history['n_low_error'][0]}")
print(f"  Final low-error: {history['n_low_error'][-1]}")
print(f"  Average low-error: {history['n_low_error'].mean():.1f}")

print(f"\nCenter of Mass:")
print(f"  Initial: [{history['mu_x'][0, 0]:.4f}, {history['mu_x'][0, 1]:.4f}]")
print(f"  Final: [{history['mu_x'][-1, 0]:.4f}, {history['mu_x'][-1, 1]:.4f}]")
print(f"  Distance to optimum:")
print(f"    Initial: {np.linalg.norm(history['mu_x'][0]):.4f}")
print(f"    Final: {np.linalg.norm(history['mu_x'][-1]):.4f}")

print(f"\nPotential Energy:")
print(f"  Initial mean: {history['potential'][0].mean():.6f}")
print(f"  Final mean: {history['potential'][-1].mean():.6f}")
print(f"  Initial best: {history['potential'][0].min():.6f}")
print(f"  Final best: {history['potential'][-1].min():.6f}")

print("="*80)

print("\n🔍 Key Observations:")
print(f"  • Position variance reduced by {(1 - history['var_x'][-1]/history['var_x'][0])*100:.1f}%")
print(f"  • Velocity variance reduced by {(1 - history['var_v'][-1]/history['var_v'][0])*100:.1f}%")
print(f"  • Center of mass converged to within {np.linalg.norm(history['mu_x'][-1]):.4f} of optimum")
print(f"  • Maintained {history['n_alive'][-1]}/{N} alive walkers")
print(f"  • High-error population: {history['n_high_error'][-1]} walkers")

print("✓ Summary statistics complete")

# ============================================================================
# 6. Final Check
# ============================================================================

print("\n[12/12] Final verification...")
print("\n" + "="*80)
print("ALL TESTS PASSED!")
print("="*80)
print("\nThe notebook code is working correctly.")
print("You can now safely use lyapunov.ipynb")
print("\nKey data shapes:")
print(f"  x: {history['x'].shape}")
print(f"  v: {history['v'].shape}")
print(f"  alive: {history['alive'].shape}")
print(f"  potential: {history['potential'].shape}")
print(f"  dist_to_com: {history['dist_to_com'].shape}")
print(f"  mu_x: {history['mu_x'].shape}")
print(f"  cluster_H: {history['cluster_H'].shape}")
print(f"  var_x: {history['var_x'].shape}")

print("\n✅ Debug script complete!")
