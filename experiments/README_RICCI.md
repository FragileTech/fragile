# Ricci Fragile Gas Experiments

This directory contains experiments for the **Ricci Fragile Gas**, a novel geometry-driven swarm optimization algorithm defined in [`docs/source/12_fractal_gas.md`](../docs/source/12_fractal_gas.md).

## Quick Start

```bash
# Run all experiments
python experiments/ricci_gas_experiments.py --experiment all

# Run specific experiment
python experiments/ricci_gas_experiments.py --experiment phase_transition
python experiments/ricci_gas_experiments.py --experiment ablation
python experiments/ricci_gas_experiments.py --experiment toy_problems
python experiments/ricci_gas_experiments.py --experiment heatmap
```

Results are saved to `experiments/ricci_gas_results/` with visualizations.

## Theory Summary

The Ricci Gas uses **geometric curvature** of the emergent manifold to drive exploration:

### Push-Pull Architecture

**Pull (Gravity):** Force aggregates toward high curvature
```
F = +ε_R ∇R    (walkers attracted to dense, curved regions)
```

**Push (Anti-Gravity):** Cloning rewards low curvature
```
Reward ∝ 1/R    (walkers in flat regions clone more)
```

### Phase Transition Hypothesis

At critical feedback strength α_c:
- **α < α_c**: Diffuse gas-like state, LSI holds
- **α > α_c**: Concentrated structures, LSI breaks down (this is a feature!)

### Ricci Curvature Proxy (3D)

```python
R(x, S) = tr(H(x, S)) - λ_min(H(x, S))
```

where H is the Hessian of the fitness potential computed via Kernel Density Estimation.

## Experiments

### 1. Phase Transition Detection (`--experiment phase_transition`)

**Goal**: Find α_c empirically

**Method**:
- Run simulations for α ∈ {0.01, 0.1, 0.5, 1.0, 2.0}
- Measure: variance, entropy, max curvature
- Plot time evolution and final states

**Expected outcome**:
- Small α → stable, diffuse distribution
- Large α → collapse, low variance, high curvature spikes

### 2. Ablation Study (`--experiment ablation`)

**Goal**: Compare push-pull vs alternatives

**Variants**:
| Variant | Force | Reward | Hypothesis |
|---------|-------|--------|------------|
| A (Ricci Gas) | +∇R | 1/R | Push-pull → structured collapse |
| B (Aligned) | -∇R | 1/R | Both seek flat → broad exploration |
| C (Force Only) | +∇R | 0 | Pure aggregation → total collapse |
| D (Reward Only) | 0 | 1/R | Pure dispersion → diffuse gas |

**Expected outcome**: Variant A shows richest dynamics

### 3. Toy Problems (`--experiment toy_problems`)

**Goal**: Test on known 3D optimization landscapes

**Problems**:
1. **Double-well**: V(x) = (x² - 1)² + y² + z²
   - Two minima at (±1, 0, 0)
   - Test: Does curvature guide to minima?

2. **3D Rastrigin**: V(x) = 10d + Σ(x_i² - 10cos(2πx_i))
   - Many local minima (fractal-like)
   - Test: Does negative curvature correlate with basins?

### 4. Curvature Heatmap (`--experiment heatmap`)

**Goal**: Visualize emergent geometry

**Output**:
- 2D slice (z=0) of Ricci curvature field
- Walker positions overlaid
- Color-coded by local curvature

## Customization

### Create Custom Variant

```python
from fragile.ricci_gas import RicciGasParams, RicciGas

params = RicciGasParams(
    epsilon_R=0.8,           # Feedback strength (α)
    kde_bandwidth=0.3,       # Smoothing length (ℓ)
    force_mode="pull",       # "pull", "push", or "none"
    reward_mode="inverse",   # "inverse", "negative", or "none"
    R_crit=10.0,             # Singularity threshold
    gradient_clip=5.0,       # Numerical stability
)

gas = RicciGas(params)
```

### Experimental Loop

```python
from fragile.ricci_gas import SwarmState
import torch

# Initialize
N, d = 100, 3
state = SwarmState(
    x=torch.randn(N, d),
    v=torch.randn(N, d) * 0.1,
    s=torch.ones(N),
)

# Run
for t in range(1000):
    # Compute curvature
    R, H = gas.compute_curvature(state, cache=True)

    # Get reward (for cloning)
    reward = gas.compute_reward(state)

    # Get force (for Langevin)
    force = gas.compute_force(state)

    # Simple Langevin step
    state.v = 0.9 * state.v + 0.1 * force + torch.randn_like(state.v) * 0.05
    state.x = state.x + state.v * 0.1

    # Singularity regulation
    state = gas.apply_singularity_regulation(state)
```

## Computational Requirements

For N walkers in 3D:
- **CPU**: ~1-10 seconds per iteration (N=1000)
- **GPU**: ~0.1-1 seconds per iteration (N=10000)
- **Memory**: O(N²) for KDE Hessian (bottleneck)

**Optimization tips**:
- Use `truncation_radius` to limit KDE neighbors
- Enable `use_tree_kde=True` for faster (approximate) KDE
- Increase `kde_bandwidth` to smooth more aggressively

## Interpreting Results

### Signs of Subcritical Regime (α < α_c)
- Variance remains ~constant or grows
- Entropy remains high
- Max curvature stays bounded
- Walkers explore broadly

### Signs of Supercritical Regime (α > α_c)
- Variance decreases sharply → 0
- Entropy drops (concentration)
- Max curvature diverges
- Walkers collapse into dense clusters

### Phase Transition Indicators
- **Sharp drop** in variance vs α
- **Kink** in entropy curve
- **Divergence** in max curvature

## Troubleshooting

**Problem**: NaN in curvature
- **Cause**: Hessian eigenvalues too extreme
- **Fix**: Increase `epsilon_Sigma`, decrease `epsilon_R`, or increase `kde_bandwidth`

**Problem**: All walkers dead
- **Cause**: R_crit too low
- **Fix**: Increase `R_crit` or set to `None`

**Problem**: No structure formation
- **Cause**: α too small
- **Fix**: Increase `epsilon_R` above estimated α_c

**Problem**: Immediate collapse
- **Cause**: α too large
- **Fix**: Decrease `epsilon_R`, check for numerical instability

## Next Steps

After running experiments:

1. **Analyze phase diagram**: Plot final variance vs α, identify α_c
2. **Visualize geometry**: Use heatmaps to see curvature evolution
3. **Compare with baselines**: Run Euclidean/Adaptive Gas for comparison
4. **Tune for application**: Adjust parameters for specific 3D physics problem

## References

- **Theory**: `docs/source/12_fractal_gas.md`
- **Implementation**: `src/fragile/ricci_gas.py`
- **Mathematical Framework**: `docs/source/02_euclidean_gas.md`, `07_adaptative_gas.md`, `08_emergent_geometry.md`

## Contact

For questions about the Ricci Gas theory or implementation, see `CLAUDE.md` in the project root.
