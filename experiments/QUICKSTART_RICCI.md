# Ricci Fragile Gas - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Run the Interactive Notebook

```bash
cd /home/guillem/fragile
jupyter notebook experiments/ricci_gas_visualization.ipynb
```

**What you'll see:**
- 3D walkers colored by Ricci curvature
- Emergent Riemannian manifold visualization
- Curvature field isosurfaces
- Lennard-Jones cluster optimization

### 2. Run Automated Experiments

```bash
python experiments/ricci_gas_experiments.py --experiment all
```

**Results saved to:** `experiments/ricci_gas_results/`

**Plots generated:**
- `phase_transition/phase_transition.png` - Time evolution for different α values
- `phase_transition/phase_diagram.png` - Variance/entropy vs α (find α_c here!)
- `ablation/ablation.png` - Compare 4 variants
- `toy_problems/double_well_optimization.png`
- `toy_problems/rastrigin_optimization.png`
- `visualization/curvature_heatmap.png`

### 3. Run a Single Variant

```python
from fragile.ricci_gas import RicciGas, RicciGasParams, SwarmState
import torch

# Create Ricci Gas
params = RicciGasParams(
    epsilon_R=0.5,          # ← Try varying this!
    kde_bandwidth=0.3,
    force_mode="pull",
    reward_mode="inverse",
)
gas = RicciGas(params)

# Initialize swarm
state = SwarmState(
    x=torch.randn(100, 3),
    v=torch.randn(100, 3) * 0.1,
    s=torch.ones(100),
)

# Single iteration
R, H = gas.compute_curvature(state, cache=True)
force = gas.compute_force(state)
reward = gas.compute_reward(state)

print(f"Ricci range: [{R.min():.2f}, {R.max():.2f}]")
```

## 🎯 Key Parameters to Experiment With

### Finding the Phase Transition

The **critical parameter** is `epsilon_R` (feedback strength α):

```python
# Subcritical (α < α_c): Stable, diffuse gas
params_sub = RicciGasParams(epsilon_R=0.1)

# Near-critical (α ≈ α_c): Interesting dynamics
params_crit = RicciGasParams(epsilon_R=0.5)

# Supercritical (α > α_c): Collapse to structures
params_super = RicciGasParams(epsilon_R=2.0)
```

**Expected behavior:**
- **Low α**: Variance stays constant, entropy high
- **High α**: Variance → 0, entropy drops, R_max diverges

### The Four Variants (Ablation Study)

```python
from fragile.ricci_gas import create_ricci_gas_variants

variants = create_ricci_gas_variants()

# A: Ricci Gas (push-pull) - Force=+∇R, Reward=1/R
gas_ricci = RicciGas(variants['ricci'])

# B: Aligned (both seek flat) - Force=-∇R, Reward=1/R
gas_aligned = RicciGas(variants['aligned'])

# C: Force only (pure aggregation) - Force=+∇R, Reward=0
gas_force = RicciGas(variants['force_only'])

# D: Reward only (pure dispersion) - Force=0, Reward=1/R
gas_reward = RicciGas(variants['reward_only'])
```

## 🔬 Physics Problem: Lennard-Jones Clusters

The notebook includes a **Lennard-Jones cluster optimization** example:

**Problem:** Find minimum energy configuration of 13 argon atoms

**Known result:** Global minimum E = -44.327 (perfect icosahedron)

**Ricci Gas approach:**
1. Use curvature to escape local minima
2. Explore saddle points (negative curvature regions)
3. Find basins of attraction

**To run:**
- Open notebook → Scroll to Section 10
- Or modify `N_atoms = 13` to try different cluster sizes

**Interesting cases:**
- N=13: Icosahedron (classic benchmark)
- N=19: Stacked structure
- N=38: Truncated octahedron
- N=55: Mackay icosahedron

## 📊 Interpreting Results

### Phase Transition Indicators

**Subcritical regime (α < α_c):**
```
✓ Variance ≈ constant or grows
✓ Entropy remains high (>1.5)
✓ R_max < 10
✓ Walkers explore broadly
```

**Supercritical regime (α > α_c):**
```
⚠ Variance drops sharply → 0
⚠ Entropy collapses (<0.5)
⚠ R_max → ∞ (or R_crit if limited)
⚠ Walkers form dense clusters
```

**How to find α_c:**
1. Run phase transition experiment
2. Plot final variance vs α (log scale)
3. Look for sharp drop → that's α_c!

### Visualization Interpretation

**Curvature Heatmap:**
- **Red** (R > 0): High curvature, dense regions
- **Blue** (R < 0): Negative curvature, saddle points
- **White** (R ≈ 0): Flat regions

**Emergent Manifold:**
- **Large markers**: High mean eigenvalue (stiff metric)
- **Bright color**: High anisotropy (directional bias)
- **Dark color**: Isotropic metric

## 🛠️ Troubleshooting

### Problem: NaN in curvature

**Symptoms:** `R` contains NaN or Inf

**Causes:**
- Hessian eigenvalues too extreme
- KDE bandwidth too small
- Numerical instability in gradient

**Fixes:**
```python
params = RicciGasParams(
    epsilon_Sigma=0.1,      # ← Increase (was 0.01)
    kde_bandwidth=0.5,      # ← Increase (was 0.3)
    gradient_clip=5.0,      # ← Decrease (was 10.0)
)
```

### Problem: All walkers dead

**Symptoms:** `alive_fraction → 0`

**Cause:** `R_crit` set too low, walkers enter singularity and die

**Fix:**
```python
params = RicciGasParams(
    R_crit=20.0,  # ← Increase
    # Or disable:
    R_crit=None,
)
```

### Problem: No structure formation

**Symptoms:** Variance stays constant, looks like random walk

**Cause:** α too small (subcritical regime)

**Fix:**
```python
params = RicciGasParams(
    epsilon_R=1.0,  # ← Increase from 0.1
)
```

### Problem: Immediate total collapse

**Symptoms:** All walkers at same point, variance = 0 at t=10

**Cause:** α too large (deep supercritical)

**Fix:**
```python
params = RicciGasParams(
    epsilon_R=0.3,  # ← Decrease from 2.0
)
```

## 📈 Next Steps

### Beginner
1. Run notebook, see visualizations
2. Vary `epsilon_R` and observe behavior
3. Try ablation study (4 variants)

### Intermediate
1. Run full experiment suite
2. Identify α_c empirically
3. Test on Lennard-Jones with different N

### Advanced
1. Implement full cloning operator
2. Compare with Euclidean/Adaptive Gas
3. Apply to your own 3D physics problem
4. Contribute to mathematical proofs (see `docs/source/12_fractal_gas.md`)

## 📚 References

**Theory:**
- Main document: `docs/source/12_fractal_gas.md`
- Mathematical framework: `docs/source/01_fragile_gas_framework.md`
- Convergence theory: `docs/source/04_convergence.md`

**Implementation:**
- Core code: `src/fragile/ricci_gas.py`
- Experiments: `experiments/ricci_gas_experiments.py`
- Visualization: `experiments/ricci_gas_visualization.ipynb`

**Related Work:**
- Patlak-Keller-Segel equations (chemotaxis)
- Ricci flow (differential geometry)
- Natural gradient methods (information geometry)

## 💡 Research Ideas

1. **Prove α_c for 3D:** Derive analytical bound or exact value
2. **Fractal dimension:** Does supercritical QSD have fractal structure?
3. **Neural proxy:** Train NN to approximate R[ρ] for speed
4. **Hybrid rewards:** Combine Ricci with task-specific reward
5. **Higher dimensions:** Generalize beyond d=3
6. **Other curvatures:** Try Weyl curvature, Einstein tensor

## 🤝 Contributing

Found a bug? Have an idea? See `CLAUDE.md` and `GEMINI.md` for collaboration workflow.

---

**Happy exploring! 🌌**

*"Matter tells spacetime how to curve, spacetime tells matter how to move."* — Wheeler
