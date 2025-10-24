# Multimodal Exploration Tuning Guide

## Overview

The default parameters in `GasConfig` have been optimized for **multimodal exploration** on complex landscapes with multiple modes/basins. The previous defaults were too exploitation-focused and caused walkers to collapse into a single mode.

## Updated Default Parameters

### Core Fitness Parameters (Exploitation vs Exploration Balance)

| Parameter | Old Default | New Default | Purpose |
|-----------|-------------|-------------|---------|
| `alpha_fit` | 0.7 | **0.4** | Reward channel exponent (lower = less exploitation) |
| `beta_fit` | 1.3 | **2.5** | Diversity channel exponent (higher = stronger repulsion) |
| `sigma_x` | 0.15 | **0.5** | Cloning jitter (higher = more spatial spread after cloning) |
| `lambda_alg` | 0.5 | **0.2** | Velocity weight in distance (lower = position-focused diversity) |
| `eta` | 0.01 | **0.003** | Positivity floor (lower = larger fitness ratios) |
| `A` | 2.0 | **3.5** | Logistic rescale amplitude (higher = larger dynamic range) |
| `epsilon_F` | 0.0 | **0.15** | Fitness force rate (disabled by default, but ready to enable) |

### Key Principle: β >> α

The fitness formula is:
```
V_i = (d'_i)^β · (r'_i)^α
```

Where:
- **r'_i**: Rescaled reward (exploitation channel)
- **d'_i**: Rescaled distance to companion (diversity/exploration channel)

**For multimodal exploration:** β should be significantly larger than α (ratio ≈ 6:1 with defaults)

## Why These Changes Work

### 1. Higher β (2.5 vs 1.3)
- Makes fitness **exponentially more sensitive** to distance differences
- Creates stronger **repulsion** between nearby walkers
- Encourages walkers to spread across different modes
- Formula impact: `d^2.5` vs `d^1.3` ≈ 1.9× stronger distance sensitivity

### 2. Lower α (0.4 vs 0.7)
- Reduces reward channel's dominance
- Prevents premature convergence to single high-reward mode
- Makes fitness less sensitive to reward differences
- Allows distance (diversity) to guide exploration more

### 3. Larger σ_x (0.5 vs 0.15)
- When walkers clone: `x'_i = x_companion + σ_x * ζ` where ζ ~ N(0, I)
- Larger jitter spreads cloned walkers further from their companions
- Prevents **positional collapse** where all walkers stack together
- Essential for maintaining spatial diversity after cloning

### 4. Lower λ_alg (0.2 vs 0.5)
- Distance formula: `d_alg = sqrt(||Δx||² + λ_alg ||Δv||²)`
- Lower λ_alg emphasizes **spatial separation** over velocity matching
- Better for finding different spatial modes
- High λ_alg makes walkers pair by kinematic similarity (not helpful for mode discovery)

### 5. Larger A (3.5 vs 2.0)
- Rescale formula: `r' = A / (1 + exp(-Z_r))`
- Larger A increases dynamic range of rescaled values
- Helps Z-scores express differences when variance is low (clustering)
- Prevents "normalization collapse" when walkers cluster

### 6. Lower η (0.003 vs 0.01)
- Fitness floor: `V_i = (d' + η)^β (r' + η)^α`
- Lower η allows larger **ratios** between high and low fitness values
- Example: If d'_max = 3.5, ratio = 3500:1 (vs 350:1 with old default)
- Stronger selection pressure for diversity

## Diagnostic Metrics

Monitor these during runs to verify multimodal exploration:

1. **Distance variance**: `std(distances)` should remain high
   - Collapse indicator: Sharp drop early in run
   - Healthy: Sustained high variance throughout

2. **Companion distance distribution**: Histogram of `d_alg(i, companion_i)`
   - Trapped: Narrow peak (everyone close together)
   - Exploring: Broad distribution

3. **Fitness channel balance**: Compare `β·log(d')` vs `α·log(r')`
   - Balanced: Contributions roughly equal
   - Over-exploiting: Reward term dominates

4. **Mode occupancy**: Number of spatial clusters over time
   - Healthy: Multiple persistent clusters
   - Collapsed: Single cluster

## When to Adjust

### Still collapsing into single mode?
- **Increase β** to 3.0-4.0 (stronger diversity pressure)
- **Enable fitness force**: `use_fitness_force=True` with `epsilon_F=0.15`
- **Try localized statistics**: Set `rho=2.5` in `FitnessOperator`

### Walkers too dispersed (ignoring rewards)?
- **Increase α** to 0.5-0.6 (stronger exploitation)
- **Decrease β** to 2.0 (weaker diversity pressure)
- Balance: Keep β > 2α as rule of thumb

### Unstable/chaotic dynamics?
- **Reduce σ_x** to 0.3-0.4 (less spatial noise)
- **Increase γ** (more friction, dampen velocity)
- **Reduce ε_F** if using fitness force

## Advanced Techniques

### 1. Adaptive β Scheduling
Start with high β (exploration), decrease over time (exploitation):
```python
beta_t = 3.0 + (1.5 - 3.0) * (t / n_steps)
```

### 2. Localized Statistics (ρ-regime)
Use finite `rho` parameter in `FitnessOperator`:
- Each walker computes statistics from local neighborhood
- Prevents normalization collapse when one mode dominates
- Cost: O(N²) vs O(N) for global statistics
- Try: `rho=2.0` to `3.0`

### 3. Softmax Companion Selection
Switch from uniform random to softmax:
```python
companion_method = "softmax"
companion_epsilon = 0.2  # Lower = prefer distant companions
```

## Theoretical Background

### The Keystone Principle
From `03_cloning.md`: Cloning creates **virtual reward** through distance dependence.

The fitness potential `V = d^β · r^α` combines:
- **High-fitness persistence** (reward term r^α)
- **Diversity maintenance** (distance term d^β)

When β >> α:
- Walkers get "bonus fitness" for being far from companions
- Opposes clustering, encourages spatial spread
- Discovers multiple modes through coverage
- Reward becomes tiebreaker among distant positions

### Why Old Defaults Failed
With β=1.3 and α=0.7:
- Ratio β/α = 1.86 (too low)
- Reward dominated fitness landscape
- Cloning became purely exploitative
- Z-score standardization compressed differences when clustered
- Positive feedback loop: clustering → low variance → weak diversity signal → more clustering

### Why New Defaults Succeed
With β=2.5 and α=0.4:
- Ratio β/α = 6.25 (strong diversity preference)
- Distance dominates initial exploration
- Spatial coverage discovers modes
- Maintained diversity prevents variance collapse
- Balanced exploration-exploitation trade-off

## Quick Reference

**Multimodal exploration preset:**
```python
config = GasConfig(dims=2)
# Defaults are now optimized for multimodal exploration!
# No changes needed for most cases
```

**Extreme exploration (many modes):**
```python
config.alpha_fit = 0.3      # Minimal exploitation
config.beta_fit = 3.5       # Maximum diversity
config.sigma_x = 0.7        # High spatial spread
config.use_fitness_force = True
config.epsilon_F = 0.2
```

**Balanced (few well-separated modes):**
```python
config.alpha_fit = 0.5      # Moderate exploitation
config.beta_fit = 2.0       # Moderate diversity
config.sigma_x = 0.4        # Moderate spread
# (Keep other defaults)
```

## References

- Fitness formula: `src/fragile/core/fitness.py:420`
- Cloning with jitter: `src/fragile/core/cloning.py:172-193`
- Algorithmic distance: `src/fragile/core/fitness.py:384`
- Theoretical framework: `docs/source/1_euclidean_gas/03_cloning.md`
- Parameter optimization: `docs/source/1_euclidean_gas/06_convergence.md`
