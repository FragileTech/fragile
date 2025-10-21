# Scutoid Metric Correction Quick Reference

## Quick Start

```python
from fragile.core.scutoids import create_scutoid_history

# Choose correction mode when creating scutoid history
scutoid_hist = create_scutoid_history(
    history,
    metric_correction='diagonal'  # 'none', 'diagonal', or 'full'
)

# Everything else is the same
scutoid_hist.build_tessellation()
scutoid_hist.compute_ricci_scalars()  # Auto-applies correction
ricci = scutoid_hist.get_ricci_scalars()  # Returns corrected values
```

## When to Use Each Mode

| Mode | Use When | Cost | Accuracy |
|------|----------|------|----------|
| `'none'` | No anisotropic diffusion | O(N log N) | Baseline |
| `'diagonal'` | Need speed, weak anisotropy | O(N log N) | Good |
| `'full'` | Need accuracy, strong anisotropy | O(N·k log N) | Best |

## What Each Mode Computes

### `'none'` - No Correction
- **Computes**: Flat-space deficit angles only
- **Measures**: Intrinsic curvature of walker configuration
- **Ignores**: Fitness landscape geometry

### `'diagonal'` - Fast Approximation  
- **Computes**: Flat + diagonal metric correction
- **Measures**: Local density scale effects
- **Formula**: `ΔR ≈ (1/2)Σ_k ∂²g_kk/∂x_k²`

### `'full'` - Accurate Coupling
- **Computes**: Flat + full metric tensor correction
- **Measures**: Complete metric gradient effects
- **Formula**: `ΔR = (1/2)∇²(tr h) - (1/4)||∇h||²`

## Accessing Results

```python
# Get appropriate Ricci scalars based on mode
ricci = scutoid_hist.get_ricci_scalars()

# Access both flat and corrected separately
ricci_flat = scutoid_hist.ricci_scalars           # Always available
ricci_corrected = scutoid_hist.ricci_scalars_corrected  # If correction enabled

# Check which mode was used
print(scutoid_hist.metric_correction)  # 'none', 'diagonal', or 'full'
```

## Common Patterns

### Compare All Modes
```python
for mode in ['none', 'diagonal', 'full']:
    sh = create_scutoid_history(history, metric_correction=mode)
    sh.build_tessellation()
    sh.compute_ricci_scalars()
    ricci = sh.get_ricci_scalars()
    print(f"{mode:10s}: mean={np.mean(ricci[~np.isnan(ricci)]):.4f}")
```

### Check Correction Magnitude
```python
sh = create_scutoid_history(history, metric_correction='full')
sh.build_tessellation()
sh.compute_ricci_scalars()

flat = sh.ricci_scalars
corrected = sh.ricci_scalars_corrected

valid_mask = ~np.isnan(flat) & ~np.isnan(corrected)
correction = corrected[valid_mask] - flat[valid_mask]
print(f"Correction magnitude: {np.mean(np.abs(correction)):.4f}")
```

## Expected Behavior

### Flat Fitness (V = 0)
- Correction ≈ 0 (no metric perturbation)
- All modes give similar results

### Quadratic Bowl (V = ½x^T A x)
- Correction ~ tr(A) (Hessian trace)
- Corrected values more physically meaningful

### High Curvature Regions
- Large corrections where ||H|| is large
- Full mode most accurate
- Diagonal mode captures essential scale

### At Equilibrium
- Walkers adapted to anisotropic diffusion
- Deficit angles reflect curved metric
- Correction provides explicit coupling

## Troubleshooting

### "No valid Ricci scalars"
- Not enough walkers at some timesteps
- Check `history.alive_mask`

### Correction seems too large/small
- Check if anisotropic diffusion is enabled in kinetic operator
- Verify fitness landscape has non-trivial Hessian

### Full mode too slow
- Use diagonal mode for exploratory analysis
- Switch to full for final results

## Mathematical Background

The corrections implement first-order perturbation theory:

```
R^manifold ≈ R^flat + ΔR^metric
```

Where:
- `R^flat`: From deficit angles (intrinsic geometry)
- `ΔR^metric`: From fitness Hessian (extrinsic geometry)

At equilibrium, this approximates the Ricci scalar of the emergent metric `g = H + ε_Σ I` without expensive Riemannian Voronoi computation.

## See Also

- `SCUTOID_METRIC_CORRECTION.md`: Full mathematical details
- `src/fragile/core/scutoids.py`: Implementation
- `example_scutoid_metric_correction.py`: Demo script
