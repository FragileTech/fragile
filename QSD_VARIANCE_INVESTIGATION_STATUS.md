# QSD Variance Investigation: Current Status

**Date**: 2025-10-24
**Investigation**: Does QSD achieve near-maximal variance for O(N^{3/2}) edge budget?

---

## Executive Summary

Following the discovery that the original QSD variance experiment used an **incorrect test case** (unimodal potential), we redesigned the experiment to test **multimodal landscapes** with systematic **α/β parameter sweeps**.

**Current Status**: Full 225-experiment grid sweep is running (~9.4 hours, 1/225 complete).

**Initial finding**: Multimodal + high diversity shows **2x improvement** (0.152 vs 0.07) but still **far below critical threshold** (0.45 needed).

---

## Background: The Critical Variance Issue

From Phase-Space Packing Lemma (`03_cloning.md:2420-2550`):

$$
N_{\text{close}} \le \binom{K}{2} \cdot \frac{D_{\text{max}}^2 - 2\mathrm{Var}_h}{D_{\text{max}}^2 - d_{\text{close}}^2}
$$

For **O(N^{3/2}) edge budget** with d_close = D_max/√N, this requires:

$$
\boxed{\mathrm{Var}_h \ge \frac{D_{\text{max}}^2}{2} - O\left(\frac{D_{\text{max}}^2}{\sqrt{N}}\right)}
$$

**Ratio needed**: Var_h / D²_max ≈ **0.45-0.50** (near-maximal variance)

---

## Discovery of Experimental Flaw

### Original Experiment (WRONG)

**Setup**:
- Potential: U(x) = 0.1·||x||²/2 (unimodal quadratic)
- Single confinement well at origin
- All walkers confined to small region near equilibrium

**Result**:
- Variance ratio: **0.0723** (7.23%)
- Far below critical threshold

**Conclusion (PREMATURE)**: "O(N^{3/2}) edge budget unprovable"

### Critical Insight from User

> **User**: "have you done the experiment with multimodal potentials? those would form clusters and thus have high variance from the walkers exploring the different modes"

**Key realization**:
- ✅ **Unimodal** potential → walkers confined → LOW variance (expected)
- ❓ **Multimodal** potential → walkers distributed across modes → HIGH variance (untested!)

The hierarchical clustering proof **assumes multimodal landscape** with walkers exploring different regions!

### Second Critical Insight

> **User**: "also alpha and beta exponents will affect"

**α (alpha_fit)**: Reward exponent in fitness function V = d^β · r^α
- Higher α → stronger exploitation → fitness-driven clustering

**β (beta_fit)**: Diversity exponent in fitness function
- Higher β → stronger diversity → anti-clustering, spreading walkers

**Hypothesis**: Multimodal + high diversity (β >> α) → walkers spread across modes → HIGH variance

---

## Corrected Experimental Design

### Multimodal Potential Functions

#### 1. Gaussian 4-Mode Mixture
```
U(x) = -Σ_k depth_k · exp(-||x - center_k||²/(2·width²_k))
```
- Centers: [(±5, ±5)] - 4 corners of domain
- Depths: [1.0, 1.0, 1.0, 1.0] - equal attractiveness
- Widths: [1.5, 1.5, 1.5, 1.5] - moderate basin size
- Inter-mode distance: ~10 (vs D_max ≈ 14.14)

#### 2. Rastrigin Periodic Lattice
```
U(x) = A·d + Σ[x_i² - A·cos(ω·x_i)]
```
- Regular lattice of ~9 modes in [-5,5]² domain
- Tests uniform mode exploration

#### 3. Unimodal Quadratic (Control)
- Baseline for comparison

### Full Parameter Grid Sweep

**Dimensions**:
- **Potentials**: 3 types (quadratic, Gaussian 4-mode, Rastrigin)
- **Alpha (exploitation)**: [0.1, 0.5, 1.0, 3.0, 10.0]
- **Beta (diversity)**: [0.1, 0.5, 1.0, 3.0, 10.0]
- **Swarm sizes**: N ∈ [50, 100, 200]

**Total**: 3 × 5 × 5 × 3 = **225 experiments**

**Key test cases**:
- **(α=10, β=0.1)**: Strong exploitation, weak diversity → collapse to one mode → LOW variance
- **(α=0.1, β=10)**: Weak exploitation, strong diversity → uniform spread → HIGH variance (?)
- **(α=1.0, β=1.0)**: Balanced (current default)

---

## Initial Test Result

**Configuration**: Gaussian 4-mode, N=50, α=0.5, β=5.0 (high diversity)

### Results
```
Variance ratio: 0.1520 ± 0.0241
  Critical threshold: 0.45

Edge budget:
  N_close: 8.84e+02
  Scaling exponent: 1.734
```

### Interpretation

**✅ Partial confirmation**: Multimodal + high diversity DOES increase variance
- **Improvement**: 0.152 vs 0.07 (unimodal) = **2.17x higher**
- **But insufficient**: Still **3x below** critical threshold (0.152 vs 0.45)

**Scaling exponent**: 1.734 (between 1.5 and 2.0, closer to O(N²))

**Hypothesis status**:
- ✅ Multimodal landscapes increase variance
- ❌ But not enough to reach O(N^{3/2}) regime (yet)
- ⚠️ Need to test all 225 combinations to see if ANY achieves threshold

---

## Current Experiment: Full Sweep Running

**Started**: 2025-10-24 10:51:30 UTC
**Status**: 1/225 experiments complete
**Progress**: ~5.4 seconds per experiment
**Estimated completion**: ~9.4 hours from start

**Output files**:
- Partial results (every 10 exps): `src/fragile/theory/qsd_variance_sweep_partial.csv`
- Final results: `src/fragile/theory/qsd_variance_sweep_results.csv`
- Log: `/tmp/qsd_sweep_output.log`

**Current experiment**: quadratic N=50 α=0.1 β=0.5

---

## Decision Criteria

After full sweep completes, we will:

### Check if ANY configuration achieves high variance

**Success criterion**: Variance ratio ≥ 0.45 in at least one (potential, α, β, N) combination

#### If YES (ratio ≥ 0.45):
- ✅ **O(N^{3/2}) edge budget IS achievable**
- ✅ **Hierarchical clustering proof IS feasible**
- Document which parameter regimes enable high variance
- Revise proof to specify required conditions

#### If NO (all ratios < 0.45):
- ❌ **O(N^{3/2}) edge budget UNPROVABLE** with realistic parameters
- ❌ **Hierarchical clustering proof via edge-counting FAILS**
- Edge budget is **O(N²)**
- Global regime concentration exp(-c√N) remains **UNPROVEN**
- Need **alternative proof strategy**

### Alternative Strategies (if sweep fails)

1. **Accept O(N²) budget**: Document limitation, explore other approaches
2. **Distance-sensitive covariance decay**: Prove |Cov(ξ_i, ξ_j)| = O(1/N³) for distant pairs
3. **Entropic/optimal transport arguments**: Information-theoretic approach
4. **Mean-field PDE analysis**: McKean-Vlasov clustering structure
5. **Numerical + asymptotic**: Verify L = Θ(√N) numerically, develop expansion

---

## Key Regime Predictions

Based on fitness function **V = d^β · r^α**:

| α (exploitation) | β (diversity) | Expected Behavior | Variance |
|------------------|---------------|-------------------|----------|
| 10.0 | 0.1 | Collapse to dominant peak | **LOW** |
| 0.1 | 10.0 | Uniform spread across modes | **HIGH** (?) |
| 10.0 | 10.0 | Competition between forces | **Medium** |
| 0.5 | 5.0 | High diversity (tested) | **0.152** |
| 1.0 | 1.0 | Balanced (typical) | **~0.10** |

**Most promising candidates** for high variance:
- α=0.1, β=10 (minimal exploitation, maximal diversity)
- α=0.5, β=10
- α=0.1, β=5.0

---

## Files Created

### Core Implementation
1. **`src/fragile/theory/qsd_variance_sweep.py`** - Comprehensive parameter sweep
   - Multimodal potential functions
   - RunHistory integration
   - Full grid sweep logic
   - 600+ lines

2. **`src/fragile/theory/test_multimodal_variance.py`** - Quick sanity check test
   - Single high-diversity multimodal experiment
   - Validates setup before full sweep

### Previous Work
3. **`src/fragile/theory/qsd_variance.py`** - Original (flawed) experiment
   - Unimodal quadratic potential only
   - Utilities reused in sweep

### Documentation
4. **`QSD_VARIANCE_INVESTIGATION_STATUS.md`** - This document

### Related from Previous Session
- `VARIANCE_REQUIREMENT_ANALYSIS.md` - Mathematical analysis of variance requirement
- `HIERARCHICAL_CLUSTERING_STATUS_SUMMARY.md` - Overall proof status
- `SESSION_SUMMARY.md` - Complete investigation summary

---

## Next Steps

### After Sweep Completes (~9 hours)

1. **Analyze results DataFrame**
   - Identify maximum variance ratio achieved
   - Find best (potential, α, β) combinations
   - Check N-scaling of successful configurations

2. **Create visualizations**
   - Heatmaps: Variance ratio vs (α, β) for each potential
   - Line plots: Variance vs α/β ratio
   - 3D surface: (α, β, variance) for multimodal potential
   - Scaling plots: N_close vs N for different regimes

3. **Generate summary report**
   - Decision on O(N^{3/2}) feasibility
   - Document successful parameter regimes (if any)
   - Recommend proof strategy based on results

4. **Update hierarchical clustering proof documents**
   - If successful: Specify parameter requirements
   - If failed: Mark edge-counting approach as infeasible

---

## Summary

**Original error**: Tested unimodal potential → low variance (expected, but wrong test case)

**User insights**:
1. Multimodal landscapes form clusters → potential for high variance
2. α/β parameters control exploitation-exploration balance

**Corrected approach**: Full 3D grid sweep (potential × α × β × N)

**Initial result**: Multimodal helps (2x improvement) but insufficient (0.152 vs 0.45)

**Current status**: 225-experiment sweep running, will complete in ~9 hours

**Decision pending**: Does ANY configuration achieve Var_h/D²_max ≥ 0.45?

**Impact**: Determines feasibility of hierarchical clustering proof via edge-counting strategy

---

**Investigation Status**: ⏳ **IN PROGRESS** (1/225 experiments complete)
**Next Update**: After full sweep completes
