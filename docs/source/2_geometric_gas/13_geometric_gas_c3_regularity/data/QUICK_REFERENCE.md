# C³ Regularity Quick Reference

## Main Result

**Theorem** (`thm-c3-regularity`):
```
‖∇³_{x_i} V_fit[f_k, ρ](x_i)‖ ≤ K_{V,3}(ρ) < ∞
```
- **k-uniform**: Independent of alive walker count
- **N-uniform**: Independent of total swarm size
- **Scaling**: K_{V,3}(ρ) = O(ρ⁻³) as ρ → 0

## Critical Path (12 Steps)

```
Foundation (4)         Pipeline (7)                  Result (1)
┌─────────────┐       ┌──────────────────────┐      ┌──────────┐
│ Assumptions │  →    │ Sequential Lemmas    │  →   │ Main Thm │
└─────────────┘       └──────────────────────┘      └──────────┘
  • Measurement         1. Telescoping identity        K_{V,3}(ρ)
  • Kernel              2. Weights: C_{w,3}(ρ)         bound
  • Rescale             3. Mean: ∇³μ_ρ                 established
  • Regularization      4. Variance: ∇³V_ρ
                        5. Chain rule formula
                        6. Reg std dev: ∇³σ'_ρ
                        7. Z-score: K_{Z,3}(ρ)
```

## Key Bounds (Sequential Dependencies)

| Stage | Bound | Scaling | Depends On |
|-------|-------|---------|------------|
| **Weights** | C_{w,3}(ρ) | O(ρ⁻³) | Kernel bounds |
| **Mean** | C_{μ,∇³}(ρ) | O(ρ⁻²) | Weights + Measurement |
| **Variance** | C_{V,∇³}(ρ) | O(ρ⁻²) | Weights + Mean |
| **Reg Std** | C_{σ',∇³}(ρ) | O(ρ⁻²) | Variance + Chain rule |
| **Z-score** | K_{Z,3}(ρ) | O(ρ⁻³) | Mean + Reg Std |
| **Fitness** | K_{V,3}(ρ) | O(ρ⁻³) | Z-score + Rescale |

## Essential Techniques

### 1. Telescoping Identity
```
∑_{j ∈ A_k} ∇^m w_{ij} = 0    (for m ∈ {1,2,3})
```
**Why it matters**: Enables k-uniform bounds by canceling naive k-dependence

**How to use**: Rewrite sums as centered differences
```
∑_j f(x_j)∇³w_ij = ∑_j [f(x_j) - f(x_i)]∇³w_ij
```

### 2. Faà di Bruno Formula
For h = g ∘ f:
```
∇³h = g'''(f)·(∇f)³ + 3g''(f)·∇f·∇²f + g'(f)·∇³f
```
**Used in**: `lem-patch-chain-rule`, `thm-c3-regularity`

### 3. Quotient Rule (Third Derivative)
For h = u/v with v > 0:
```
∇³(u/v) has 5 terms involving up to ∇³u, ∇³v, and powers of 1/v
```
**Critical**: Need v ≥ v_min > 0 to bound terms with v⁻² and v⁻³

**Used in**: `lem-weight-third-derivative`, `lem-zscore-third-derivative`

### 4. Leibniz Rule
For h = u·v:
```
∇³(uv) = ∑_{|α|=3} (3 choose α) (∇^α u)(∇^{3-α} v)
```
**Used in**: Mean and variance proofs

## Time Step Constraint

```
Δt ≤ min(1/(2γ), ρ^{3/2}/√K_{V,3}(ρ))
    = min(1/(2γ), ρ^{3/2}/√(Cρ⁻³))
    ≈ min(1/(2γ), √C · ρ³)
```

**Practical values** (assume C ≈ 1, γ = 1):
- ρ = 1.0   → Δt ≤ 0.5
- ρ = 0.5   → Δt ≤ 0.06
- ρ = 0.1   → Δt ≤ 0.001

**Rule of thumb**: Δt ~ ρ³ for hyper-local regime

## Label Index

### By Type
**Assumptions** (4):
- `assump-c3-measurement` - d(x) ∈ C³
- `assump-c3-kernel` - K_ρ ∈ C³
- `assump-c3-rescale` - g_A ∈ C³
- `assump-c3-patch` - σ'_reg ∈ C^∞

**Lemmas** (7):
- `lem-telescoping-derivatives` - ∑_j ∇^m w_ij = 0
- `lem-weight-third-derivative` - C_{w,3}(ρ)
- `lem-mean-third-derivative` - ∇³μ_ρ bound
- `lem-variance-third-derivative` - ∇³V_ρ bound
- `lem-patch-chain-rule` - Faà di Bruno
- `lem-patch-third-derivative` - ∇³σ'_ρ bound
- `lem-zscore-third-derivative` - K_{Z,3}(ρ)

**Theorems** (2):
- `thm-c3-regularity` - **MAIN RESULT**
- `thm-continuity-third-derivatives` - Joint continuity

**Propositions** (4):
- `prop-scaling-kv3` - ρ-scaling analysis
- `prop-scaling-k-v-3` - Regime-dependent scaling
- `prop-timestep-constraint` - Δt bound
- `prop-explicit-k-v-3` - Explicit formula

**Corollaries** (4):
- `cor-baoab-validity` - BAOAB discretization
- `cor-lyapunov-c3` - V_total ∈ C³
- `cor-smooth-perturbation` - F_adapt bounds
- `cor-regularity-hierarchy` - Completeness

### By Topic
**Foundations**: assump-*
**Techniques**: lem-telescoping-derivatives, lem-patch-chain-rule
**Pipeline**: lem-weight-third-derivative → lem-mean-third-derivative → lem-variance-third-derivative → lem-patch-third-derivative → lem-zscore-third-derivative
**Result**: thm-c3-regularity
**Applications**: cor-*, prop-*

## External Dependencies

### Required from `11_geometric_gas.md` Appendix A
- `thm-c1-review` - K_{Z,1}(ρ) bound
- `thm-c2-review` - K_{Z,2}(ρ) bound
- Lemmas A.2-A.5 - C¹, C² bounds on μ_ρ, V_ρ

### Required from `06_convergence.md`
- Theorem 1.7.2 - BAOAB discretization theorem

### Forward Reference (⚠️ CRITICAL)
- `14_geometric_gas_cinf_regularity_full.md` Lemma 5.1
- Provides corrected scaling C_{μ,∇^m}(ρ) = O(ρ^{-(m-1)})
- **Needed for**: `prop-scaling-kv3` proof

## Common Pitfalls

❌ **Don't**: Use naive quotient rule bounds for Z-score
✅ **Do**: Exploit σ'_min > 0 lower bound from regularization

❌ **Don't**: Expect k-independent bounds from naive sums ∑_j ∇³w_ij
✅ **Do**: Apply telescoping identity first

❌ **Don't**: Assume K_{V,3}(ρ) = O(ρ⁻³) from first principles
✅ **Do**: Use corrected moment scaling from C∞ document

❌ **Don't**: Ignore ρ-dependence in time step selection
✅ **Do**: Scale Δt ~ ρ³ for stability

## Validation Status

- ✅ Extraction: 22 directives found
- ✅ Labels: All conform to pipeline convention
- ✅ Cross-refs: All internal references valid
- ⚠️ External refs: Require verification in source documents
- ⚠️ Forward ref: Needs explicit citation
- ✅ Math expressions: 161 total, all well-formed

## Quick Lookup

**Need the main bound?** → `thm-c3-regularity`

**Need k-uniformity technique?** → `lem-telescoping-derivatives`

**Need time step guidance?** → `prop-timestep-constraint`

**Need explicit formula?** → `prop-explicit-k-v-3`

**Need BAOAB validity?** → `cor-baoab-validity`

**Need scaling analysis?** → `prop-scaling-kv3`

**Need regularization details?** → `assump-c3-patch`

---

**Last Updated**: 2025-10-26
**Document Version**: v1.0 (validated)
