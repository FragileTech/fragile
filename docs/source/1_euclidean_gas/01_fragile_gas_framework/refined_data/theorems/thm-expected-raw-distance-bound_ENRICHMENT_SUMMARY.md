# Enrichment Summary: thm-expected-raw-distance-bound

## Original Raw Data
- **Source**: `docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data/theorems/thm-expected-raw-distance-bound.json`
- **Section**: 10.2.7 "Bound on the Expected Raw Distance Vector Change"
- **Document Lines**: 2583-2606

## Enrichments Applied

### 1. Semantic Classification
- **output_type**: Classified as **"Bound"** (establishes upper bound on expected raw distance vector change)
- **Rationale**: Theorem provides deterministic bound: ||E[d(S1)] - E[d(S2)]||² ≤ f(Δ_pos², n_c)

### 2. Input Objects Identified (3)
- `obj-alg-distance` - Algorithmic distance function d_alg
- `obj-expected-distance-error-coefficients` - Coefficients C_pos,d, C_status,d⁽¹⁾, C_status,d⁽²⁾(k1)
- `obj-distance-measurement-ms-constants` - Mean-square continuity constants for distance measurement

### 3. Input Axioms Identified (1)
- `axiom-bounded-measurement-variance` - Ensures bounded variance in measurement operators

### 4. Input Parameters Identified (2)
- `param-n` - Swarm size (N walkers)
- `param-kappa-variance` - Maximum measurement variance parameter

### 5. Internal Lemmas (2)
- `lem-total-squared-error-stable` - Bounds error from stable walkers (those alive in both states)
- `lem-total-squared-error-unstable` - Bounds error from unstable walkers (status changes)

### 6. Dependency Graph
```
thm-total-expected-distance-error-decomposition
    ├─→ lem-total-squared-error-stable
    └─→ lem-total-squared-error-unstable
         └─→ thm-expected-raw-distance-bound (this theorem)
```

### 7. Prerequisites Definitions (4)
- `def-displacement-components` - Defines Δ_pos² and n_c
- `def-expected-distance-error-coefficients` - Defines C_pos,d, C_status,d⁽¹⁾, C_status,d⁽²⁾(k1)
- `def-swarm-state` - Defines swarm state S = (x, v, s)
- `def-alive-set` - Defines alive walker set A(S)

### 8. Natural Language Statement
Comprehensive prose statement capturing:
- **Assumptions**: Two swarm states S1, S2 with k1 ≥ 2 alive walkers in S1
- **Conclusion**: Squared L2 distance between expected raw distance vectors is bounded
- **Bound structure**: Linear in Δ_pos², linear and quadratic in n_c
- **Coefficients**: Explicit values and dependencies on D_Y and k1

**Full statement**: "Let S1 and S2 be two swarm states, with at least two alive walkers in S1 (k1 ≥ 2). The squared Euclidean distance between the expected raw distance vectors of the two swarms is deterministically bounded by a function of the displacement components (positional displacement Δ_pos² and status change count n_c), with coefficients that depend on the algorithmic space diameter D_Y and the initial number of alive walkers k1. Specifically: ||E[d(S1)] - E[d(S2)]||² ≤ C_pos,d·Δ_pos² + C_status,d⁽¹⁾·n_c + C_status,d⁽²⁾(k1)·n_c² where the coefficients are C_pos,d = 12, C_status,d⁽¹⁾ = D_Y², and C_status,d⁽²⁾(k1) = 8k1 D_Y²/(k1-1)²."

## Mathematical Significance

### Role in Framework
This theorem consolidates the error bounds from decomposition and lemma-level bounds to establish the **deterministic part** of the continuity axiom for the distance operator.

### Key Features
1. **Explicit coefficients**: C_pos,d = 12 (constant), C_status,d⁽¹⁾ = D_Y² (diameter-dependent), C_status,d⁽²⁾(k1) = 8k1 D_Y²/(k1-1)² (singular as k1→1)
2. **Separates contributions**: Positional changes (Δ_pos²) vs. status changes (n_c)
3. **N-uniform bound**: Does not grow with swarm size N
4. **Singularity at k1=1**: Coefficient C_status,d⁽²⁾(k1) → ∞ as k1→1, reflecting fundamental discontinuity

### Downstream Usage
Used in:
- `thm-distance-operator-mean-square-continuity` - Proves distance operator satisfies mean-square continuity axiom

## Validation Status
- ✓ Pydantic schema validation: **PASSED**
- ✓ All required fields present
- ✓ Label pattern validation: `thm-expected-raw-distance-bound` matches `^(thm|lem|prop)-[a-z0-9-]+$`
- ✓ Output type: Valid TheoremOutputType enum value ("Bound")
- ✓ Input object prefixes validated: obj-* pattern
- ✓ Input axiom prefixes validated: axiom-* pattern
- ✓ Input parameter prefixes validated: param-* pattern

## Enrichment Method
- **Stage**: Stage 2 (Semantic Enrichment)
- **Model used**: Gemini 2.5 Pro
- **Method**: Manual refinement with LLM assistance
- **Validation**: Pydantic TheoremBox schema
- **Date**: 2025-10-28
