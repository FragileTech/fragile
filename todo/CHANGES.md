# Pseudoscalar Fix + Chirality Implementation

## Summary of Changes

### 1. Pseudoscalar Operator Fix (3 files, 3 locations)

**Problem:** The pseudoscalar operator was using `.real` which gives a parity-EVEN projection.
Under parity, the color bilinear c_i† c_j → (c_i† c_j)*, so Re[z] → Re[z] (even) and
Im[z] → -Im[z] (odd). The pseudoscalar (0⁻⁺) requires parity-odd → must use `.imag`.

The γ₅ = diag(1,-1,1) matrix was not contributing to the parity quantum number — it just
reweighted spatial components. Removed.

**Files changed:**

#### a) `src/fragile/physics/new_channels/correlator_channels.py` (line ~1125)
```python
# BEFORE (parity-even, WRONG):
gamma5_diag = self.gamma["5"].to(color_i.device)
return (color_i.conj() * gamma5_diag * color_j).sum(dim=-1).real

# AFTER (parity-odd, CORRECT):
return (color_i.conj() * color_j).sum(dim=-1).imag
```

#### b) `src/fragile/fractalai/qft/aggregation.py` — `compute_pseudoscalar_operators()` (line ~723)
```python
# BEFORE:
gamma5_diag = gamma_matrices["5"].to(color_i.device)
op_values = (color_i.conj() * gamma5_diag * color_j).sum(dim=-1).real

# AFTER:
op_values = (color_i.conj() * color_j).sum(dim=-1).imag
```

#### c) `src/fragile/fractalai/qft/aggregation.py` — `compute_pseudoscalar_operators_per_walker()` (line ~1135)
```python
# BEFORE:
gamma5_diag = gamma_matrices["5"]
def pseudoscalar_projection(color_i, color_j):
    return (color_i.conj() * gamma5_diag.to(color_i.device) * color_j).sum(dim=-1).real

# AFTER:
def pseudoscalar_projection(color_i, color_j):
    return (color_i.conj() * color_j).sum(dim=-1).imag
```

### 2. Chirality Module (new file)

**File:** `src/fragile/physics/electroweak/chirality.py`

**Purpose:** Electroweak fermion mass extraction via chirality autocorrelation.

**Walker classification (4-way):**
- Delta (D): `will_clone[i] = True` — phases destroyed by teleportation
- Strong resister (SR): target of a delta — neighborhood disrupted
- Weak resister (WR): not targeted, has fitter peer — phases untouched
- Persister (P): not targeted, no fitter peer — phases untouched

**Chirality label:** χ_i = +1 for L (D+SR), -1 for R (WR+P)

**Key functions:**

- `classify_walkers_vectorized()` — fully vectorized 4-way classification
- `compute_chirality_autocorrelation()` — computes C_χ(τ) and extracts fermion mass
- `compute_lr_coupling()` — measures L↔R phase transfer during cloning (Dirac mass term)

**Usage:**
```python
from fragile.physics.electroweak.chirality import compute_chirality_autocorrelation

result = compute_chirality_autocorrelation(history, max_lag=80)
print(f"Fermion mass: {result.fermion_mass:.4f} ± {result.fermion_mass_err:.4f}")
print(f"L-R transition rate: {result.role_transition_rate:.4f}")
print(f"Mean left fraction: {result.left_fraction.mean():.4f}")
```

## Operator Quantum Number Summary (after fix)

| Channel | Operator | Parity | J^P |
|---|---|---|---|
| Scalar | Re[c_i† c_j] | even | 0⁺ |
| **Pseudoscalar** | **Im[c_i† c_j]** | **odd** | **0⁻** |
| Vector | (check Re/Im) | (verify) | 1⁻ |
| Axial vector | (check Re/Im) | (verify) | 1⁺ |

**Note:** Vector and axial vector operators may also need Re/Im review.
Currently they use ad hoc γ_μ matrices with imaginary off-diagonal entries
that may accidentally produce the right parity. Since their mass ratios are
currently correct, they are lower priority, but should be verified.

### 3. Dirac Spinor Module (new file)

**File:** `src/fragile/physics/new_channels/dirac_spinors.py`

**Purpose:** Proper Dirac spinor representation for strong-force meson operators.
Provides guaranteed quantum numbers via the Clifford algebra, serving as
an independent cross-check of the direct Re/Im operators.

**Key insight — parity matching:**
```
Under parity: c^α → -(c^α)*
  Re(c^α) → -Re(c^α)    parity-ODD   →  ψ_R (lower 2 components)
  Im(c^α) → +Im(c^α)    parity-EVEN  →  ψ_L (upper 2 components)
```

The map ℝ³ → ℂ² uses the Hopf fibration section (spinor harmonics),
with chart switching to avoid the south pole singularity.

**Standard 4×4 Dirac gamma matrices:**
- {γ_μ, γ_ν} = 2g_μν (Clifford algebra)
- γ₅ anticommutes with all γ_μ
- Parity: ψ → γ₀ψ (automatic for all bilinears)

**Key functions:**

- `build_dirac_gamma_matrices()` — standard representation γ₀, γ_k, γ₅, σ_μν
- `verify_clifford_algebra()` — sanity check {γ_μ,γ_ν} = 2g_μν
- `vector_to_weyl_spinor()` — Hopf map ℝ³ → ℂ² with chart switching
- `color_to_dirac_spinor()` — Im(c) → ψ_L, Re(c) → ψ_R
- `compute_dirac_bilinear()` — ψ̄_i Γ ψ_j via einsum (handles single/multi Γ)
- `compute_dirac_operator_series()` — all channels from pre-aggregated data
- `compute_dirac_operators_from_agg()` — convenience wrapper for AggregatedTimeSeries

**Usage:**
```python
from fragile.physics.new_channels.dirac_spinors import compute_dirac_operators_from_agg

# After aggregate_time_series():
dirac_ops = compute_dirac_operators_from_agg(agg_data)

# Compare with existing operators:
print(f"Scalar mass should match Re[c†c] channel")
print(f"Pseudoscalar mass should match Im[c†c] channel")
print(f"Vector mass may differ from current ad hoc γ_μ channel")
```

**Verification strategy:**
1. Run both old and new operators on same data
2. Scalar masses must agree (both are correct)
3. Pseudoscalar masses must agree (both now use parity-odd projection)
4. If vector masses disagree → spinor version is correct, old was wrong
5. Once verified, can use spinor operators for all new channels
