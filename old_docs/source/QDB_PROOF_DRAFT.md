# Quantum Detailed Balance Proof (DRAFT)

**Status**: 🚧 **WORK IN PROGRESS** - Draft for Gemini Review
**Date**: 2025-10-14
**Goal**: Prove Γ_death/Γ_birth = exp(β(H_eff - μ)) in mean-field limit

---

## Extracted Formulas from Framework

### Birth Rate (From Cloning Mechanism)

From `03_cloning.md` Definition 5.7.2 (lines 1958-1974):

**Cloning score**:
```
S_i(c) = (V_fit,c - V_fit,i) / (V_fit,i + ε_clone)
```

**Total cloning probability**:
```
p_i = E_{c ~ C_i(S)} [min(1, max(0, S_i(c)/p_max))]
```

where C_i(S) is the companion distribution.

**Birth rate** (particle creation per unit time):
```
Γ_birth(i; S) = p_i / τ
```

where τ is the time step.

### Fitness Formula

From `01_fragile_gas_framework.md` Definition 11.2.1 (lines 4140-4171):

```
V_fit,i = (g_A(z_d,i) + η)^β · (g_A(z_r,i) + η)^α
```

where:
- `z_r,i = (r_i - μ_r(S)) / (σ_r(S) + ε_std)` - reward Z-score
- `z_d,i = (d_i - μ_d(S)) / (σ_d(S) + ε_std)` - diversity Z-score
- `g_A(z)` - smooth rescale function, bounded in (0, g_A,max]
- `η > 0` - rescale floor
- `α, β > 0` - exploitation/exploration weights

**Bounds** (from Lemma 12.2.1, lines 4201-4231):
```
V_pot,min = η^(α+β) ≤ V_fit,i ≤ (g_A,max + η)^(α+β) = V_pot,max
```

---

## Strategy: Mean-Field Limit

**Key Insight**: In the mean-field limit N→∞, the Z-scores become deterministic functionals of the density ρ.

### Step 1: Mean-Field Fitness

As N→∞, the empirical measure converges:
```
μ_N(S) := (1/N) Σ_i δ_{(x_i, v_i)} → ρ(x,v)  (in probability)
```

The mean and variance become functionals:
```
μ_r(S) → μ_r[ρ] = ∫ r(x,v) ρ(x,v) dx dv
σ_r(S)² → σ_r[ρ]² = ∫ (r(x,v) - μ_r[ρ])² ρ(x,v) dx dv
```

Similarly for diversity. Therefore, in the mean-field limit:
```
z_r,i(S) → z_r(x,v; ρ) = (r(x,v) - μ_r[ρ]) / (σ_r[ρ] + ε_std)
```

And fitness becomes:
```
V_fit(x,v; ρ) = (g_A(z_d(x,v; ρ)) + η)^β · (g_A(z_r(x,v; ρ)) + η)^α
```

This is now a **smooth functional of density ρ**, not a random quantity.

### Step 2: Death Rate from Companion Selection

**Observation**: A walker "dies" when it is selected as a companion by someone with higher fitness and gets replaced.

From the cloning mechanism:
- Walker j selects companion c with probability P_comp(c|j; S)
- If V_fit,c < V_fit,j, then S_j(c) > 0 → j may clone
- When j clones, it replaces itself with a copy of c

**Death probability**: Walker i is replaced when it is selected as companion by a walker j with HIGHER fitness:
```
p_death,i ≈ Σ_{j: V_fit,j > V_fit,i} P_comp(i|j; S) · p_j
```

In mean-field limit, this becomes an integral over the density.

**Death rate**:
```
Γ_death(x,v; ρ) = ∫_{V_fit(y,w;ρ) > V_fit(x,v;ρ)} P_comp((x,v)|(y,w); ρ) · p_clone(y,w; ρ) ρ(y,w) dy dw / τ
```

---

## The Problem: Companion Distribution

**CRITICAL ISSUE**: The companion distribution P_comp(c|i; S) is NOT uniform. From framework, it depends on algorithmic distance:

```
P_comp(k|i) ∝ 1 / d_alg(i,k)^(2+ν)
```

where d_alg is the algorithmic distance in state space.

This means birth and death rates have DIFFERENT spatial structure - they don't trivially cancel!

---

## Approach 1: Symmetry Argument (ATTEMPT)

**Hypothesis**: In the mean-field limit at QSD, the system has sufficient symmetry that the ratio simplifies.

**If QSD is stationary**:
```
∂ρ_QSD/∂t = 0
```

This means:
```
∫ [Γ_birth(x,v; ρ_QSD) - Γ_death(x,v; ρ_QSD)] ρ_QSD(x,v) dx dv = 0
```

**Detailed balance** is stronger: requires point-wise balance:
```
Γ_birth(x,v; ρ_QSD) · ρ_QSD(x,v) = Γ_death(x,v; ρ_QSD) · ρ_QSD(x,v)
```

which gives:
```
Γ_death/Γ_birth = 1  (at QSD)
```

But we need:
```
Γ_death/Γ_birth = exp(β(H_eff - μ))
```

These are consistent only if:
```
H_eff(x,v; ρ_QSD) = μ  (constant at QSD)
```

**This seems wrong!** H_eff varies with (x,v).

---

## Approach 2: Gibbs Ansatz (ATTEMPT)

**Assume** QSD has Gibbs form (from Stratonovich proof):
```
ρ_QSD(x,v) = (1/Z) √(det g(x)) exp(-β H_eff(x,v; ρ_QSD))
```

where H_eff = U(x) - ε_F·V_fit(x,v; ρ_QSD) + (1/2)m‖v‖²

**Self-consistency**: This is a fixed-point equation because V_fit depends on ρ_QSD through Z-scores.

**Question**: Does the cloning mechanism *produce* this distribution, or is it just consistent with it?

**Distinction**:
- **Forward direction**: Cloning rates → QSD distribution (what we need to prove)
- **Backward direction**: QSD distribution → implies certain cloning rates (easier, but not sufficient)

---

## Approach 3: Logarithmic Relationship (KEY INSIGHT)

**Observation**: Take logarithm of fitness:
```
log V_fit,i = β log(g_A(z_d,i) + η) + α log(g_A(z_r,i) + η)
```

**In mean-field limit with smooth g_A**:
If we Taylor expand around mean values and keep leading order:
```
log V_fit(x,v; ρ) ≈ β log(g_A(0) + η) + α log(g_A(0) + η)
                     + β (∂ log / ∂z_d) · z_d(x,v; ρ)
                     + α (∂ log / ∂z_r) · z_r(x,v; ρ)
```

The Z-scores are:
```
z_r(x,v; ρ) ∝ (r(x,v) - μ_r[ρ])
```

**Key question**: Can we show that this linear-in-Z-scores structure produces an exponential relationship with H_eff?

**Potential connection**:
If reward r(x,v) ∝ -H_eff(x,v), then:
```
z_r ∝ -H_eff(x,v) + const
```

And:
```
log V_fit ∝ α·(-H_eff) + ...
```

Which gives:
```
V_fit ∝ exp(-α·H_eff)
```

But cloning score is:
```
S_i(c) = (V_c - V_i)/(V_i + ε)
```

If V ∝ exp(-αH), then:
```
S_i(c) ≈ (exp(-αH_c) - exp(-αH_i))/exp(-αH_i)
       = exp(-α(H_c - H_i)) - 1
       ≈ -α(H_c - H_i)  (for small α(H_c - H_i))
```

Hmm, this is linear in energy difference, not exponential...

---

## Status: STUCK

**What we need**: Show that companion-based cloning with power-law fitness produces Gibbs distribution.

**What we have**:
- ✓ Fitness formula (explicit)
- ✓ Cloning probability (explicit)
- ✓ Stratonovich → Gibbs form (proven separately)
- ✗ Connection between cloning rates and Boltzmann factor (MISSING)

**The gap**: The cloning mechanism operates through:
1. Fitness differences (not ratios)
2. Companion selection (non-uniform)
3. Stochastic thresholding (clip to [0, p_max])

None of these obviously produce exp(β(H-μ)) form!

**Possible resolutions**:
1. **Collective effect**: Maybe individual rates don't satisfy QDB, but the *collective* dynamics still produces Gibbs? (This would be "global balance" not "detailed balance")
2. **Hidden symmetry**: Maybe there's a transformation that makes QDB manifest?
3. **Approximate QDB**: Maybe QDB holds to leading order in 1/N with O(1/N) corrections?
4. **Wrong approach**: Maybe QDB is NOT the right condition - maybe LSI + free energy minimization is the correct route?

---

## Questions for Gemini

1. **Is my extraction of formulas correct?** (Check lines 1958-1974 of 03_cloning.md and lines 4140-4171 of 01_fragile_gas_framework.md)

2. **Is the mean-field limit formulation correct?** (Z-scores become functionals of density)

3. **Is there an error in my reasoning about death rates?** (Walker replaced when selected as companion by higher-fitness walker)

4. **Does the companion distribution P_comp break detailed balance?** (Non-uniform weighting by algorithmic distance)

5. **Is there a known result connecting fitness-based cloning to Gibbs states?** (Literature reference?)

6. **Should I try the LSI + free energy route instead?** (Alternative approach in §20.12.3)

7. **Is "global balance" (not detailed balance) sufficient for KMS condition?** (Weaker requirement?)

8. **Am I missing something obvious?** (Critical insight I've overlooked?)

---

## Next Steps

**Option A**: Ask Gemini to verify my formulas and reasoning, identify errors

**Option B**: Pivot to LSI + free energy minimization approach (bypass QDB entirely)

**Option C**: Numerical validation - simulate and measure Γ_death/Γ_birth empirically

**Recommendation**: Do Option A first (verify with Gemini), then Option C (numerical check), then Option B if needed (alternative proof strategy).
