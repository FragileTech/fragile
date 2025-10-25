# Proof Sketch: Total Boundary Safety from Dual Mechanisms (CORRECTED)

**Label:** `cor-total-boundary-safety`
**Type:** Corollary
**Document:** `05_kinetic_contraction.md`
**Line:** 2533
**Status:** Draft (Corrected after dual review)

---

## Statement (CORRECTED)

Combining the Safe Harbor mechanism from cloning (03_cloning.md, Ch 11) with the confining potential:

**From cloning:**

$$
\mathbb{E}_{\text{clone}}[W_b(S'_1, S'_2) \mid S_1, S_2] \leq (1 - \kappa_b) W_b(S_1, S_2) + C_b
$$

**From kinetics:**

$$
\mathbb{E}_{\text{kin}}[W_b(\tilde{S}_1, \tilde{S}_2) \mid S'_1, S'_2] \leq (1 - \kappa_{\text{pot}}\tau) W_b(S'_1, S'_2) + C_{\text{pot}}\tau
$$

**Combined (multiplicative composition):**

$$
\mathbb{E}_{\text{total}}[\Delta W_b] \leq -(\kappa_b + \kappa_{\text{pot}}\tau - \kappa_b\kappa_{\text{pot}}\tau) W_b + [(1 - \kappa_{\text{pot}}\tau)C_b + C_{\text{pot}}\tau]
$$

**Simplified bound (relaxing constants upward):**

$$
\mathbb{E}_{\text{total}}[\Delta W_b] \leq -(\kappa_b + \kappa_{\text{pot}}\tau - \kappa_b\kappa_{\text{pot}}\tau) W_b + (C_b + C_{\text{pot}}\tau)
$$

**Result:** **Layered defense** - even if one mechanism temporarily fails, the other provides safety. The combined contraction rate includes a small positive cross term $\kappa_b\kappa_{\text{pot}}\tau$ that represents the interaction between mechanisms.

---

## Corrections from Original Sketch

### Critical Errors Fixed:

1. **Operator order corrected**: Changed from $\Psi_{\text{clone}} \circ \Psi_{\text{kin}}$ to $\Psi_{\text{kin}} \circ \Psi_{\text{clone}}$ (cloning then kinetics), matching {prf:ref}`alg-euclidean-gas` (02_euclidean_gas.md § 3.1, lines 164-165).

2. **Invalid inequality substitution removed**: The original Step 6 incorrectly substituted an upper bound for $\mathbb{E}[W_b]$ into a term with negative coefficient $-\kappa_b$, which reverses the inequality direction. The corrected proof composes the bounds on $W_b$ itself (not $\Delta W_b$) using the tower property.

3. **Cross term included**: The correct combined contraction rate is $\kappa_b + \kappa_{\text{pot}}\tau - \kappa_b\kappa_{\text{pot}}\tau$, not the purely additive $\kappa_b + \kappa_{\text{pot}}\tau$. For small $\tau$, the cross term $\kappa_b\kappa_{\text{pot}}\tau = O(\tau)$ is small but mathematically required.

4. **Bias term corrected**: Changed from $C_{\text{pot}}\tau(1 - \kappa_b) + C_b$ to $(1 - \kappa_{\text{pot}}\tau)C_b + C_{\text{pot}}\tau$.

---

## Proof Sketch

### Strategy

This corollary demonstrates **multiplicative composition** of two independent boundary safety mechanisms. The proof strategy is:

1. **Correct operator decomposition**: Use $\Psi_{\text{total}} = \Psi_{\text{kin}} \circ \Psi_{\text{clone}}$ per the canonical pipeline
2. **Compose bounds on $W_b$ itself**: Apply tower property to compose multiplicative drift bounds
3. **Extract total drift**: Convert composed bound on $W_b^{t+1}$ to bound on $\Delta W_b$
4. **Interpret the layered defense**: Show redundancy despite the cross term

### Dependencies

- **{prf:ref}`thm-boundary-potential-contraction`** (03_cloning.md § 11.3, line 7209): Provides $\mathbb{E}_{\text{clone}}[W_b(S') \mid S] \leq (1 - \kappa_b) W_b(S) + C_b$
- **{prf:ref}`thm-boundary-potential-contraction-kinetic`** (05_kinetic_contraction.md § 7.3, line 2733): Provides $\mathbb{E}_{\text{kin}}[\Delta W_b] \leq -\kappa_{\text{pot}} W_b \tau + C_{\text{pot}}\tau$, equivalent to $\mathbb{E}_{\text{kin}}[W_b(\tilde{S}) \mid S'] \leq (1 - \kappa_{\text{pot}}\tau) W_b(S') + C_{\text{pot}}\tau$
- **{prf:ref}`alg-euclidean-gas`** (02_euclidean_gas.md § 3.1): Canonical pipeline is cloning (Stage 3) then kinetics (Stage 4)

### Core Argument

**STEP 1: Operator Decomposition (CORRECTED)**

The full Euclidean Gas operator ({prf:ref}`alg-euclidean-gas`, 02_euclidean_gas.md lines 164-165) is:

$$
\Psi_{\text{total}} = \Psi_{\text{kin}} \circ \Psi_{\text{clone}}
$$

**Stage 3** (Cloning): $(S_1, S_2) \mapsto (S'_1, S'_2)$ via cloning operator

**Stage 4** (Kinetics): $(S'_1, S'_2) \mapsto (\tilde{S}_1, \tilde{S}_2)$ via BAOAB Langevin step

The boundary potential evolves as:

$$
W_b^{t+1} = W_b(\tilde{S}_1, \tilde{S}_2) = W_b(\Psi_{\text{kin}}(\Psi_{\text{clone}}(S_1, S_2)))
$$

**STEP 2: Cloning Bound (Multiplicative Form)**

From {prf:ref}`thm-boundary-potential-contraction` (03_cloning.md line 7215):

$$
\mathbb{E}_{\text{clone}}[W_b(S'_1, S'_2) \mid S_1, S_2] \leq (1 - \kappa_b) W_b(S_1, S_2) + C_b
$$

**Physical mechanism:** Safe Harbor Axiom ensures boundary-proximate walkers have low fitness and are replaced by interior clones.

**STEP 3: Kinetic Bound (Multiplicative Form)**

From {prf:ref}`thm-boundary-potential-contraction-kinetic` (05_kinetic_contraction.md line 2738):

$$
\mathbb{E}_{\text{kin}}[\Delta W_b] \leq -\kappa_{\text{pot}} W_b \tau + C_{\text{pot}}\tau
$$

This is equivalent to the multiplicative form:

$$
\mathbb{E}_{\text{kin}}[W_b(\tilde{S}_1, \tilde{S}_2) \mid S'_1, S'_2] \leq (1 - \kappa_{\text{pot}}\tau) W_b(S'_1, S'_2) + C_{\text{pot}}\tau
$$

**Physical mechanism:** Confining potential creates inward-pointing force that reduces boundary potential.

**STEP 4: Composition via Tower Property**

The full composition gives:

$$
\mathbb{E}[W_b^{t+1} \mid S^t] = \mathbb{E}\left[\mathbb{E}[W_b(\tilde{S}) \mid S'] \mid S\right]
$$

Apply the kinetic bound to the inner expectation:

$$
\mathbb{E}[W_b(\tilde{S}) \mid S'] \leq (1 - \kappa_{\text{pot}}\tau) W_b(S') + C_{\text{pot}}\tau
$$

Take expectation with respect to the cloning operator:

$$
\mathbb{E}[W_b^{t+1} \mid S] \leq \mathbb{E}_{\text{clone}}\left[(1 - \kappa_{\text{pot}}\tau) W_b(S') + C_{\text{pot}}\tau \mid S\right]
$$

By linearity:

$$
= (1 - \kappa_{\text{pot}}\tau) \mathbb{E}_{\text{clone}}[W_b(S') \mid S] + C_{\text{pot}}\tau
$$

Apply the cloning bound:

$$
\leq (1 - \kappa_{\text{pot}}\tau) \left[(1 - \kappa_b) W_b(S) + C_b\right] + C_{\text{pot}}\tau
$$

**STEP 5: Expand the Composition**

Expanding the product:

$$
\mathbb{E}[W_b^{t+1} \mid S] \leq (1 - \kappa_{\text{pot}}\tau)(1 - \kappa_b) W_b(S) + (1 - \kappa_{\text{pot}}\tau) C_b + C_{\text{pot}}\tau
$$

Expand the coefficient of $W_b(S)$:

$$
(1 - \kappa_{\text{pot}}\tau)(1 - \kappa_b) = 1 - \kappa_b - \kappa_{\text{pot}}\tau + \kappa_b\kappa_{\text{pot}}\tau
$$

Therefore:

$$
\mathbb{E}[W_b^{t+1} \mid S] \leq \left[1 - \kappa_b - \kappa_{\text{pot}}\tau + \kappa_b\kappa_{\text{pot}}\tau\right] W_b(S) + (1 - \kappa_{\text{pot}}\tau) C_b + C_{\text{pot}}\tau
$$

**STEP 6: Extract Total Drift**

The drift is:

$$
\mathbb{E}[\Delta W_b \mid S] = \mathbb{E}[W_b^{t+1} - W_b \mid S]
$$

From Step 5:

$$
\mathbb{E}[\Delta W_b \mid S] \leq -\kappa_b W_b - \kappa_{\text{pot}}\tau W_b + \kappa_b\kappa_{\text{pot}}\tau W_b + (1 - \kappa_{\text{pot}}\tau) C_b + C_{\text{pot}}\tau
$$

Factoring:

$$
\boxed{\mathbb{E}[\Delta W_b] \leq -(\kappa_b + \kappa_{\text{pot}}\tau - \kappa_b\kappa_{\text{pot}}\tau) W_b + [(1 - \kappa_{\text{pot}}\tau) C_b + C_{\text{pot}}\tau]}
$$

**This is the exact combined bound.**

**STEP 7: Simplified Bound (Relaxing Constants)**

For practical purposes, we can upper-bound the bias term:

$$
(1 - \kappa_{\text{pot}}\tau) C_b + C_{\text{pot}}\tau \leq C_b + C_{\text{pot}}\tau
$$

since $(1 - \kappa_{\text{pot}}\tau) < 1$ for $\tau > 0$. This gives the simplified bound:

$$
\boxed{\mathbb{E}[\Delta W_b] \leq -(\kappa_b + \kappa_{\text{pot}}\tau - \kappa_b\kappa_{\text{pot}}\tau) W_b + (C_b + C_{\text{pot}}\tau)}
$$

**STEP 8: Interpretation — Layered Defense with Cross Term**

The combined drift inequality reveals:

1. **Combined contraction rate:** $\kappa_{\text{combined}} = \kappa_b + \kappa_{\text{pot}}\tau - \kappa_b\kappa_{\text{pot}}\tau$
   - For small $\tau$: $\kappa_{\text{combined}} \approx \kappa_b + \kappa_{\text{pot}}\tau$ (nearly additive)
   - The cross term $\kappa_b\kappa_{\text{pot}}\tau = O(\tau)$ represents the interaction: when both mechanisms act, the second operates on the already-contracted state

2. **Redundancy property preserved:** Even with the cross term:
   - If $\kappa_b \approx 0$: $\kappa_{\text{combined}} \approx \kappa_{\text{pot}}\tau > 0$ (kinetics alone)
   - If $\kappa_{\text{pot}} \approx 0$: $\kappa_{\text{combined}} \approx \kappa_b > 0$ (cloning alone)
   - **Layered defense still holds**

3. **Complementary timescales:**
   - Cloning: discrete events (every timestep)
   - Kinetics: continuous SDE evolution over time $\tau$
   - Combined effect: persistent boundary protection

4. **Equilibrium bound (corrected):**

At equilibrium, $\mathbb{E}[\Delta W_b] \approx 0$:

$$
(\kappa_b + \kappa_{\text{pot}}\tau - \kappa_b\kappa_{\text{pot}}\tau) W_b^{\text{eq}} \approx (1 - \kappa_{\text{pot}}\tau) C_b + C_{\text{pot}}\tau
$$

Therefore:

$$
W_b^{\text{eq}} \lesssim \frac{(1 - \kappa_{\text{pot}}\tau) C_b + C_{\text{pot}}\tau}{\kappa_b + \kappa_{\text{pot}}\tau - \kappa_b\kappa_{\text{pot}}\tau}
$$

For small $\tau$, the cross term in the denominator is negligible, giving approximately:

$$
W_b^{\text{eq}} \lesssim \frac{C_b + C_{\text{pot}}\tau}{\kappa_b + \kappa_{\text{pot}}\tau}
$$

**Interpretation:** The denominator's sum (minus small cross term) ensures **lower equilibrium boundary potential** than either mechanism alone.

### Physical Interpretation of Cross Term

The cross term $\kappa_b\kappa_{\text{pot}}\tau$ has a clear physical meaning:

- **Positive sign** (reduces total contraction): When cloning has already contracted $W_b$ by factor $(1 - \kappa_b)$, the kinetic operator then acts on this **reduced** $W_b$, so its absolute contribution $\kappa_{\text{pot}}\tau W_b$ is smaller.

- **Second-order effect**: The term is $O(\tau)$ for fixed $\kappa_b$, so it becomes significant only when both $\kappa_b$ and $\kappa_{\text{pot}}\tau$ are of order unity.

- **Still beneficial**: Despite reducing the combined rate slightly, the cross term confirms that **both mechanisms are active** — it's the signature of true multiplicative composition.

---

## Comparison with Original Corollary Statement

The **original corollary statement** in 05_kinetic_contraction.md (line 2533) claims:

$$
\mathbb{E}_{\text{total}}[\Delta W_b] \leq -(\kappa_b + \kappa_{\text{pot}}\tau) W_b + (C_b + C_{\text{pot}}\tau)
$$

This is **slightly stronger** than the rigorous bound (missing the cross term $+\kappa_b\kappa_{\text{pot}}\tau W_b$).

**Options:**

1. **Update the corollary statement** to the exact bound with cross term (most rigorous)
2. **Keep the statement as an approximation** with a note that it holds to leading order in $\tau$ (practical)
3. **Argue the cross term can be absorbed** into the bias constants via a rescaling argument (technical)

**Recommendation:** Update to the exact bound for mathematical rigor, noting that for small $\tau$ the cross term is negligible.

---

## Mathematical Rigor Assessment (Post-Correction)

### Strengths
- **Correct operator composition**: Matches the canonical pipeline
- **Valid inequality manipulations**: Composes bounds on $W_b$, not $\Delta W_b$
- **Proper tower property application**: Correctly handles nested conditioning
- **Exact cross term included**: No artificial simplifications
- **Clear physical interpretation**: Explains the interaction effect

### Remaining Minor Points
1. **$N$-uniformity**: Should verify that all constants are $N$-uniform (inherited from parent theorems)
2. **Measurability**: Should state that $W_b$ is measurable w.r.t. the appropriate $\sigma$-algebras
3. **Reference to algorithm**: Changed citation from "Definition 2.3.1" to {prf:ref}`alg-euclidean-gas`

### Required Context
- Understanding of Markov operator composition
- Tower property for conditional expectations
- Foster-Lyapunov drift conditions
- Multiplicative vs. additive composition of contraction rates

---

## Next Steps for Full Proof

1. **Verify parent theorem forms**:
   - Check that {prf:ref}`thm-boundary-potential-contraction` uses multiplicative form $(1 - \kappa_b)$
   - Check that {prf:ref}`thm-boundary-potential-contraction-kinetic` can be written in multiplicative form

2. **Add explicit measurability statements**:
   - $W_b$ is measurable w.r.t. $\mathcal{F}_t$ (natural filtration)
   - Conditional expectations are well-defined

3. **Verify $N$-uniformity**:
   - $\kappa_b, \kappa_{\text{pot}}, C_b, C_{\text{pot}}$ are all $N$-uniform (from parent theorems)

4. **Update the corollary statement** in 05_kinetic_contraction.md to reflect the exact bound with cross term

5. **Add numerical estimates**:
   - For typical parameter regimes, compute $\kappa_b\kappa_{\text{pot}}\tau$ to show it's small
   - Provide example equilibrium bounds

---

## Dependencies for Full Proof

**Theorems:**
- {prf:ref}`thm-boundary-potential-contraction` (03_cloning.md § 11.3, line 7209)
- {prf:ref}`thm-boundary-potential-contraction-kinetic` (05_kinetic_contraction.md § 7.3, line 2733)

**Definitions:**
- Boundary potential $W_b$ (03_cloning.md Definition 11.2.1)
- Barrier function $\varphi_{\text{barrier}}$ (03_cloning.md § 11.2)
- Kinetic operator $\Psi_{\text{kin}}$ (05_kinetic_contraction.md § 3.2)
- Cloning operator $\Psi_{\text{clone}}$ (03_cloning.md § 9)

**Axioms:**
- Safe Harbor Axiom (Axiom EG-2, 03_cloning.md § 4.3)
- Confining Potential Axiom (Axiom 3.3.1, 05_kinetic_contraction.md § 3.3.1)

**Algorithm:**
- {prf:ref}`alg-euclidean-gas` (02_euclidean_gas.md § 3.1, lines 155-201)

---

## Proof Depth: Thorough (Corrected)

This corrected sketch provides:
- Complete logical structure with correct operator order
- Valid inequality manipulations (no substitution errors)
- Exact combined bound including cross term
- Clear identification of dependencies and algorithm structure
- Physical interpretation of results and cross term
- Comparison with original statement and recommendations

The proof is mathematically sound and ready for rigorous formalization.
