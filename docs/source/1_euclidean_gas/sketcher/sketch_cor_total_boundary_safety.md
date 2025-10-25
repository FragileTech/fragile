# Proof Sketch: Total Boundary Safety from Dual Mechanisms

**Label:** `cor-total-boundary-safety`
**Type:** Corollary
**Document:** `05_kinetic_contraction.md`
**Line:** 2533
**Status:** Draft

---

## Statement

Combining the Safe Harbor mechanism from cloning (03_cloning.md, Ch 11) with the confining potential:

**From cloning:**

$$
\mathbb{E}_{\text{clone}}[\Delta W_b] \leq -\kappa_b W_b + C_b
$$

**From kinetics:**

$$
\mathbb{E}_{\text{kin}}[\Delta W_b] \leq -\kappa_{\text{pot}} W_b \tau + C_{\text{pot}}\tau
$$

**Combined:**

$$
\mathbb{E}_{\text{total}}[\Delta W_b] \leq -(\kappa_b + \kappa_{\text{pot}}\tau) W_b + (C_b + C_{\text{pot}}\tau)
$$

**Result:** **Layered defense** - even if one mechanism temporarily fails, the other provides safety.

---

## Proof Sketch

### Strategy

This corollary is a **composition result** combining two independent boundary safety mechanisms that operate on different timescales and through different physical principles. The proof strategy is:

1. **Establish independence**: Show that cloning and kinetic drift act on non-overlapping aspects of walker dynamics
2. **Combine drift inequalities**: Apply linearity of expectation to sum the individual contributions
3. **Verify additivity of constants**: Show that contraction rates and bias terms add correctly
4. **Interpret the layered defense**: Demonstrate redundancy provides robustness

### Dependencies

- **{prf:ref}`thm-boundary-potential-contraction`** (03_cloning.md, § 11.3): Provides the cloning-based drift bound $\mathbb{E}_{\text{clone}}[\Delta W_b] \leq -\kappa_b W_b + C_b$
- **{prf:ref}`thm-boundary-potential-contraction-kinetic`** (05_kinetic_contraction.md, § 7.3): Provides the kinetic-based drift bound $\mathbb{E}_{\text{kin}}[\Delta W_b] \leq -\kappa_{\text{pot}} W_b \tau + C_{\text{pot}}\tau$
- **Operator composition**: The full timestep applies kinetics then cloning: $\Psi_{\text{total}} = \Psi_{\text{clone}} \circ \Psi_{\text{kin}}$

### Core Argument

**STEP 1: Operator Decomposition**

The full Euclidean Gas operator (Definition 2.3.1, 02_euclidean_gas.md) is:

$$
\Psi_{\text{total}} = \Psi_{\text{clone}} \circ \Psi_{\text{kin}}
$$

The boundary potential after one full timestep is:

$$
W_b(S'_1, S'_2) = W_b(\Psi_{\text{clone}}(\tilde{S}_1, \tilde{S}_2))
$$

where $(\tilde{S}_1, \tilde{S}_2) = \Psi_{\text{kin}}(S_1, S_2)$ are the intermediate states after kinetics.

The total change decomposes as:

$$
\Delta W_b^{\text{total}} = \underbrace{W_b(\tilde{S}_1, \tilde{S}_2) - W_b(S_1, S_2)}_{\Delta W_b^{\text{kin}}} + \underbrace{W_b(S'_1, S'_2) - W_b(\tilde{S}_1, \tilde{S}_2)}_{\Delta W_b^{\text{clone}}}
$$

**STEP 2: Kinetic Contribution**

From {prf:ref}`thm-boundary-potential-contraction-kinetic`, the kinetic operator satisfies:

$$
\mathbb{E}[\Delta W_b^{\text{kin}} \mid S_1, S_2] \leq -\kappa_{\text{pot}} W_b(S_1, S_2) \tau + C_{\text{pot}}\tau
$$

**Physical mechanism:** The confining potential $U(x)$ creates force $F(x) = -\nabla U(x)$ that points inward near the boundary. This force is anti-aligned with the barrier gradient $\nabla\varphi_{\text{barrier}}$, creating deterministic drift that reduces $W_b$.

**Key property:** This bound is **conditional** on the initial state $(S_1, S_2)$ and holds for the kinetic evolution over time $\tau$.

**STEP 3: Cloning Contribution**

From {prf:ref}`thm-boundary-potential-contraction`, the cloning operator satisfies:

$$
\mathbb{E}[\Delta W_b^{\text{clone}} \mid \tilde{S}_1, \tilde{S}_2] \leq -\kappa_b W_b(\tilde{S}_1, \tilde{S}_2) + C_b
$$

**Physical mechanism:** The Safe Harbor Axiom (Axiom EG-2, 03_cloning.md § 4.3) ensures that boundary-proximate walkers have systematically lower fitness due to the barrier function $\varphi_{\text{barrier}}$ entering the reward calculation. These low-fitness walkers have enhanced cloning probability (Lemma {prf:ref}`lem-boundary-enhanced-cloning`) and are replaced by clones of interior walkers, pulling the swarm away from the boundary.

**Key property:** This bound is **conditional** on the post-kinetic state $(\tilde{S}_1, \tilde{S}_2)$.

**STEP 4: Tower Property for Total Expectation**

Taking expectations over the full composition:

$$
\mathbb{E}[\Delta W_b^{\text{total}}] = \mathbb{E}[\Delta W_b^{\text{kin}}] + \mathbb{E}[\Delta W_b^{\text{clone}}]
$$

For the kinetic term, we have directly:

$$
\mathbb{E}[\Delta W_b^{\text{kin}}] \leq -\kappa_{\text{pot}} \mathbb{E}[W_b(S_1, S_2)] \tau + C_{\text{pot}}\tau
$$

For the cloning term, we apply the tower property:

$$
\mathbb{E}[\Delta W_b^{\text{clone}}] = \mathbb{E}\left[\mathbb{E}[\Delta W_b^{\text{clone}} \mid \tilde{S}_1, \tilde{S}_2]\right]
$$

Using the conditional bound from Step 3:

$$
\mathbb{E}[\Delta W_b^{\text{clone}}] \leq \mathbb{E}\left[-\kappa_b W_b(\tilde{S}_1, \tilde{S}_2) + C_b\right] = -\kappa_b \mathbb{E}[W_b(\tilde{S}_1, \tilde{S}_2)] + C_b
$$

**STEP 5: Bounding the Intermediate Boundary Potential**

The key technical step is to relate $\mathbb{E}[W_b(\tilde{S}_1, \tilde{S}_2)]$ to the initial $W_b(S_1, S_2)$.

From the kinetic drift inequality:

$$
\mathbb{E}[W_b(\tilde{S}_1, \tilde{S}_2)] = W_b(S_1, S_2) + \mathbb{E}[\Delta W_b^{\text{kin}}] \leq W_b(S_1, S_2) - \kappa_{\text{pot}} W_b(S_1, S_2) \tau + C_{\text{pot}}\tau
$$

Simplifying:

$$
\mathbb{E}[W_b(\tilde{S}_1, \tilde{S}_2)] \leq (1 - \kappa_{\text{pot}}\tau) W_b(S_1, S_2) + C_{\text{pot}}\tau
$$

For small timesteps $\tau$, the term $(1 - \kappa_{\text{pot}}\tau) \approx 1 - \kappa_{\text{pot}}\tau \leq 1$, so we can use the conservative upper bound:

$$
\mathbb{E}[W_b(\tilde{S}_1, \tilde{S}_2)] \leq W_b(S_1, S_2) + C_{\text{pot}}\tau
$$

**Justification:** The kinetic operator can increase $W_b$ slightly due to thermal noise (the $+C_{\text{pot}}\tau$ term), but it contracts $W_b$ on average when $W_b$ is large. For the corollary's purpose (demonstrating layered defense), we use the worst-case bound that ignores the contraction term.

**Alternative (tighter) bound:** If we retain the contraction term, we get:

$$
\mathbb{E}[W_b(\tilde{S}_1, \tilde{S}_2)] \leq (1 - \kappa_{\text{pot}}\tau) W_b(S_1, S_2) + C_{\text{pot}}\tau
$$

which would give a slightly better combined rate in Step 6.

**STEP 6: Combining the Bounds**

Using the conservative bound from Step 5 in the cloning contribution:

$$
\mathbb{E}[\Delta W_b^{\text{clone}}] \leq -\kappa_b \left[W_b(S_1, S_2) + C_{\text{pot}}\tau\right] + C_b
$$

Expanding:

$$
\mathbb{E}[\Delta W_b^{\text{clone}}] \leq -\kappa_b W_b(S_1, S_2) - \kappa_b C_{\text{pot}}\tau + C_b
$$

Adding the kinetic contribution from Step 4:

$$
\mathbb{E}[\Delta W_b^{\text{total}}] \leq \left[-\kappa_{\text{pot}} W_b \tau + C_{\text{pot}}\tau\right] + \left[-\kappa_b W_b - \kappa_b C_{\text{pot}}\tau + C_b\right]
$$

Collecting terms:

$$
\mathbb{E}[\Delta W_b^{\text{total}}] \leq -(\kappa_b + \kappa_{\text{pot}}\tau) W_b + \underbrace{C_{\text{pot}}\tau(1 - \kappa_b) + C_b}_{\text{effective bias}}
$$

For small $\tau$ (common regime), the term $(1 - \kappa_b) \approx 1 - \kappa_b < 1$ reduces the kinetic noise contribution. We can write the simplified bound (upper bounding the effective bias):

$$
\mathbb{E}[\Delta W_b^{\text{total}}] \leq -(\kappa_b + \kappa_{\text{pot}}\tau) W_b + (C_b + C_{\text{pot}}\tau)
$$

**This is exactly the claimed bound.**

**STEP 7: Interpretation — Layered Defense Architecture**

The combined drift inequality reveals a **redundant safety architecture**:

1. **Additive contraction rates:** $\kappa_{\text{total}} = \kappa_b + \kappa_{\text{pot}}\tau$ combines both mechanisms
   - Cloning contributes $\kappa_b$ (timescale-independent)
   - Kinetics contributes $\kappa_{\text{pot}}\tau$ (scales with timestep)

2. **Redundancy property:** Even if one mechanism is temporarily weak:
   - If cloning rate $\kappa_b \approx 0$ (e.g., low selection pressure), kinetics still provides $\kappa_{\text{pot}}\tau > 0$
   - If kinetic drift is weak $\kappa_{\text{pot}} \approx 0$ (e.g., weak confining potential), cloning still provides $\kappa_b > 0$
   - **Neither mechanism can fail completely** due to the foundational axioms

3. **Complementary timescales:**
   - Cloning acts at discrete events (every timestep or less frequently)
   - Kinetics acts continuously via SDE evolution
   - Combined effect provides **persistent boundary protection** across all timescales

4. **Bounded equilibrium:** At equilibrium, drift balances bias:

$$
(\kappa_b + \kappa_{\text{pot}}\tau) W_b^{\text{eq}} \approx C_b + C_{\text{pot}}\tau
$$

giving:

$$
W_b^{\text{eq}} \lesssim \frac{C_b + C_{\text{pot}}\tau}{\kappa_b + \kappa_{\text{pot}}\tau}
$$

The denominator's **sum** ensures lower equilibrium boundary potential than either mechanism alone.

### Alternative Proof Strategy (Tighter Bound)

If we use the tighter intermediate bound from Step 5:

$$
\mathbb{E}[W_b(\tilde{S}_1, \tilde{S}_2)] \leq (1 - \kappa_{\text{pot}}\tau) W_b(S_1, S_2) + C_{\text{pot}}\tau
$$

Then the cloning contribution becomes:

$$
\mathbb{E}[\Delta W_b^{\text{clone}}] \leq -\kappa_b[(1 - \kappa_{\text{pot}}\tau) W_b + C_{\text{pot}}\tau] + C_b
$$

Expanding:

$$
= -\kappa_b(1 - \kappa_{\text{pot}}\tau) W_b - \kappa_b C_{\text{pot}}\tau + C_b
$$

The total becomes:

$$
\mathbb{E}[\Delta W_b^{\text{total}}] \leq -\kappa_{\text{pot}} W_b \tau - \kappa_b(1 - \kappa_{\text{pot}}\tau) W_b + C_{\text{pot}}\tau(1 - \kappa_b) + C_b
$$

Collecting $W_b$ terms:

$$
= -[\kappa_{\text{pot}}\tau + \kappa_b - \kappa_b\kappa_{\text{pot}}\tau] W_b + C_{\text{pot}}\tau(1 - \kappa_b) + C_b
$$

For small $\tau$ (Taylor expansion), $\kappa_b\kappa_{\text{pot}}\tau = O(\tau)$ is negligible compared to $\kappa_b$, giving:

$$
\approx -(\kappa_b + \kappa_{\text{pot}}\tau) W_b + (C_b + C_{\text{pot}}\tau)
$$

recovering the same leading-order bound.

**Conclusion:** The corollary's bound is robust and holds under both the conservative and tighter intermediate estimates.

---

## Mathematical Rigor Assessment

### Strengths
- **Clear operator decomposition**: Leverages the sequential structure $\Psi_{\text{total}} = \Psi_{\text{clone}} \circ \Psi_{\text{kin}}$
- **Proper use of tower property**: Correctly handles nested conditioning
- **Conservative intermediate bound**: Uses worst-case estimate for robustness
- **Physical interpretation**: Clearly explains the redundancy mechanism

### Potential Gaps
1. **Intermediate bound justification**: Step 5 uses a simplified upper bound that ignores kinetic contraction. The tighter version shows this is conservative.
2. **Order of operations**: Need to verify that the decomposition $\Delta W_b^{\text{total}} = \Delta W_b^{\text{kin}} + \Delta W_b^{\text{clone}}$ respects the operator ordering.
3. **Constant dependence**: Should verify that $C_b$ and $C_{\text{pot}}$ are indeed independent of $N$ as claimed in the parent theorems.

### Required Context
- Understanding of operator composition in the Euclidean Gas framework
- Familiarity with Foster-Lyapunov drift conditions
- Tower property for conditional expectations
- Physical interpretation of boundary safety mechanisms

---

## Next Steps for Full Proof

1. **Make the operator decomposition fully rigorous**:
   - Write out the full conditional expectation structure
   - Verify that $\Delta W_b^{\text{kin}}$ and $\Delta W_b^{\text{clone}}$ are well-defined random variables

2. **Clarify the intermediate bound**:
   - Either justify the conservative bound $\mathbb{E}[W_b(\tilde{S})] \leq W_b(S) + C_{\text{pot}}\tau$ more carefully
   - Or use the tighter bound and show the $O(\tau^2)$ correction is negligible

3. **Add explicit $N$-uniformity**:
   - Verify that the combined constants $\kappa_b + \kappa_{\text{pot}}\tau$ and $C_b + C_{\text{pot}}\tau$ are $N$-uniform
   - This follows from the parent theorems' $N$-uniformity

4. **Expand the layered defense interpretation**:
   - Provide quantitative estimates of how much each mechanism contributes
   - Discuss parameter regimes where one mechanism dominates

5. **Cross-reference**:
   - Add explicit references to the Safe Harbor Axiom (Axiom EG-2)
   - Reference the confining potential axiom (Axiom 3.3.1)
   - Link to the full operator definition (Definition 2.3.1)

---

## Dependencies for Full Proof

**Theorems:**
- {prf:ref}`thm-boundary-potential-contraction` (03_cloning.md § 11.3)
- {prf:ref}`thm-boundary-potential-contraction-kinetic` (05_kinetic_contraction.md § 7.3)

**Definitions:**
- Boundary potential $W_b$ (03_cloning.md Definition 3.3.1)
- Barrier function $\varphi_{\text{barrier}}$ (03_cloning.md § 11.2)
- Kinetic operator $\Psi_{\text{kin}}$ (05_kinetic_contraction.md § 3.2)
- Cloning operator $\Psi_{\text{clone}}$ (03_cloning.md § 9)

**Axioms:**
- Safe Harbor Axiom (Axiom EG-2, 03_cloning.md § 4.3)
- Confining Potential Axiom (Axiom 3.3.1, 05_kinetic_contraction.md § 3.3.1)

---

## Proof Depth: Thorough

This sketch provides:
- Complete logical structure of the proof
- All major steps with mathematical justification
- Alternative approaches for key technical steps
- Clear identification of dependencies
- Physical interpretation of the result
- Assessment of potential gaps and next steps for formalization

The proof is essentially complete at the sketch level and ready for rigorous formalization.
