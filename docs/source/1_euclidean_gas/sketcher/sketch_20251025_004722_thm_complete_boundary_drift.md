# Proof Sketch: Complete Boundary Potential Drift Characterization

**Theorem**: {prf:ref}`thm-complete-boundary-drift`

**Generated**: 2025-10-25 00:47:22

**Status**: Sketch (Stage 1/3)

**Document**: docs/source/1_euclidean_gas/03_cloning.md (lines 7640-7739)

---

## Executive Summary

This sketch outlines a proof strategy for establishing that the cloning operator induces exponential drift toward safety in the boundary potential functional $W_b = \sum_{i \in \mathcal{A}} U(x_i)/N$. The theorem asserts:

$$
\mathbb{E}_{\text{clone}}[\Delta W_b] \leq -\kappa_b W_b + C_b
$$

with **N-independent** contraction constant $\kappa_b > 0$ and bounded noise term $C_b = O(\sigma_x^2 + N^{-1})$.

The proof exploits the Safe Harbor mechanism: walkers near the boundary have low fitness (penalized by the barrier function $\varphi_{\text{barrier}}$), leading to high cloning probability and replacement by safer interior walkers. The challenge is showing this mechanism provides **uniform contraction** across all swarm sizes and configurations.

---

## High-Level Strategy

**Proof Technique**: Direct computation via Foster-Lyapunov drift analysis with conditional expectation decomposition.

**Core Insight**: The boundary potential can be decomposed into contributions from "boundary-exposed" walkers (those with $\varphi_{\text{barrier}}(x_i) > \phi_{\text{thresh}}$) and safe interior walkers. For exposed walkers:
1. Their low fitness guarantees a minimum cloning probability (Lemma 11.4.1)
2. When cloned, their barrier penalty drops dramatically (Lemma 11.4.2)
3. This creates systematic negative drift proportional to the total boundary exposure

The N-uniformity follows from the fact that cloning probabilities and fitness differentials depend only on local geometric properties (barrier values), not swarm size.

**Key Mathematical Tools**:
- **Conditional expectation decomposition**: Split drift into cloning vs. non-cloning contributions
- **Lower bounds on cloning probability**: Use fitness gap from barrier function to bound $p_i$ below
- **Barrier reduction upon cloning**: Companion selection from safer regions guarantees expected barrier reduction
- **Concentration of measure**: Control stochastic jitter terms and dead walker revival contributions

---

## Key Steps

### Step 1: Establish Boundary-Exposed Set Structure

**Goal**: Formalize the set of dangerous walkers and relate $W_b$ to their contributions.

**Approach**:
- Define $\mathcal{E}_{\text{boundary}}(S) := \{i \in \mathcal{A}(S) : \varphi_{\text{barrier}}(x_i) > \phi_{\text{thresh}}\}$
- Define boundary-exposed mass: $M_{\text{boundary}}(S) := \frac{1}{N} \sum_{i \in \mathcal{E}_{\text{boundary}}} \varphi_{\text{barrier}}(x_i)$
- Show that when $W_b$ is large, most mass comes from exposed walkers:

$$
W_b(S) \leq M_{\text{boundary}}(S) + \frac{k_{\text{alive}}}{N} \phi_{\text{thresh}}
$$

**Technical detail**: The threshold $\phi_{\text{thresh}}$ should be chosen to ensure exposed walkers have significantly higher barrier penalty than the background.

---

### Step 2: Prove Enhanced Cloning Probability for Boundary-Exposed Walkers (Lemma 11.4.1)

**Goal**: Show that walkers in $\mathcal{E}_{\text{boundary}}$ have cloning probability uniformly bounded below.

**Approach**:
1. **Fitness deficit from barrier**: Use Lemma 11.2.2 (Fitness Gradient from Boundary Proximity) to show:

$$
V_{\text{fit},i} \leq V_{\text{fit},j} - f(\phi_{\text{thresh}})
$$

where $j$ is a safe interior walker and $f$ is monotonically increasing.

2. **Companion selection**: With positive probability $p_{\text{interior}} > 0$, walker $i$ selects a safe companion (even with spatial weighting, cannot eliminate interior mass).

3. **Cloning score lower bound**: Conditioning on safe companion selection:

$$
S_i = \frac{V_{\text{fit},j} - V_{\text{fit},i}}{V_{\text{fit},i} + \varepsilon_{\text{clone}}} \geq \frac{f(\phi_{\text{thresh}})}{V_{\text{pot,max}} + \varepsilon_{\text{clone}}}
$$

4. **Probability calculation**: Since cloning occurs when $S_i > T_i$ with $T_i \sim \text{Uniform}(0, p_{\max})$:

$$
p_i \geq \min\left(1, \frac{s_{\text{min}}(\phi_{\text{thresh}})}{p_{\max}}\right) \cdot p_{\text{interior}} =: p_{\text{boundary}}(\phi_{\text{thresh}})
$$

**Key observation**: This bound is **N-independent** because it depends only on:
- Barrier function geometry (determines $f(\phi_{\text{thresh}})$)
- Algorithmic parameters ($p_{\max}$, $\varepsilon_{\text{clone}}$, fitness weights)
- Interior mass availability (guaranteed by Safe Harbor Axiom)

---

### Step 3: Prove Barrier Reduction Upon Cloning (Lemma 11.4.2)

**Goal**: Show that cloning from a companion reduces expected barrier penalty.

**Approach**:
1. **Cloning mechanism**: New position is $x'_i = x_{c_i} + \sigma_x \zeta_i^x$ where $\zeta_i^x \sim \mathcal{N}(0, I_d)$.

2. **Expected barrier after cloning**:

$$
\mathbb{E}[\varphi_{\text{barrier}}(x'_i) \mid i \text{ clones}] = \mathbb{E}_{c_i}[\mathbb{E}_{\zeta}[\varphi_{\text{barrier}}(x_{c_i} + \sigma_x \zeta_i^x)]]
$$

3. **Safe companion case**: If $c_i \in \mathcal{I}_{\text{safe}}$ (interior), then $\varphi_{\text{barrier}}(x_{c_i}) = 0$. With small jitter ($\sigma_x < \delta_{\text{safe}}$), most jittered positions stay safe:

$$
\mathbb{E}[\varphi_{\text{barrier}}(x'_i) \mid c_i \text{ safe}] \leq C_{\text{jitter}}
$$

where $C_{\text{jitter}} = O(\sigma_x^2)$ accounts for tail probability of jittering into boundary region.

4. **General companion case**: Even if companion is not in deepest safe region, expected barrier is centered around $\varphi_{\text{barrier}}(x_{c_i})$, which is lower than $\varphi_{\text{barrier}}(x_i)$ on average (companions selected from higher-fitness, hence lower-barrier, walkers).

**Key observation**: The barrier reduction is dramatic when original walker has high barrier penalty and companion is from interior.

---

### Step 4: Decompose Expected Drift by Walker Status

**Goal**: Write $\mathbb{E}[\Delta W_b]$ as sum over walkers, conditioned on cloning/non-cloning and alive/dead status.

**Approach**:
1. **Split by swarms**: $\Delta W_b = \sum_{k=1,2} \Delta W_b^{(k)}$

2. **Split by cloning action for alive walkers**:

$$
\mathbb{E}[\Delta W_b^{(k)}] = \frac{1}{N} \sum_{i \in \mathcal{A}(S_k)} p_{k,i} \left[\mathbb{E}[\varphi_{\text{barrier}}(x'_{k,i}) \mid \text{clone}] - \varphi_{\text{barrier}}(x_{k,i})\right] + \text{(non-cloning term)} + \text{(dead revival term)}
$$

3. **Non-cloning contribution**: Walkers that don't clone retain their barrier penalty (no change in expectation, assuming no position update during cloning phase).

4. **Dead walker revival**: Dead walkers revive by cloning from alive companions. This adds bounded positive contribution $O(N_{\text{dead}}/N)$.

**Technical detail**: Must account for the fact that dead walkers are typically at or near boundary, so their revival contributes positively to $W_b$. However, if swarm is healthy ($k_{\text{alive}} \approx N$), this term is negligible.

---

### Step 5: Focus on Boundary-Exposed Walkers for Main Contraction

**Goal**: Show that boundary-exposed walkers provide dominant negative drift.

**Approach**:
1. **Isolate exposed walker contribution**:

$$
\begin{aligned}
\sum_{i \in \mathcal{E}_{\text{boundary}}(S_k)} p_{k,i} &\left[\mathbb{E}[\varphi_{\text{barrier}}(x'_{k,i}) \mid \text{clone}] - \varphi_{\text{barrier}}(x_{k,i})\right] \\
&\leq \sum_{i \in \mathcal{E}_{\text{boundary}}} p_{\text{boundary}} [C_{\text{jitter}} - \varphi_{\text{barrier}}(x_{k,i})]
\end{aligned}
$$

2. **Use lower bound on cloning probability**: From Step 2, $p_{k,i} \geq p_{\text{boundary}}(\phi_{\text{thresh}})$ for exposed walkers.

3. **Use barrier reduction**: From Step 3, expected barrier after cloning is $\leq C_{\text{jitter}}$.

4. **Collect terms**: Since $\varphi_{\text{barrier}}(x_{k,i}) > \phi_{\text{thresh}}$ for exposed walkers:

$$
\leq -p_{\text{boundary}} \sum_{i \in \mathcal{E}_{\text{boundary}}} [\varphi_{\text{barrier}}(x_{k,i}) - C_{\text{jitter}}]
$$

5. **Relate to total boundary mass**: When $W_b$ is large, most mass is in exposed set:

$$
\sum_{i \in \mathcal{E}_{\text{boundary}}} \varphi_{\text{barrier}}(x_{k,i}) \approx N \cdot M_{\text{boundary}}(S_k) \approx N \cdot W_b(S_k)
$$

6. **Result**: Main negative drift term is approximately

$$
\mathbb{E}[\Delta W_b^{(k)}]_{\text{main}} \approx -p_{\text{boundary}} W_b(S_k) + \text{(correction terms)}
$$

---

### Step 6: Bound Correction Terms (Interior Walkers, Dead Revival, Jitter)

**Goal**: Show all non-dominant terms contribute $O(1)$ or $O(N^{-1})$ offsets, not affecting contraction rate.

**Approach**:
1. **Interior (non-exposed) walkers**: These have $\varphi_{\text{barrier}}(x_i) \leq \phi_{\text{thresh}}$, so total contribution is:

$$
\frac{1}{N} \sum_{i \notin \mathcal{E}_{\text{boundary}}} \varphi_{\text{barrier}}(x_i) \leq \frac{k_{\text{alive}}}{N} \phi_{\text{thresh}} = O(1)
$$

2. **Dead walker revival**: Number of dead walkers is $N_{\text{dead}} = N - k_{\text{alive}}$. Each revives to expected barrier $\leq C_{\text{jitter}}$ (cloning from alive population). Total contribution:

$$
\frac{N_{\text{dead}}}{N} C_{\text{jitter}} = O(N^{-1}) \quad \text{if } k_{\text{alive}} = N - O(1)
$$

More generally, this is $O(N_{\text{dead}}/N)$, which is small in viable regimes.

3. **Position jitter for cloned walkers**: Already accounted in $C_{\text{jitter}}$ term in barrier reduction.

4. **Combine**: All correction terms sum to constant $C_b$:

$$
C_b = O(\sigma_x^2 + N^{-1} + \phi_{\text{thresh}})
$$

**Key observation**: These are **state-independent** bounds, not growing with $W_b$.

---

### Step 7: Assemble Final Drift Inequality

**Goal**: Combine Steps 5-6 to obtain the claimed Foster-Lyapunov inequality.

**Approach**:
1. **Combine swarms**: Sum over $k = 1, 2$:

$$
\mathbb{E}[\Delta W_b] = \mathbb{E}[\Delta W_b^{(1)}] + \mathbb{E}[\Delta W_b^{(2)}]
$$

2. **Apply bounds**: From Step 5, main contraction is $-p_{\text{boundary}} W_b(S_k)$ for each swarm. From Step 6, corrections are $O(1)$ per swarm.

3. **Result**:

$$
\mathbb{E}[\Delta W_b] \leq -p_{\text{boundary}} [W_b(S_1) + W_b(S_2)] + 2C_b = -p_{\text{boundary}} W_b + 2C_b
$$

4. **Identify constants**:
   - Contraction rate: $\kappa_b := p_{\text{boundary}}(\phi_{\text{thresh}})$
   - Offset: $C_b := 2C_{\text{total}}$ where $C_{\text{total}}$ includes jitter, dead revival, and interior walker contributions

5. **Verify N-independence**:
   - $\kappa_b$ depends only on $\phi_{\text{thresh}}$ and algorithmic parameters (Step 2)
   - $C_b$ is $O(\sigma_x^2 + N^{-1})$, which is $O(1)$ in large-N limit

**Final form**:

$$
\mathbb{E}_{\text{clone}}[\Delta W_b] \leq -\kappa_b W_b + C_b
$$

with $\kappa_b > 0$ independent of $N$ and $W_b$.

---

### Step 8: Verify Key Properties

**Goal**: Check that the drift inequality satisfies the stated theorem properties.

**Properties to verify**:

1. **Unconditional contraction**: When $W_b > C_b/\kappa_b$, drift is negative:

$$
\mathbb{E}[\Delta W_b] \leq -\kappa_b W_b + C_b = -\kappa_b \left(W_b - \frac{C_b}{\kappa_b}\right) < 0
$$

✓ Satisfied by construction.

2. **N-independence of $\kappa_b$**: From Step 2, $\kappa_b = p_{\text{boundary}}(\phi_{\text{thresh}})$ depends only on:
   - Barrier function geometry
   - Fitness scaling parameters ($\alpha$, $\beta$)
   - Companion selection geometry (guarantees $p_{\text{interior}} > 0$)

   None of these depend on $N$.

   ✓ Verified.

3. **Strengthening near danger**: As walkers approach boundary, $\varphi_{\text{barrier}}(x_i) \to \infty$, so:
   - More walkers enter $\mathcal{E}_{\text{boundary}}$ (set grows)
   - Their cloning probability increases (fitness gap widens)
   - Barrier reduction upon cloning is more dramatic

   This can be formalized by considering threshold-dependent rates: $\kappa_b(\phi)$ is increasing in $\phi$.

   ✓ Qualitatively correct; quantitative statement requires refined analysis.

4. **Complementarity with variance contraction**: Variance contraction (Chapter 10) provides cohesion ("stay together"), while boundary contraction provides safety ("stay away from danger"). Both are needed:
   - Variance contraction alone could cause swarm to collapse near boundary
   - Boundary contraction alone could cause swarm to spread throughout interior

   Together, they drive convergence to QSD supported in safe interior with bounded variance.

   ✓ Conceptually verified; full synthesis in Chapter 12.

---

## Technical Tools Required

### Foundational Framework
- **Def 2.4.1**: Boundary barrier function $\varphi_{\text{barrier}}(x)$ with divergence at $\partial \mathcal{X}_{\text{valid}}$
- **Def 3.3.1**: Boundary potential $W_b(S_1, S_2) = \frac{1}{N} \sum \varphi_{\text{barrier}}(x_i)$
- **Axiom 4.3** (Safe Harbor): Guarantees existence of safe interior region and barrier-based fitness penalties

### Fitness and Measurement
- **Chapter 5**: Measurement pipeline (reward calculation, standardization, fitness potential)
- **Lemma 11.2.2** (Fitness Gradient from Boundary Proximity): Boundary-exposed walkers have systematically lower fitness
- **Def 11.2.3**: Boundary-exposed set $\mathcal{E}_{\text{boundary}}$ and boundary-exposed mass $M_{\text{boundary}}$

### Cloning Mechanism
- **Chapter 9**: Formal definition of $\Psi_{\text{clone}}$ operator
- **Def 9.3.3**: Companion selection operator with spatial weighting
- **Def 9.4.2**: Cloning score and probability calculation
- **Def 9.5.1**: Position jitter upon cloning ($x'_i = x_{c_i} + \sigma_x \zeta_i^x$)

### Lemmas to Prove
- **Lemma 11.4.1** (Enhanced Cloning Probability Near Boundary): $p_i \geq p_{\text{boundary}}(\phi_{\text{thresh}}) > 0$ for exposed walkers
- **Lemma 11.4.2** (Expected Barrier Reduction for Cloned Walker): $\mathbb{E}[\varphi_{\text{barrier}}(x'_i) \mid \text{clone}] \leq C_{\text{jitter}}$ when companion is safe

### Probabilistic Tools
- **Conditional expectation**: Decompose $\mathbb{E}[\Delta W_b]$ by cloning status
- **Synchronous coupling**: Same random variables used for both swarms (threshold $T_i$, jitter $\zeta_i^x$)
- **Gaussian tail bounds**: Control jitter-induced boundary entry probability
- **Uniform distribution properties**: $P(S_i > T_i)$ calculation for $T_i \sim \text{Uniform}(0, p_{\max})$

### Analytical Techniques
- **Foster-Lyapunov drift theory**: Framework for proving exponential convergence via one-step drift inequalities
- **Decomposition by walker subsets**: Split sum over walkers into exposed vs. interior, alive vs. dead
- **Lower bound propagation**: Track how fitness gaps translate to probability bounds through nonlinear transformations

---

## Challenging Points

### 1. N-Uniformity of Companion Selection Probability

**Challenge**: Must show that $p_{\text{interior}} > 0$ uniformly in $N$.

**Issue**: Companion selection uses spatial weighting (e.g., $\varepsilon$-dependent kernels). As $N \to \infty$, if all walkers concentrate near boundary, could $p_{\text{interior}} \to 0$?

**Resolution**:
- Safe Harbor Axiom guarantees existence of safe interior region with positive measure
- Even with spatial weighting, cannot completely eliminate interior mass
- If swarm is dangerously concentrated near boundary (violating Safe Harbor), extinction probability becomes non-negligible, which is addressed separately (Corollary 11.5.2)

**Refined approach**: Condition on viable regime where $W_b < W_{\text{critical}}$. In this regime, sufficient interior mass exists by necessity.

### 2. Handling Dead Walker Revival

**Challenge**: Dead walkers revive by cloning from alive population. This adds positive contribution to $W_b$ (new walkers enter with some barrier penalty).

**Issue**: If many walkers are dead ($N_{\text{dead}} \gg 1$), does this positive contribution cancel the negative drift from exposed walkers?

**Resolution**:
- In viable regime (which the swarm must be in for QSD convergence), $k_{\text{alive}} \approx N$, so $N_{\text{dead}} = O(1)$
- Contribution scales as $\frac{N_{\text{dead}}}{N} C_{\text{jitter}} = O(N^{-1})$, which vanishes in large-N limit
- For finite $N$, this is absorbed into constant $C_b$

**Caveat**: If swarm is in extinction regime ($N_{\text{dead}} \approx N$), drift analysis breaks down—but this is expected, as extinction is a separate failure mode.

### 3. Relating Boundary-Exposed Mass to Total Boundary Potential

**Challenge**: Need to show $M_{\text{boundary}} \approx W_b$ when $W_b$ is large.

**Issue**: If many walkers are slightly below threshold ($\varphi_{\text{barrier}}(x_i) \approx \phi_{\text{thresh}}$), most of $W_b$ could come from non-exposed walkers, weakening the contraction.

**Resolution**:
- If $W_b$ is large but most mass is below threshold, then $\phi_{\text{thresh}}$ is too high—choose $\phi_{\text{thresh}}$ adaptively as fraction of $W_b$
- Alternatively, use decomposition:

$$
W_b = M_{\text{boundary}} + M_{\text{subthreshold}}
$$

where $M_{\text{subthreshold}} \leq \frac{k_{\text{alive}}}{N} \phi_{\text{thresh}}$ is bounded.

- Main contraction acts on $M_{\text{boundary}}$, which dominates when $W_b$ is large.

**Refinement**: The proof should include explicit bound:

$$
M_{\text{boundary}}(S) \geq W_b(S) - \frac{k_{\text{alive}}}{N} \phi_{\text{thresh}}
$$

### 4. Controlling Position Jitter Effects

**Challenge**: Position jitter $\sigma_x \zeta_i^x$ can cause safe walkers to enter boundary region, adding positive drift.

**Issue**: Is $C_{\text{jitter}} = O(\sigma_x^2)$ sufficient to ensure net negative drift when $W_b$ is large?

**Resolution**:
- Jitter is isotropic Gaussian with variance $\sigma_x^2$
- Probability of jittering from deep interior (distance $> \delta_{\text{safe}}$ from boundary) to boundary region is exponentially small (Gaussian tail)
- Most jitter contribution comes from walkers already near boundary—but these have high cloning probability and get replaced
- Net effect: $C_{\text{jitter}}$ is genuinely $O(\sigma_x^2)$, not growing with $W_b$

**Quantitative bound**: Use Chebyshev or Gaussian concentration:

$$
P(\|\zeta_i^x\| > \delta_{\text{safe}}/\sigma_x) \leq e^{-\delta_{\text{safe}}^2/(2\sigma_x^2)}
$$

If $\sigma_x \ll \delta_{\text{safe}}$, jitter-induced boundary entry is rare.

### 5. Ensuring $\kappa_b > 0$ Uniformly

**Challenge**: Must show contraction rate $\kappa_b = p_{\text{boundary}}(\phi_{\text{thresh}})$ is uniformly positive.

**Issue**: Could there be pathological configurations where exposed walkers fail to clone?

**Resolution**:
- By Lemma 11.2.2, fitness gap $f(\phi_{\text{thresh}})$ is positive and N-independent
- By companion selection geometry, $p_{\text{interior}} > 0$ (Safe Harbor Axiom)
- Combining: $\kappa_b \geq p_{\min} > 0$ for some universal constant

**Potential failure mode**: If $\phi_{\text{thresh}}$ is too small, fitness gap may be negligible. Solution: Choose $\phi_{\text{thresh}}$ large enough (in safe region where $\varphi_{\text{barrier}}$ starts growing).

### 6. Interaction with Velocity Variance

**Challenge**: Cloning also affects velocities (inelastic collision model). Could velocity changes indirectly affect positions in a way that increases $W_b$?

**Issue**: The drift inequality is for one-step cloning operator, which updates positions via jitter but not via velocity-driven motion.

**Resolution**:
- Cloning phase does not include kinetic evolution, so velocities don't affect positions during cloning
- Velocity changes affect future kinetic steps, but those are analyzed separately in kinetic operator analysis
- Within cloning operator analysis, position and velocity effects decouple

**Caveat**: In full system (cloning + kinetic), there is coupling. Chapter 12 addresses this via synergistic Lyapunov function.

---

## Proof Structure

### Logical Flow

The proof proceeds in three stages:

**Stage A: Setup and Decomposition (Steps 1, 4)**
1. Define boundary-exposed set and boundary-exposed mass
2. Decompose $\mathbb{E}[\Delta W_b]$ by walker status (exposed/interior, cloning/non-cloning, alive/dead)
3. Identify main contraction term (exposed walkers who clone) vs. correction terms

**Stage B: Mechanism Analysis (Steps 2, 3)**
1. Prove Lemma 11.4.1: Exposed walkers have enhanced cloning probability
   - Fitness deficit from barrier → fitness gap
   - Fitness gap + companion selection → cloning score lower bound
   - Cloning score → probability lower bound $p_{\text{boundary}}$
2. Prove Lemma 11.4.2: Cloning reduces barrier penalty
   - Safe companions have zero barrier
   - Position jitter has bounded effect
   - Expected barrier after cloning is $\leq C_{\text{jitter}}$

**Stage C: Assembly (Steps 5, 6, 7, 8)**
1. Bound main contraction term using Lemmas 11.4.1 + 11.4.2
2. Bound all correction terms (interior, dead, jitter) as $O(1)$
3. Combine to obtain Foster-Lyapunov inequality
4. Verify N-independence and key properties

### Dependency Graph

```
Fitness Gradient Lemma 11.2.2
         |
         v
Enhanced Cloning Probability 11.4.1 ----+
                                         |
Cloning Mechanism (Ch 9)                 |
         |                               |
         v                               |
Barrier Reduction 11.4.2 ----------------+
                                         |
                                         v
                           Main Theorem 11.3.1 (via Steps 1-8)
                                         |
                                         v
                           Corollaries (11.5.1, 11.5.2)
```

### Connection to Broader Framework

**Upstream dependencies**:
- **Chapter 4**: Foundational axioms, especially Safe Harbor (Axiom 4.3)
- **Chapter 5**: Measurement and fitness pipeline
- **Chapter 9**: Cloning operator definition

**Downstream consequences**:
- **Corollary 11.5.1**: Bounded boundary exposure in equilibrium ($\limsup \mathbb{E}[W_b] \leq C_b/\kappa_b$)
- **Corollary 11.5.2**: Exponentially suppressed extinction probability
- **Chapter 12**: Synergistic drift analysis combining variance contraction (Ch 10) + boundary contraction (Ch 11) + inter-swarm error

**Role in convergence proof**:
- Boundary contraction ensures swarm stays away from danger (safety)
- Variance contraction (Ch 10) ensures swarm cohesion (positional concentration)
- Together, they drive convergence to QSD supported in safe interior with bounded spread

---

## Refinements for Full Proof

The following refinements should be addressed when expanding this sketch to a complete proof:

### 1. Make Threshold Choice Explicit

**Sketch assumption**: Threshold $\phi_{\text{thresh}}$ is "chosen appropriately."

**Full proof needs**: Explicit specification of $\phi_{\text{thresh}}$ as function of:
- Safe interior width $\delta_{\text{safe}}$
- Barrier function growth rate near boundary
- Fitness sensitivity parameter $\beta$

**Proposed**: $\phi_{\text{thresh}} := \inf\{\varphi_{\text{barrier}}(x) : d(x, \partial \mathcal{X}_{\text{valid}}) = \delta_{\text{safe}}/2\}$

This ensures exposed walkers are genuinely in danger zone, not just edge of safe region.

### 2. Quantify $p_{\text{interior}}$ Lower Bound

**Sketch assumption**: Companion selection ensures $p_{\text{interior}} > 0$.

**Full proof needs**: Explicit bound on $p_{\text{interior}}$ in terms of:
- Spatial kernel bandwidth $\varepsilon$
- Safe interior measure $\mu(\mathcal{I}_{\text{safe}})$
- Swarm configuration (worst-case over viable states)

**Approach**: Use Safe Harbor Axiom to guarantee $\mu(\mathcal{I}_{\text{safe}}) > c_{\text{safe}} \mu(\mathcal{X}_{\text{valid}})$ for some $c_{\text{safe}} > 0$. Then spatial kernel must place at least $c_{\text{safe}}/2$ mass on interior (by measure preservation), yielding $p_{\text{interior}} \geq c_{\text{safe}}/2$.

### 3. Rigorous Treatment of Dead Walker Revival

**Sketch treatment**: Order-of-magnitude bound $O(N_{\text{dead}}/N)$.

**Full proof needs**: Detailed analysis of revival mechanism:
- Dead walkers revive by cloning from alive walkers
- New position is $x'_i = x_{c_i} + \sigma_x \zeta_i^x$ where $c_i$ is selected from alive population
- Expected barrier depends on alive population distribution

**Approach**: Condition on $k_{\text{alive}} \geq k_{\text{viable}}$ for some threshold. In this regime:

$$
\mathbb{E}\left[\frac{1}{N} \sum_{i \in \mathcal{D}} \varphi_{\text{barrier}}(x'_i)\right] \leq \frac{N_{\text{dead}}}{N} [\bar{W}_b + C_{\text{jitter}}]
$$

where $\bar{W}_b$ is average barrier of alive population. This is $O(N_{\text{dead}}/N)$ as claimed.

### 4. Address Coupling Between Swarms

**Sketch treatment**: Sum over $k = 1, 2$ independently.

**Full proof needs**: Verify that synchronous coupling doesn't introduce correlations that violate independence.

**Observation**: Synchronous coupling uses same random variables ($T_i$, $\zeta_i^x$) for both swarms, but cloning decisions are state-dependent, so outcomes differ. This is acceptable—the coupling is designed to minimize distance, and the drift bound still holds.

**Caveat**: If swarms have very different configurations (e.g., $S_1$ has many exposed walkers, $S_2$ has none), the contraction rates differ per swarm. The bound applies to total $W_b = W_b(S_1) + W_b(S_2)$, which is the sum.

### 5. Smooth vs. Discrete Barrier Functions

**Sketch assumption**: $\varphi_{\text{barrier}} \in C^2(\mathcal{X}_{\text{valid}})$ with bounded derivatives.

**Full proof needs**: Verify that all bounds (fitness gap, jitter effect) hold under stated smoothness.

**Potential issue**: If $\varphi_{\text{barrier}}$ grows very rapidly near boundary (e.g., logarithmic divergence), derivatives may be unbounded, affecting jitter terms.

**Resolution**: Proposition 4.3.2 constructs barrier function with controlled growth rate. For specific domain geometries (e.g., convex domains), explicit barrier functions (e.g., $\varphi(x) = 1/(d(x, \partial \mathcal{X}))^\alpha$ for $\alpha > 0$) satisfy requirements.

### 6. Extension to Velocity-Dependent Barriers

**Future extension**: If barrier function also depends on velocity ($\varphi_{\text{barrier}}(x, v)$), the analysis extends with modifications:
- Velocity variance expansion (from Ch 10) can increase barrier if velocities grow
- Synergistic analysis (Ch 12) must account for coupling

**Current scope**: Theorem as stated assumes barrier depends only on position, which is standard for boundary safety.

---

## Summary

This sketch outlines a direct proof strategy for the Complete Boundary Potential Drift Characterization theorem via Foster-Lyapunov analysis. The key steps are:

1. **Identify dangerous walkers** (boundary-exposed set)
2. **Prove they clone frequently** (fitness deficit → enhanced cloning probability)
3. **Prove cloning reduces danger** (safe companions → barrier reduction)
4. **Decompose drift** (main contraction from exposed walkers, bounded corrections)
5. **Assemble inequality** ($\mathbb{E}[\Delta W_b] \leq -\kappa_b W_b + C_b$)
6. **Verify N-uniformity** (all constants independent of $N$)

The main technical challenges are:
- Ensuring $p_{\text{interior}} > 0$ uniformly (Safe Harbor Axiom)
- Controlling jitter-induced boundary entry (Gaussian concentration)
- Relating exposed mass to total boundary potential (threshold choice)

All challenges are tractable using the framework's foundational axioms and standard probabilistic tools.

**Next stage**: Expand to full proof (Stage 2/3) with detailed epsilon-delta arguments, explicit constants, and rigorous justifications for all bounds.
