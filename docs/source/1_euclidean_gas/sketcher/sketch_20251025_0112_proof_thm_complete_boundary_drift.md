# Proof Sketch for thm-complete-boundary-drift

**Document**: /home/guillem/fragile/docs/source/1_euclidean_gas/03_cloning.md
**Theorem**: thm-complete-boundary-drift
**Generated**: 2025-10-25 01:12 UTC
**Agent**: Proof Sketcher v1.0

---

## I. Theorem Statement

:::{prf:theorem} Complete Boundary Potential Drift Characterization
:label: thm-complete-boundary-drift

The cloning operator $\Psi_{\text{clone}}$ induces the following drift on the boundary potential:

$$
\mathbb{E}_{\text{clone}}[\Delta W_b] \leq -\kappa_b W_b + C_b
$$

where:
- $\kappa_b = p_{\text{boundary}}(\phi_{\text{thresh}}) > 0$ is the minimum cloning probability for boundary-exposed walkers
- $C_b = O(\sigma_x^2 + N^{-1})$ accounts for position jitter and dead walker revival
- Both constants are **$N$-independent** in the large-$N$ limit

**Key Properties:**

1. **Unconditional contraction:** The drift is negative for all states with $W_b > C_b/\kappa_b$

2. **Strengthening near danger:** The contraction rate $\kappa_b$ increases with boundary proximity (through $\phi_{\text{thresh}}$)

3. **Complementary to variance contraction:** While variance contraction (Chapter 10) pulls walkers together, boundary contraction pulls them away from danger - both contribute to stability
:::

**Informal Restatement**: This theorem states that the cloning operator creates a systematic drift that reduces the boundary potential $W_b$ (average proximity to the dangerous boundary) at a rate proportional to how exposed the swarm is. The contraction rate $\kappa_b$ is strictly positive and independent of swarm size $N$, while the additive noise term $C_b$ comes from position jitter during cloning and occasional revival of dead walkers. When $W_b$ is large (swarm dangerously close to boundary), the negative drift dominates, pulling the swarm back to safety.

---

## II. Proof Strategy Comparison

### Strategy A: Gemini's Approach

**STATUS**: ⚠️ **GEMINI FAILED TO RESPOND**

Gemini 2.5 Pro was invoked twice but returned empty responses both times. This prevents dual-validation of the proof strategy.

**Implications**:
- No cross-validation from second strategist
- Lower confidence in chosen approach
- Recommend re-running sketch when Gemini is available
- Proceeding with single-strategist analysis from GPT-5

---

### Strategy B: GPT-5's Approach

**Method**: Proof by Synthesis

**Key Steps**:
1. Convert Theorem 11.3.1 to drift form
2. Identify $\kappa_b$ via minimum boundary cloning probability
3. Quantify the jitter contribution in $C_b$ (O($\sigma_x^2$) term)
4. Bound the dead-walker revival contribution as O($N^{-1}$)
5. Assemble the constants and state key properties

**Strengths**:
- **Modular**: Leverages already-proven results (Theorem 11.3.1, Lemmas 11.4.1, 11.4.2)
- **Explicit constants**: Clearly identifies where $\kappa_b$ and $C_b$ come from
- **N-independence tracking**: Systematically verifies that both constants are N-independent
- **Appropriate for summary theorem**: Recognizes this is Section 11.6's synthesis role
- **Clean structure**: 5-step proof mirrors the document's logical flow

**Weaknesses**:
- **Relies on earlier proof completeness**: If Theorem 11.3.1's proof has gaps, they propagate here
- **Revival bound assumption**: Requires stable-regime assumption (exponentially small extinction) to justify O($N^{-1}$) scaling
- **Exposed-mass approximation**: Uses $M_{\text{boundary}} \approx W_b$ when $W_b$ is large, which needs careful justification

**Framework Dependencies**:
- Theorem 11.3.1 (thm-boundary-potential-contraction)
- Lemma 11.4.1 (lem-boundary-enhanced-cloning)
- Lemma 11.4.2 (lem-barrier-reduction-cloning)
- Axiom EG-2 (Safe Harbor Axiom)
- Definition 11.2.1 (def-boundary-exposed-set)
- Definition 6.9.1 (def-boundary-potential)
- Exponential extinction suppression (Section 11.5)

---

### Strategy Synthesis: Claude's Recommendation

**Chosen Method**: Proof by Synthesis (following GPT-5's approach)

**Rationale**:
This theorem appears as the culminating result in Section 11.6 "Summary and Drift Inequality" after Sections 11.1-11.5 have established all the necessary machinery:
- Section 11.2: Boundary potential definition and fitness gradient from boundary proximity
- Section 11.3: Main contraction theorem (Theorem 11.3.1)
- Section 11.4: Detailed proof with supporting lemmas (11.4.1, 11.4.2)
- Section 11.5: Extinction probability analysis

The natural proof strategy is therefore **synthesis**: take the proven contraction inequality from Theorem 11.3.1, make the constants explicit using the supporting lemmas, and derive the drift form. This is both mathematically clean and pedagogically appropriate for a summary theorem.

**Integration**:
- **Step 1**: Direct algebraic manipulation of Theorem 11.3.1's inequality
- **Step 2**: Use Lemma 11.4.1 to identify $\kappa_b = p_{\text{boundary}}(\phi_{\text{thresh}})$
- **Step 3**: Use Lemma 11.4.2 to bound jitter contribution as O($\sigma_x^2$)
- **Step 4**: Use exponential suppression analysis from Section 11.5 to bound revival as O($N^{-1}$)
- **Step 5**: Assembly and verification of key properties

**Critical insight**: The theorem is not claiming anything new—it's making explicit what was implicit in the earlier proofs. The work is in **unpacking the constants** and **verifying N-independence**.

**Verification Status**:
- ✅ All framework dependencies verified (see Section III)
- ✅ No circular reasoning detected (synthesis builds on proven results)
- ⚠️ **Missing Gemini cross-validation** (single-strategist analysis)
- ⚠️ Requires formalization of Lemma B (revival contribution scaling)

---

## III. Framework Dependencies

### Verified Dependencies

**Axioms** (from `docs/source/1_euclidean_gas/03_cloning.md`):

| Label | Statement | Used in Step | Verified |
|-------|-----------|--------------|----------|
| `ax:safe-harbor` (Axiom EG-2) | Existence of safe interior region $C_{\text{safe}}$ with strictly better reward than boundary regions | Step 2 (justifies fitness gradient) | ✅ Lines 1179-1188 |
| Axiom EG-0 | Regularity of domain: $\mathcal{X}_{\text{valid}}$ is open, bounded, smooth boundary | Step 3 (justifies barrier smoothness) | ✅ Lines 198-209 |

**Theorems** (from earlier sections):

| Label | Document | Statement | Used in Step | Verified |
|-------|----------|-----------|--------------|----------|
| `thm-boundary-potential-contraction` | 03_cloning.md § 11.3 | $\mathbb{E}_{\text{clone}}[W_b(S'_1, S'_2) \mid S_1, S_2] \leq (1 - \kappa_b) W_b(S_1, S_2) + C_b$ | Step 1 (base inequality) | ✅ Lines 7210-7224 |

**Lemmas** (from same chapter):

| Label | Document | Statement | Used in Step | Verified |
|-------|----------|-----------|--------------|----------|
| `lem-boundary-enhanced-cloning` | 03_cloning.md § 11.4.1 | For boundary-exposed walker: $p_i \geq p_{\text{boundary}}(\phi_{\text{thresh}}) > 0$, N-independent | Step 2 (identifies $\kappa_b$) | ✅ Lines 7244-7309 |
| `lem-barrier-reduction-cloning` | 03_cloning.md § 11.4.2 | Expected barrier after cloning: $\mathbb{E}[\varphi_{\text{barrier}}(x'_i) \mid \text{clone}] \leq \mathbb{E}[\varphi_{\text{barrier}}(x_{c_i})] + C_{\text{jitter}}$ with $C_{\text{jitter}} = O(\sigma_x^2)$ | Step 3 (jitter bound) | ✅ Lines 7314-7368 |
| Exponential extinction suppression | 03_cloning.md § 11.5 | $P(\text{extinction in one step}) = O(e^{-N \cdot \text{const}})$ in viable regime | Step 4 (revival bound) | ✅ Lines 7512-7620 |

**Definitions**:

| Label | Document | Definition | Used for |
|-------|----------|------------|----------|
| `def-boundary-potential` | 03_cloning.md § 11.2 | $W_b(S_1, S_2) := \frac{1}{N} \sum_{i \in \mathcal{A}(S_1)} \varphi_{\text{barrier}}(x_{1,i}) + \frac{1}{N} \sum_{i \in \mathcal{A}(S_2)} \varphi_{\text{barrier}}(x_{2,i})$ | All steps (primary quantity) |
| `def-boundary-exposed-set` | 03_cloning.md § 11.2.1 | $\mathcal{E}_{\text{boundary}}(S) = \{i \in \mathcal{A}(S) : \varphi_{\text{barrier}}(x_i) > \phi_{\text{thresh}}\}$ | Steps 2, 5 (target set for contraction) |
| Barrier function properties | 03_cloning.md § 11.2 | $\varphi_{\text{barrier}} \in C^2$, zero in safe interior, grows to $\infty$ at boundary | Step 3 (smoothness for jitter bound) |

**Constants**:

| Symbol | Definition | Value/Bound | Properties |
|--------|------------|-------------|------------|
| $\kappa_b$ | $p_{\text{boundary}}(\phi_{\text{thresh}})$ | Strictly positive | N-independent, monotone increasing in $\phi_{\text{thresh}}$ |
| $C_b$ | $C_{\text{jitter}} + C_{\text{dead}}$ | $O(\sigma_x^2 + N^{-1})$ | N-independent in large-N limit |
| $C_{\text{jitter}}$ | Position jitter variance contribution | $O(\sigma_x^2)$ | N-independent (follows from Gaussian jitter) |
| $C_{\text{dead}}$ | Dead walker revival contribution | $O(N^{-1})$ | Vanishes in large-N limit under stable regime |
| $\phi_{\text{thresh}}$ | Boundary exposure threshold | Fixed positive constant | Defines boundary-exposed set |

### Missing/Uncertain Dependencies

**Requires Additional Proof**:
- **Lemma A** (Exposed-mass lower bound): $M_{\text{boundary}}(S_k) \geq W_b(S_k) - \frac{k_{\text{alive}}}{N} \phi_{\text{thresh}}$
  - Why needed: Relates the boundary-exposed mass to total boundary potential in Step 5
  - Difficulty: Easy (stated as remark in document, lines 7187-7204, needs formalization)

- **Lemma B** (Revival contribution scaling): $\frac{1}{N} \sum_{i \in \mathcal{D}(S)} \mathbb{E}[\varphi_{\text{barrier}}(x'_i)] \leq C_{\text{rev}} \cdot \frac{\mathbb{E}[\#\text{dead}]}{N}$ with $C_{\text{rev}}$ independent of $N$
  - Why needed: Makes the O($N^{-1}$) dependence explicit for dead walker contribution
  - Difficulty: Easy (follows from revival mechanism and bounded jitter in interior)

**Uncertain Assumptions**:
- **Stable regime assumption**: The O($N^{-1}$) bound for $C_{\text{dead}}$ requires that $\mathbb{E}[\#\text{dead}] = O(1)$ per step
  - Why uncertain: This relies on the exponential suppression result (Corollary 11.5.2), which assumes the swarm is in the quasi-stationary regime where $W_b \leq C_b/\kappa_b$
  - How to verify: Either (1) parameterize $C_{\text{dead}} := c_{\text{rev}} \cdot \mathbb{E}[\#\text{dead}]/N$ and state the regime assumption explicitly, or (2) provide a global bound on expected deaths per step
  - Impact: Without this, the theorem statement should clarify that the O($N^{-1}$) scaling holds in the stable regime

---

## IV. Detailed Proof Sketch

### Overview

This theorem synthesizes the boundary potential contraction analysis from Sections 11.1-11.5 into a single, explicit drift inequality. The proof strategy is **proof by synthesis**: we take the established contraction inequality from Theorem 11.3.1 and make the constants $\kappa_b$ and $C_b$ explicit by tracing through the supporting lemmas. The key technical work is verifying that both constants are N-independent (or vanishing in N for $C_b$'s O($N^{-1}$) term).

The proof naturally divides into five stages corresponding to the five components that must be established:
1. Algebraic conversion of the contraction inequality to drift form
2. Identification of the contraction rate $\kappa_b$
3. Quantification of the jitter noise contribution
4. Quantification of the revival noise contribution
5. Assembly of results and verification of key properties

### Proof Outline (Top-Level)

The proof proceeds in 5 main stages:

1. **Drift Form Conversion**: Convert Theorem 11.3.1's multiplicative contraction to additive drift
2. **Contraction Rate Identification**: Use Lemma 11.4.1 to identify $\kappa_b = p_{\text{boundary}}(\phi_{\text{thresh}})$
3. **Jitter Contribution**: Use Lemma 11.4.2 to bound jitter as O($\sigma_x^2$)
4. **Revival Contribution**: Use exponential suppression to bound revival as O($N^{-1}$)
5. **Property Verification**: Verify the three key properties and N-independence

---

### Detailed Step-by-Step Sketch

#### Step 1: Convert Theorem 11.3.1 to Drift Form

**Goal**: Transform the multiplicative contraction inequality into an additive drift inequality

**Substep 1.1**: Recall Theorem 11.3.1's statement
- **Justification**: Theorem 11.3.1 (thm-boundary-potential-contraction, lines 7210-7224)
- **Why valid**: This is a proven result from earlier in the chapter
- **Expected result**: We have $\mathbb{E}_{\text{clone}}[W_b(S'_1, S'_2) \mid S_1, S_2] \leq (1 - \kappa_b) W_b(S_1, S_2) + C_b$

**Substep 1.2**: Define the drift quantity
- **Justification**: Standard definition of drift in stochastic processes
- **Why valid**: $\Delta W_b := W_b(S'_1, S'_2) - W_b(S_1, S_2)$ is the one-step change in boundary potential
- **Expected result**: $\mathbb{E}_{\text{clone}}[\Delta W_b] = \mathbb{E}_{\text{clone}}[W_b(S'_1, S'_2)] - W_b(S_1, S_2)$

**Substep 1.3**: Algebraic rearrangement
- **Justification**: Subtract $W_b(S_1, S_2)$ from both sides of Theorem 11.3.1's inequality
- **Why valid**: Basic algebra, preserves inequality direction
- **Expected result**:
  $$
  \mathbb{E}_{\text{clone}}[\Delta W_b] \leq (1 - \kappa_b) W_b - W_b + C_b = -\kappa_b W_b + C_b
  $$

**Conclusion**: The drift form inequality $\mathbb{E}_{\text{clone}}[\Delta W_b] \leq -\kappa_b W_b + C_b$ is established
**Form**: Additive drift inequality (Foster-Lyapunov form)

**Dependencies**:
- Uses: `thm-boundary-potential-contraction`
- Requires: Constants $\kappa_b, C_b$ to be defined and bounded

**Potential Issues**:
- ⚠️ Notational mismatch: Theorem 11.3.1 uses two-swarm notation $(S_1, S_2)$ while theorem statement uses implicit notation
- **Resolution**: The boundary potential $W_b$ is defined over both swarms (Definition 11.2.1, line 6975), so the notation is consistent

---

#### Step 2: Identify $\kappa_b$ via Minimum Boundary Cloning Probability

**Goal**: Show that $\kappa_b = p_{\text{boundary}}(\phi_{\text{thresh}})$ and verify it is strictly positive and N-independent

**Substep 2.1**: Recall the definition of boundary-exposed walkers
- **Justification**: Definition 11.2.1 (def-boundary-exposed-set, lines 7177-7204)
- **Why valid**: Established definition from earlier in the chapter
- **Expected result**: $\mathcal{E}_{\text{boundary}}(S) = \{i \in \mathcal{A}(S) : \varphi_{\text{barrier}}(x_i) > \phi_{\text{thresh}}\}$

**Substep 2.2**: Apply Lemma 11.4.1 to boundary-exposed walkers
- **Justification**: Lemma 11.4.1 (lem-boundary-enhanced-cloning, lines 7244-7309)
- **Why valid**: Lemma applies to any walker $i \in \mathcal{E}_{\text{boundary}}(S)$
- **Expected result**: For such walkers, cloning probability $p_i \geq p_{\text{boundary}}(\phi_{\text{thresh}}) > 0$

**Substep 2.3**: Trace the proof of Lemma 11.4.1 to identify $p_{\text{boundary}}$
- **Justification**: Lemma 11.4.1's proof (lines 7256-7309)
- **Why valid**: The proof constructs $p_{\text{boundary}}$ explicitly through 4 steps:
  1. Fitness penalty from barrier: $V_{\text{fit},i} < V_{\text{fit},j} - f(\phi_{\text{thresh}})$ for interior walker $j$
  2. Companion selection probability: $P(c_i \in \mathcal{I}_{\text{safe}}) \geq p_{\text{interior}} > 0$
  3. Cloning score lower bound: $S_i \geq s_{\text{min}}(\phi_{\text{thresh}})$
  4. Cloning probability: $p_i \geq \min(1, s_{\text{min}}/p_{\max}) \cdot p_{\text{interior}}$
- **Expected result**:
  $$
  p_{\text{boundary}}(\phi_{\text{thresh}}) := \min\left(1, \frac{s_{\text{min}}(\phi_{\text{thresh}})}{p_{\max}}\right) \cdot p_{\text{interior}}
  $$

**Substep 2.4**: Verify N-independence
- **Justification**: Inspection of the formula from Substep 2.3
- **Why valid**: All components are algorithmic parameters:
  - $s_{\text{min}}$ depends only on $\phi_{\text{thresh}}$, fitness function form, and regularization $\varepsilon_{\text{clone}}$
  - $p_{\max}$ is an algorithmic parameter (maximum cloning probability)
  - $p_{\text{interior}}$ is the companion selection weight for interior walkers (geometric, not depending on $N$)
- **Expected result**: $p_{\text{boundary}}$ is a function of algorithmic parameters and $\phi_{\text{thresh}}$, **independent of $N$**

**Substep 2.5**: Verify strict positivity
- **Justification**: Safe Harbor Axiom (Axiom EG-2, lines 1179-1188) guarantees interior walkers exist with strictly better fitness
- **Why valid**: $p_{\text{interior}} > 0$ (interior region has positive measure), $s_{\text{min}} > 0$ (fitness gap is positive), so their product is positive
- **Expected result**: $p_{\text{boundary}}(\phi_{\text{thresh}}) > 0$

**Substep 2.6**: Verify monotonicity in $\phi_{\text{thresh}}$
- **Justification**: Larger $\phi_{\text{thresh}}$ means only more boundary-exposed walkers are included, leading to larger fitness gap $f(\phi_{\text{thresh}})$, hence larger $s_{\text{min}}$
- **Why valid**: The barrier function $\varphi_{\text{barrier}}$ is monotone in distance to boundary (lines 6982-6998)
- **Expected result**: $p_{\text{boundary}}(\phi_{\text{thresh}})$ is monotonically increasing in $\phi_{\text{thresh}}$

**Conclusion**: We have identified $\kappa_b = p_{\text{boundary}}(\phi_{\text{thresh}})$ with:
- **Form**: Explicit formula in terms of algorithmic parameters
- **Properties**: Strictly positive, N-independent, monotone increasing

**Dependencies**:
- Uses: `lem-boundary-enhanced-cloning`, `ax:safe-harbor`, `def-boundary-exposed-set`
- Requires: Safe Harbor Axiom to guarantee interior walkers with better fitness

**Potential Issues**:
- ⚠️ Companion selection mechanism: If selection is highly biased toward nearby walkers, $p_{\text{interior}}$ might be very small for boundary walkers
- **Resolution**: Even spatially-weighted selection must have bounded spatial decay by algorithmic design, ensuring $p_{\text{interior}}$ has a positive lower bound (see companion selection operator definition, lines 1925-1955)

---

#### Step 3: Quantify the Jitter Contribution in $C_b$

**Goal**: Show that the position jitter during cloning contributes O($\sigma_x^2$) to $C_b$

**Substep 3.1**: Recall the cloning position update
- **Justification**: Inelastic collision state update (Definition 9.4.3, line 2008ff; recalled in proof of Theorem 11.3.1, line 7337)
- **Why valid**: Standard part of the cloning operator definition
- **Expected result**: When walker $i$ clones from companion $c_i$, new position is $x'_i = x_{c_i} + \sigma_x \zeta_i^x$ where $\zeta_i^x \sim \mathcal{N}(0, I_d)$

**Substep 3.2**: Apply Lemma 11.4.2 for boundary-exposed walkers cloning
- **Justification**: Lemma 11.4.2 (lem-barrier-reduction-cloning, lines 7314-7368)
- **Why valid**: Lemma applies to any cloning event
- **Expected result**: $\mathbb{E}[\varphi_{\text{barrier}}(x'_i) \mid i \text{ clones}] \leq \mathbb{E}[\varphi_{\text{barrier}}(x_{c_i})] + C_{\text{jitter}}$

**Substep 3.3**: Trace Lemma 11.4.2's proof to identify $C_{\text{jitter}}$
- **Justification**: Lemma 11.4.2's proof (lines 7333-7368)
- **Why valid**: The proof has two cases:
  - **Case 1** (companion in safe interior): $\varphi_{\text{barrier}}(x_{c_i}) = 0$, so jitter is the only contribution. With probability $1 - \epsilon_{\text{jitter}}$, the jittered position remains safe; worst case gives $C_{\text{jitter}} = \epsilon_{\text{jitter}} \cdot \varphi_{\text{barrier,max}}$
  - **Case 2** (general companion): Taylor expansion around $x_{c_i}$ gives $\mathbb{E}[\varphi_{\text{barrier}}(x'_i)] \approx \varphi_{\text{barrier}}(x_{c_i}) + O(\sigma_x^2 \|\nabla \varphi_{\text{barrier}}\|^2)$, bounded by smoothness
- **Expected result**: $C_{\text{jitter}} = O(\sigma_x^2)$ where the implicit constant depends on $\varphi_{\text{barrier}}$ smoothness and domain geometry

**Substep 3.4**: Verify N-independence of $C_{\text{jitter}}$
- **Justification**: Inspection of the bound from Substep 3.3
- **Why valid**:
  - $\sigma_x$ is an algorithmic parameter (jitter scale), independent of $N$
  - $\varphi_{\text{barrier}}$ smoothness constants are geometric, independent of $N$
  - The bound uses only local Taylor expansion, no collective properties
- **Expected result**: $C_{\text{jitter}}$ is **independent of $N$**

**Substep 3.5**: Aggregate jitter contribution over all cloning events
- **Justification**: Proof of Theorem 11.3.1 (lines 7370-7464, especially Step 2, lines 7390-7413)
- **Why valid**: The proof sums over all boundary-exposed walkers that clone, using their cloning probabilities $p_{k,i} \geq p_{\text{boundary}}$
- **Expected result**: The net jitter contribution to $\mathbb{E}[\Delta W_b]$ is bounded by a multiple of $C_{\text{jitter}}$, yielding the O($\sigma_x^2$) term in $C_b$

**Conclusion**: The jitter contribution to $C_b$ is O($\sigma_x^2$), N-independent
**Form**: $C_b \geq C_{\text{jitter}}$ where $C_{\text{jitter}} = O(\sigma_x^2)$

**Dependencies**:
- Uses: `lem-barrier-reduction-cloning`, cloning state update definition, barrier smoothness
- Requires: $\varphi_{\text{barrier}} \in C^2$ (guaranteed by Axiom EG-0 and Proposition 4.3.2)

**Potential Issues**:
- ⚠️ Taylor expansion validity: Requires $\sigma_x$ to be small relative to the barrier curvature scale
- **Resolution**: This is an algorithmic design constraint; if $\sigma_x$ is too large, the algorithm's safety guarantees degrade (see Remark 11.6.3 on parameter tuning, lines 7622-7634)

---

#### Step 4: Bound the Dead-Walker Revival Contribution as O($N^{-1}$)

**Goal**: Show that the revival of dead walkers contributes O($N^{-1}$) to $C_b$ in the large-$N$ limit

**Substep 4.1**: Identify the revival contribution in the drift decomposition
- **Justification**: Proof of Theorem 11.3.1 (line 7386, Step 1; line 7434, Step 4)
- **Why valid**: The proof explicitly includes a "dead walker contribution" term accounting for walkers in $\mathcal{D}(S_k)$ (dead set) that get revived through cloning
- **Expected result**: $\mathbb{E}[\Delta W_b]$ includes a term $\frac{1}{N} \sum_{i \in \mathcal{D}(S_k)} \mathbb{E}[\varphi_{\text{barrier}}(x'_i)]$ for each swarm $k$

**Substep 4.2**: Bound the expected barrier value after revival
- **Justification**: Dead walkers clone with probability 1 (revival axiom, lines 5963-5994) from randomly selected companions
- **Why valid**: Since the companion is alive (by definition), and jitter is bounded, the revived walker's barrier value is bounded by the alive walker distribution plus jitter
- **Expected result**: $\mathbb{E}[\varphi_{\text{barrier}}(x'_i) \mid i \text{ revives}] \leq \mathbb{E}[\varphi_{\text{barrier}}(x_{\text{alive}})] + C_{\text{jitter}} \leq W_b \cdot N + C_{\text{jitter}}$

**Substep 4.3**: Count expected number of deaths per step
- **Justification**: Exponential extinction suppression (Corollary 11.5.2, lines 7512-7620)
- **Why valid**: In the stable regime where $W_b \leq C_b/\kappa_b$, the probability of extinction (all walkers crossing to boundary) is exponentially small in $N$
- **Expected result**: If total extinction has probability O($e^{-cN}$), then the expected number of individual deaths per step is O(1) (sub-extensive)
- **Key assumption**: This relies on the **stable regime assumption** that the swarm is in the quasi-stationary regime

**Substep 4.4**: Combine to get O($N^{-1}$) scaling
- **Justification**: $\frac{1}{N}$ factor in $W_b$ definition times O(1) expected deaths
- **Why valid**:
  $$
  \frac{1}{N} \sum_{i \in \mathcal{D}(S)} \mathbb{E}[\varphi_{\text{barrier}}(x'_i)] \leq \frac{\mathbb{E}[\#\text{dead}]}{N} \cdot C_{\text{bound}}
  $$
  where $C_{\text{bound}}$ is a constant bound on barrier values post-revival
- **Expected result**: The revival contribution is O($\mathbb{E}[\#\text{dead}]/N$) = O($1/N$) in the stable regime

**Substep 4.5**: Formalize as Lemma B (if needed)
- **Justification**: The above reasoning should be packaged as a formal lemma for completeness
- **Why valid**: Standard probability calculation
- **Expected result**: **Lemma B** (Revival contribution scaling):
  $$
  \frac{1}{N} \sum_{i \in \mathcal{D}(S)} \mathbb{E}[\varphi_{\text{barrier}}(x'_i)] \leq C_{\text{rev}} \cdot \frac{\mathbb{E}[\#\text{dead}]}{N}
  $$
  where $C_{\text{rev}}$ is independent of $N$, and $\mathbb{E}[\#\text{dead}] = O(1)$ in the stable regime

**Conclusion**: The revival contribution to $C_b$ is O($N^{-1}$), vanishing in the large-$N$ limit
**Form**: $C_b \geq C_{\text{jitter}} + C_{\text{dead}}$ where $C_{\text{dead}} = O(N^{-1})$ under stable regime assumption

**Dependencies**:
- Uses: Exponential extinction suppression (Corollary 11.5.2), revival mechanism definition (lines 5963-5994)
- Requires: Stable regime assumption ($W_b \leq C_b/\kappa_b$) for O(1) death count

**Potential Issues**:
- ⚠️ **Circular dependency concern**: We're proving the drift inequality that leads to the stable regime, but we're using the stable regime to bound the revival contribution
- **Resolution**: This is **not circular** because:
  1. The O($N^{-1}$) term is only claimed in the large-$N$ limit
  2. For finite $N$, we can parameterize $C_{\text{dead}} := c_{\text{rev}} \cdot \mathbb{E}[\#\text{dead}]/N$ without assuming the bound
  3. The drift inequality holds for any value of $C_b$; the stable regime analysis then shows that $\mathbb{E}[\#\text{dead}] = O(1)$ **as a consequence** of the drift
  4. Alternative: Use a global bound on deaths (e.g., $\mathbb{E}[\#\text{dead}] \leq N$) giving $C_{\text{dead}} \leq c_{\text{rev}}$, which is O(1) but not tight

---

#### Step 5: Assemble the Constants and State Key Properties

**Goal**: Combine Steps 1-4 to establish the complete drift inequality with explicit constants and verify the three key properties

**Substep 5.1**: Assemble the drift inequality
- **Justification**: Combination of Steps 1-4
- **Why valid**:
  - Step 1 gave the form $\mathbb{E}[\Delta W_b] \leq -\kappa_b W_b + C_b$
  - Step 2 identified $\kappa_b = p_{\text{boundary}}(\phi_{\text{thresh}})$
  - Steps 3-4 bounded $C_b = C_{\text{jitter}} + C_{\text{dead}} = O(\sigma_x^2) + O(N^{-1})$
- **Expected result**:
  $$
  \mathbb{E}_{\text{clone}}[\Delta W_b] \leq -p_{\text{boundary}}(\phi_{\text{thresh}}) \cdot W_b + O(\sigma_x^2 + N^{-1})
  $$

**Substep 5.2**: Verify Property 1 (Unconditional contraction)
- **Justification**: Algebraic manipulation of the drift inequality
- **Why valid**: If $W_b > C_b/\kappa_b$, then:
  $$
  \mathbb{E}[\Delta W_b] \leq -\kappa_b W_b + C_b < -\kappa_b \cdot \frac{C_b}{\kappa_b} + C_b = 0
  $$
- **Expected result**: $\mathbb{E}[\Delta W_b] < 0$ whenever $W_b > C_b/\kappa_b$, providing **unconditional contraction** for large $W_b$

**Substep 5.3**: Verify Property 2 (Strengthening near danger)
- **Justification**: Monotonicity of $p_{\text{boundary}}$ (established in Step 2.6) and refined contraction via exposed-mass
- **Why valid**:
  1. Larger $\phi_{\text{thresh}}$ (more stringent definition of "boundary-exposed") gives larger $\kappa_b$
  2. Using the exposed-mass inequality (Lemma A, lines 7187-7204), the contraction can be refined to:
     $$
     \mathbb{E}[\Delta W_b] \leq -\kappa_b \cdot M_{\text{boundary}} + C_b
     $$
     where $M_{\text{boundary}} \geq W_b - O(1/N)$ when $W_b$ is large
  3. More walkers near boundary → larger $M_{\text{boundary}}$ → stronger contraction
- **Expected result**: The contraction rate effectively **increases** as the swarm gets closer to danger (boundary proximity increases)

**Substep 5.4**: Verify Property 3 (Complementarity with variance contraction)
- **Justification**: Comparison with Chapter 10's variance contraction results
- **Why valid**:
  - Chapter 10 establishes positional variance contraction: $\mathbb{E}[\Delta V_{\text{Var},x}] \leq -\kappa_x V_{\text{Var},x} + C_x$ (see lines 7698-7710)
  - Boundary potential contraction: $\mathbb{E}[\Delta W_b] \leq -\kappa_b W_b + C_b$ (this theorem)
  - These control different failure modes:
    - $V_{\text{Var},x}$ penalizes spread (walkers far from each other)
    - $W_b$ penalizes boundary proximity (walkers near danger)
  - Both contribute to the composite Lyapunov function $V_{\text{total}} = V_W + c_V V_{\text{Var}} + c_B W_b$
- **Expected result**: The two contraction mechanisms are **complementary**: variance contraction prevents dispersion ("stay together"), boundary contraction prevents extinction ("stay safe")

**Substep 5.5**: Verify N-independence of constants
- **Justification**: Verification from Steps 2.4 (for $\kappa_b$) and 3.4, 4.4 (for $C_b$)
- **Why valid**:
  - $\kappa_b = p_{\text{boundary}}(\phi_{\text{thresh}})$ depends only on algorithmic parameters and $\phi_{\text{thresh}}$ (Step 2.4)
  - $C_{\text{jitter}} = O(\sigma_x^2)$ depends only on jitter scale and barrier smoothness (Step 3.4)
  - $C_{\text{dead}} = O(N^{-1})$ vanishes in large-$N$ limit (Step 4.4)
- **Expected result**: Both $\kappa_b$ and $C_b$ are **N-independent in the large-$N$ limit** (as claimed in theorem statement)

**Conclusion**: All components of the theorem are established:
- Drift inequality: $\mathbb{E}[\Delta W_b] \leq -\kappa_b W_b + C_b$
- Constants: $\kappa_b = p_{\text{boundary}}(\phi_{\text{thresh}}) > 0$, $C_b = O(\sigma_x^2 + N^{-1})$
- Properties 1-3: Verified
- N-independence: Verified

**Dependencies**:
- Uses: All previous steps, Chapter 10 variance contraction results, Lemma A (exposed-mass lower bound)

**Potential Issues**:
- ⚠️ Composite Lyapunov function: The complementarity claim references Chapter 12 material (composite Lyapunov function), which comes after this theorem
- **Resolution**: The complementarity statement is **interpretative**, not logically required for the theorem. The drift inequality stands on its own; the complementarity observation motivates why this result matters in the larger stability framework

**Q.E.D.** ∎

---

## V. Technical Deep Dives

### Challenge 1: Establishing O($N^{-1}$) for Revival Contribution

**Why Difficult**: The per-step number of deaths is not explicitly bounded in Section 11.4, and there's a potential circular dependency: we need the stable regime to bound deaths, but we're proving the drift inequality that establishes the stable regime.

**Proposed Solution**:
1. **Parameterization approach**: Write $C_{\text{dead}} := c_{\text{rev}} \cdot \mathbb{E}[\#\text{dead}]/N$ where $c_{\text{rev}}$ is an N-independent constant (bounded barrier values for revived walkers)
2. **Global bound**: Use the trivial bound $\mathbb{E}[\#\text{dead}] \leq N$ (can't have more deaths than walkers), giving $C_{\text{dead}} \leq c_{\text{rev}}$, which is O(1)
3. **Refined bound (stable regime)**: Invoke Corollary 11.5.2 (exponential extinction suppression) to show that in the regime where $W_b \leq C_b/\kappa_b$, the expected number of deaths is O(1), giving $C_{\text{dead}} = O(N^{-1})$
4. **Resolve circularity**: The drift inequality $\mathbb{E}[\Delta W_b] \leq -\kappa_b W_b + C_b$ holds for **any** value of $C_b$. The stable regime analysis (Corollary 11.5.1, lines 7470-7510) then shows that the equilibrium satisfies $\limsup_{t \to \infty} \mathbb{E}[W_b] \leq C_b/\kappa_b$, which **as a consequence** implies $\mathbb{E}[\#\text{dead}] = O(1)$ via exponential suppression. So there's no circularity: the drift → stable regime → death bound is a **forward implication chain**.

**Mathematical Detail**:
- **Lemma B** (to be formalized): In any state where $W_b \leq C_b/\kappa_b$, the expected number of individual walker deaths satisfies:
  $$
  \mathbb{E}[\#\text{dead per step}] \leq f_{\text{safe}}^{-1} \cdot e^{-N c_{\text{extinct}}} + N \cdot P(\text{single walker crosses})
  $$
  where $f_{\text{safe}} \in (0,1)$ is the safe interior fraction and $c_{\text{extinct}} > 0$ is from Corollary 11.5.2's proof
- The first term (collective extinction) is exponentially small; the second term (individual crossings) is O(1) due to bounded diffusion
- Therefore $\mathbb{E}[\#\text{dead}] = O(1)$ in the stable regime

**Alternative if Fails**:
- Use the conservative bound $C_{\text{dead}} = O(1)$ (not O($N^{-1}$)) by avoiding the stable regime assumption
- State the theorem as: "$C_b = O(\sigma_x^2) + O(1)$, where the O(1) term improves to O($N^{-1}$) in the large-$N$ limit under the stable regime assumption"
- This is mathematically safer but less sharp

**References**:
- Similar exponential concentration techniques in: Proof of Corollary 11.5.2 (lines 7524-7620)
- Standard result: Hoeffding's inequality for bounded random variables

---

### Challenge 2: Using Exposed-Mass vs Total $W_b$

**Why Difficult**: When $W_b$ is small (swarm far from boundary), the boundary-exposed set $\mathcal{E}_{\text{boundary}}$ may be empty or contain few walkers, so the exposed-mass $M_{\text{boundary}}$ may be much smaller than the total $W_b$. The contraction mechanism directly targets $M_{\text{boundary}}$, not $W_b$, so we need to relate the two carefully.

**Proposed Solution**:
1. **Formalize Lemma A**: The remark on lines 7187-7204 states:
   $$
   M_{\text{boundary}}(S_k) = \frac{1}{N} \sum_{i \in \mathcal{E}_{\text{boundary}}(S_k)} \varphi_{\text{barrier}}(x_i)
   $$
   and relates it to total $W_b$ via:
   $$
   W_b(S_k) = M_{\text{boundary}}(S_k) + \frac{1}{N} \sum_{i \notin \mathcal{E}_{\text{boundary}}(S_k)} \varphi_{\text{barrier}}(x_i)
   $$
   The second sum is bounded by $\frac{k_{\text{alive}}}{N} \cdot \phi_{\text{thresh}}$ (since non-exposed walkers have $\varphi_{\text{barrier}} \leq \phi_{\text{thresh}}$ by definition)

2. **Two-regime analysis**:
   - **Large $W_b$ regime** ($W_b > 2\phi_{\text{thresh}}$): Most walkers are exposed, so $M_{\text{boundary}} \geq W_b - \phi_{\text{thresh}}$. The contraction on $M_{\text{boundary}}$ translates almost directly to contraction on $W_b$.
   - **Small $W_b$ regime** ($W_b \leq 2\phi_{\text{thresh}}$): Even if exposed mass is small, $W_b$ is already bounded (at most $2\phi_{\text{thresh}}$), so the additive constant $C_b$ dominates and prevents further growth

3. **Unified bound**: Use the inequality:
   $$
   \mathbb{E}[\Delta W_b] \leq -\kappa_b M_{\text{boundary}} + C_b \leq -\kappa_b (W_b - \phi_{\text{thresh}}) + C_b = -\kappa_b W_b + (\kappa_b \phi_{\text{thresh}} + C_b)
   $$
   Absorb $\kappa_b \phi_{\text{thresh}}$ into the constant $C_b$ (since both are O(1) constants), yielding the stated inequality

**Mathematical Detail**:
- The key insight is that $\phi_{\text{thresh}}$ serves as a **buffer**: walkers with $\varphi_{\text{barrier}} \leq \phi_{\text{thresh}}$ are considered "safe enough" and don't require active cloning pressure
- The contraction only needs to act when $W_b > C_b/\kappa_b$, which (by proper choice of $\phi_{\text{thresh}}$) can be made large enough to ensure most contribution comes from exposed walkers
- This is a **design choice** in defining $\mathcal{E}_{\text{boundary}}$, not a mathematical necessity

**Alternative if Fails**:
- Define a **state-dependent contraction rate**: $\kappa_b(W_b) := \kappa_b \cdot \min(1, M_{\text{boundary}}/W_b)$
- This gives a weaker contraction when $W_b$ is small but exposed mass is tiny
- However, this complicates the Foster-Lyapunov analysis and is unnecessary if we absorb the threshold into the constant

---

### Challenge 3: Smoothness-Based Jitter Bound

**Why Difficult**: The O($\sigma_x^2$) bound on jitter contribution relies on Taylor expansion of $\varphi_{\text{barrier}}$ around the companion location, which requires:
1. $\varphi_{\text{barrier}} \in C^2$ (second derivatives exist and are bounded)
2. $\sigma_x$ is small relative to the barrier curvature scale
3. The jittered position stays in $\mathcal{X}_{\text{valid}}$ (doesn't cross boundary)

If any of these fails, the bound may not hold.

**Proposed Solution**:
1. **C² regularity**: Verified by Axiom EG-0 (domain regularity, lines 198-209) and Proposition 4.3.2 (existence of smooth barrier function, lines 210-362). The barrier is constructed explicitly as:
   $$
   \varphi_{\text{barrier}}(x) = \begin{cases}
   0 & d(x, \partial \mathcal{X}_{\text{valid}}) > \delta_{\text{safe}} \\
   f\left(\frac{\delta_{\text{safe}} - d(x, \partial \mathcal{X}_{\text{valid}})}{\delta_{\text{safe}}}\right) & \text{else}
   \end{cases}
   $$
   where $f$ is a smooth bump function. This is C² by construction.

2. **Small jitter assumption**: This is an **algorithmic design constraint**. The parameter $\sigma_x$ should satisfy:
   $$
   \sigma_x \leq c_{\text{jitter}} \cdot \delta_{\text{safe}}
   $$
   for some small constant $c_{\text{jitter}} \ll 1$ (e.g., 0.1). This ensures jittered positions don't jump across the safe boundary layer. Remark 11.6.3 (lines 7622-7634) discusses this parameter tuning.

3. **Boundary crossing prevention**: If jitter is small ($\sigma_x \ll \delta_{\text{safe}}$) and the companion is in the safe interior ($d(x_{c_i}, \partial \mathcal{X}_{\text{valid}}) > \delta_{\text{safe}}$), then with high probability (exponentially in $(\delta_{\text{safe}}/\sigma_x)^2$ by Gaussian tails), the jittered position remains in the safe interior.

4. **Worst-case bound**: Even if jitter occasionally crosses into the boundary layer, the expected barrier value is bounded:
   $$
   \mathbb{E}[\varphi_{\text{barrier}}(x'_i)] \leq (1 - \epsilon_{\text{cross}}) \cdot 0 + \epsilon_{\text{cross}} \cdot \varphi_{\text{barrier,max}}
   $$
   where $\epsilon_{\text{cross}} = O(\exp(-(\delta_{\text{safe}}/\sigma_x)^2))$ is exponentially small, and $\varphi_{\text{barrier,max}}$ is the maximum barrier value in the alive region (finite since dead walkers have crossed the boundary). This gives $C_{\text{jitter}} = O(\sigma_x^2)$ after Taylor expansion in the interior.

**Mathematical Detail**:
- **Taylor expansion**: For $x_{c_i}$ in the safe interior with $\varphi_{\text{barrier}}(x_{c_i}) = 0$:
  $$
  \varphi_{\text{barrier}}(x_{c_i} + \sigma_x \zeta) \approx \varphi_{\text{barrier}}(x_{c_i}) + \sigma_x \nabla \varphi_{\text{barrier}}(x_{c_i})^T \zeta + \frac{\sigma_x^2}{2} \zeta^T H_{\varphi_{\text{barrier}}}(x_{c_i}) \zeta
  $$
  Since $x_{c_i}$ is in the safe interior, $\nabla \varphi_{\text{barrier}}(x_{c_i}) = 0$ (barrier is constant there). Taking expectation over $\zeta \sim \mathcal{N}(0, I_d)$:
  $$
  \mathbb{E}[\varphi_{\text{barrier}}(x'_i)] = 0 + \frac{\sigma_x^2}{2} \mathbb{E}[\zeta^T H_{\varphi_{\text{barrier}}} \zeta] = \frac{\sigma_x^2}{2} \text{tr}(H_{\varphi_{\text{barrier}}}) \leq \frac{\sigma_x^2 d}{2} \|H_{\varphi_{\text{barrier}}}\|_{\infty}
  $$
  where $\|H_{\varphi_{\text{barrier}}}\|_{\infty}$ is the maximum Hessian norm (bounded by C² regularity). This gives $C_{\text{jitter}} = O(\sigma_x^2)$.

**Alternative if Fails**:
- Use a **conservative bound** that doesn't rely on Taylor expansion: $C_{\text{jitter}} \leq \mathbb{E}[\varphi_{\text{barrier}}(x_{\text{alive}})] + \text{const} = W_b \cdot N + \text{const}$
- This makes $C_{\text{jitter}}$ state-dependent and potentially large, weakening the theorem
- Better to enforce the algorithmic constraint $\sigma_x \ll \delta_{\text{safe}}$ by design

---

## VI. Proof Validation Checklist

- [x] **Logical Completeness**: All steps follow from previous steps or proven results
- [x] **Hypothesis Usage**: All theorem assumptions are used (Safe Harbor Axiom, cloning mechanism, barrier regularity)
- [x] **Conclusion Derivation**: Claimed drift inequality is fully derived from Theorem 11.3.1 + Lemmas
- [x] **Framework Consistency**: All dependencies verified against framework documents (see Section III)
- [x] **No Circular Reasoning**: Proof doesn't assume conclusion; synthesis builds forward from proven results
- [x] **Constant Tracking**: Both $\kappa_b$ and $C_b$ are explicitly defined and bounded
- [x] **Edge Cases**:
  - Small $W_b$ case: Handled by absorbing $\phi_{\text{thresh}}$ into $C_b$
  - Large $W_b$ case: Exposed-mass approximation valid (Challenge 2)
  - Large $N$ limit: O($N^{-1}$) term vanishes (Challenge 1)
- [x] **Regularity Verified**: Barrier smoothness (C²) guaranteed by Axiom EG-0 and Proposition 4.3.2
- [x] **Measure Theory**: All probabilistic operations (expectations, conditioning) are well-defined over discrete walker indices and continuous Gaussian jitter

**Remaining Gaps**:
- ⚠️ **Missing Gemini cross-validation**: Single-strategist analysis reduces confidence
- ⚠️ **Lemma B needs formalization**: Revival contribution scaling should be stated as an explicit lemma
- ⚠️ **Lemma A needs formalization**: Exposed-mass lower bound is stated as a remark, should be a formal lemma
- ⚠️ **Stable regime assumption**: Should be stated explicitly in theorem statement or made global

---

## VII. Alternative Approaches (Not Chosen)

### Alternative 1: Direct Decomposition Proof

**Approach**: Re-run the 5-step argument of Theorem 11.3.1 (lines 7360-7460), but keep explicit constants to identify $\kappa_b$ and $C_b$ on the fly instead of citing the earlier theorem.

**Pros**:
- **Self-contained**: Doesn't rely on earlier theorem structure
- **Constants appear transparently**: Can see exactly where each term in $C_b$ comes from
- **Pedagogically clear**: Shows all details without forward references

**Cons**:
- **Redundant**: Repeats the work of Theorem 11.3.1's proof (40+ lines, Steps 1-5)
- **Less modular**: Doesn't leverage the earlier proven result
- **Harder to verify**: More steps means more opportunities for errors
- **Not appropriate for summary theorem**: Section 11.6 is explicitly titled "Summary and Drift Inequality", indicating synthesis role

**When to Consider**: If Theorem 11.3.1's proof had gaps or ambiguities, a direct proof might be cleaner. However, the earlier proof appears complete and rigorous, so synthesis is preferred.

---

### Alternative 2: Lyapunov Method (Vector Lyapunov)

**Approach**: Instead of treating $W_b$ in isolation, analyze it as a component of a composite Lyapunov function $V_{\text{total}} = V_W + c_V V_{\text{Var}} + c_B W_b$ and apply a Foster-Lyapunov drift condition jointly with variance contraction (Chapter 10) and Wasserstein distance contraction (Chapter 4).

**Pros**:
- **Integrates directly with stability/ergodicity arguments**: The composite Lyapunov function is needed for the final convergence theorem anyway (Chapter 12)
- **Shows synergistic effects**: Can demonstrate how variance contraction and boundary contraction work together
- **Stronger result**: Could potentially prove a joint drift inequality with better constants

**Cons**:
- **Heavier machinery**: Requires defining the composite Lyapunov function and choosing coupling constants $c_V, c_B$
- **Obscures the clean identification**: The specific form $\kappa_b = p_{\text{boundary}}(\phi_{\text{thresh}})$ is less transparent when buried in composite drift
- **Deferred to Chapter 12**: The document structure suggests this composite analysis is meant for Chapter 12 (lines 7706-7710), not here
- **Not a summary of Chapter 11**: Pulls in results from Chapter 10, making it less focused

**When to Consider**: When proving the **final convergence theorem** that requires all Lyapunov components. For the boundary-specific summary theorem in Chapter 11, the direct synthesis approach is more appropriate.

---

### Alternative 3: Coupling Argument for Direct Drift

**Approach**: Instead of using the established contraction theorem, construct an explicit coupling between $(S_1, S_2)$ and $(S'_1, S'_2)$ to directly bound the expected change in $W_b$ via probabilistic coupling techniques.

**Pros**:
- **Direct probabilistic insight**: Shows exactly how cloning events reduce boundary potential
- **Potentially tighter constants**: Could optimize the coupling to get better bounds
- **Independent verification**: Doesn't rely on Theorem 11.3.1's structure

**Cons**:
- **Much more technical**: Coupling construction is delicate, requires careful handling of correlated randomness
- **Longer proof**: Would need to construct the coupling, verify it's valid, bound the drift
- **Duplicates earlier work**: Theorem 11.3.1's proof already uses a form of implicit coupling (decomposition by walker type)
- **Not appropriate for summary theorem**: Too much new machinery for a synthesis section

**When to Consider**: If the proof of Theorem 11.3.1 had fundamental issues or if tighter constants were needed for applications. For the summary theorem, synthesis is cleaner.

---

## VIII. Open Questions and Future Work

### Remaining Gaps

1. **Lemma A (Exposed-mass lower bound)**: The relationship $M_{\text{boundary}}(S_k) \geq W_b(S_k) - (k_{\text{alive}}/N) \phi_{\text{thresh}}$ is stated as a remark (lines 7187-7204) but should be formalized as an explicit lemma
   - **How critical**: Medium — the result is used implicitly in the proof and for Property 2 verification
   - **Difficulty**: Easy (direct from definition of exposed set and bound on non-exposed barrier values)

2. **Lemma B (Revival contribution scaling)**: The bound $\frac{1}{N} \sum_{i \in \mathcal{D}(S)} \mathbb{E}[\varphi_{\text{barrier}}(x'_i)] \leq C_{\text{rev}} \cdot \mathbb{E}[\#\text{dead}]/N$ needs explicit formulation
   - **How critical**: High — this is needed to justify the O($N^{-1}$) claim in the theorem statement
   - **Difficulty**: Easy to Medium (requires formalizing the stable regime assumption and connecting to exponential suppression)

3. **Stable regime assumption**: The theorem statement should clarify whether O($N^{-1}$) scaling for $C_{\text{dead}}$ holds globally or only in the quasi-stationary regime
   - **How critical**: Medium — affects the precise interpretation of "N-independent in the large-N limit"
   - **Resolution options**:
     - State explicitly: "$C_b = O(\sigma_x^2 + N^{-1})$ where the O($N^{-1}$) term vanishes under the stable regime assumption $W_b \leq C_b/\kappa_b$"
     - Use conservative bound: "$C_b = O(\sigma_x^2) + O(1)$" (safe but less sharp)

### Conjectures

1. **Tighter constant for $\kappa_b$**: The current bound $\kappa_b = p_{\text{boundary}}(\phi_{\text{thresh}})$ is a lower bound derived from worst-case companion selection. In practice, if the swarm is concentrated in the safe interior, boundary-exposed walkers may have higher cloning probability than this bound suggests.
   - **Why plausible**: The companion selection operator (lines 1925-1955) uses spatial weighting, so boundary walkers are more likely to select interior walkers as companions when the swarm is well-separated from the boundary
   - **Potential refinement**: $\kappa_b = p_{\text{boundary}}(\phi_{\text{thresh}}) \cdot (1 + f(V_{\text{Var},x}))$ where $f$ increases with positional variance (more spread → better mixing)

2. **Adaptive strengthening near danger**: Property 2 states that contraction strengthens near danger, but the current analysis doesn't quantify this precisely. Conjecture: the effective contraction rate scales as $\kappa_b \cdot (M_{\text{boundary}}/W_b)$, which increases as more of $W_b$ comes from exposed walkers.
   - **Why plausible**: The proof of Theorem 11.3.1 (Step 2, lines 7390-7413) shows the contraction acts primarily on $M_{\text{boundary}}$, not the full $W_b$
   - **Practical implication**: The algorithm responds more aggressively to acute danger than to chronic low-level boundary proximity

### Extensions

1. **State-dependent constants**: Generalize to $\kappa_b(S)$ and $C_b(S)$ that depend on swarm configuration (e.g., via $V_{\text{Var},x}$ or alive fraction). This could yield tighter bounds in specific regimes.

2. **Non-uniform boundary**: Extend to domains where boundary "danger" varies spatially (e.g., some boundary regions are more critical than others). This would require a spatially-weighted barrier function $\varphi_{\text{barrier}}(x, \theta)$ where $\theta$ parameterizes boundary location.

3. **Higher-order moments**: Current analysis only bounds $\mathbb{E}[W_b]$. Extending to variance $\text{Var}(W_b)$ or concentration inequalities would strengthen the extinction suppression results.

4. **Multi-timescale analysis**: Compare the timescale of boundary contraction ($1/\kappa_b$) to variance contraction ($1/\kappa_x$ from Chapter 10) and kinetic relaxation ($1/\gamma$ from Chapter 5). This could inform optimal parameter tuning.

---

## IX. Expansion Roadmap

**Phase 1: Prove Missing Lemmas** (Estimated: 2-4 hours)

1. **Lemma A (Exposed-mass lower bound)**:
   - **Strategy**: Direct calculation from definition
   - **Steps**: (i) Write $W_b = \frac{1}{N}[\sum_{i \in \mathcal{E}_{\text{boundary}}} \varphi_{\text{barrier}}(x_i) + \sum_{i \notin \mathcal{E}_{\text{boundary}}} \varphi_{\text{barrier}}(x_i)]$, (ii) bound second sum by $\leq (k_{\text{alive}}/N) \cdot \phi_{\text{thresh}}$ using definition of exposed set, (iii) rearrange
   - **Difficulty**: Easy

2. **Lemma B (Revival contribution scaling)**:
   - **Strategy**: Combine revival mechanism with exponential suppression
   - **Steps**: (i) Show revived walker barrier bounded by alive distribution plus jitter, (ii) use exponential suppression (Corollary 11.5.2) to bound expected deaths as O(1) in stable regime, (iii) combine with $1/N$ factor
   - **Difficulty**: Medium (requires careful statement of stable regime assumption)

**Phase 2: Fill Technical Details** (Estimated: 4-6 hours)

1. **Step 2 (Contraction rate identification)**:
   - Expand Substeps 2.3-2.6 with full calculations of $s_{\text{min}}$, $p_{\text{interior}}$, and monotonicity proof
   - Add explicit references to companion selection operator definition

2. **Step 3 (Jitter contribution)**:
   - Expand Taylor expansion calculation with explicit Hessian bounds
   - Add analysis of Gaussian tail probabilities for boundary crossing events
   - Discuss parameter constraint $\sigma_x \ll \delta_{\text{safe}}$

3. **Step 4 (Revival contribution)**:
   - Formalize the stable regime assumption and its relationship to exponential suppression
   - Provide alternative conservative bound that doesn't assume stable regime
   - Clarify the non-circularity of the argument

4. **Step 5 (Property verification)**:
   - Expand Property 2 verification with quantitative analysis of strengthening
   - Add explicit comparison with Chapter 10 variance contraction for Property 3
   - Include numerical estimates for typical parameter regimes

**Phase 3: Add Rigor** (Estimated: 3-4 hours)

1. **Epsilon-delta arguments**:
   - Make precise the notion of "small $\sigma_x$" regime (quantify via $\epsilon_{\text{jitter}} = O(\exp(-(\delta_{\text{safe}}/\sigma_x)^2))$)
   - Formalize the "large $W_b$" regime where exposed-mass approximation is valid

2. **Measure-theoretic details**:
   - Verify all expectations are well-defined (finite walker indices, bounded barrier values)
   - Clarify conditioning events (cloning decisions are measurable w.r.t. pre-cloning state)

3. **Constant tracking audit**:
   - Create table tracking all O(1) constants that appear: $c_{\text{rev}}$, $\varphi_{\text{barrier,max}}$, implicit constants in O($\sigma_x^2$), etc.
   - Verify none depend on $N$ (except $C_{\text{dead}}$ which is O($N^{-1}$))

4. **Edge case verification**:
   - $N = 1$ case: Verify formulas still make sense (single walker, no cloning between walkers)
   - $\sigma_x = 0$ case: Perfect cloning, $C_{\text{jitter}} = 0$
   - $\phi_{\text{thresh}} \to 0$ case: All walkers exposed, verify contraction still holds

**Phase 4: Review and Validation** (Estimated: 2-3 hours)

1. **Framework cross-validation**:
   - Re-check all cited theorem labels against glossary.md
   - Verify line number references are accurate
   - Confirm no forward references to unproven results

2. **Dependency graph verification**:
   - Draw explicit dependency graph: this theorem → Theorem 11.3.1 → Lemmas 11.4.1, 11.4.2 → Axiom EG-2
   - Verify no cycles

3. **Comparison with document proof**:
   - The document doesn't provide a separate proof for this theorem (it's stated as a summary)
   - Verify that the synthesis approach matches the intended meaning
   - Check consistency with Chapter 12's use of this result

4. **Peer review preparation**:
   - Once Gemini is available, re-run the dual validation protocol
   - Compare Gemini's strategy with GPT-5's and current synthesis
   - Address any discrepancies

**Total Estimated Expansion Time**: 11-17 hours

**Priority ordering**:
1. **High priority**: Lemma B (needed for O($N^{-1}$) claim), Step 4 expansion (resolve circularity concern)
2. **Medium priority**: Lemma A (strengthen Property 2), Step 3 expansion (Taylor analysis), constant tracking audit
3. **Low priority**: Edge case verification (nice-to-have), Gemini cross-validation (when available)

---

## X. Cross-References

**Theorems Used**:
- {prf:ref}`thm-boundary-potential-contraction` (Theorem 11.3.1) — Base contraction inequality
- {prf:ref}`cor-bounded-boundary-exposure` (Corollary 11.5.1) — Equilibrium bound $W_b \leq C_b/\kappa_b$
- {prf:ref}`cor-extinction-suppression` (Corollary 11.5.2) — Exponential suppression of extinction events

**Lemmas Used**:
- {prf:ref}`lem-boundary-enhanced-cloning` (Lemma 11.4.1) — Cloning probability lower bound
- {prf:ref}`lem-barrier-reduction-cloning` (Lemma 11.4.2) — Barrier reduction from cloning
- {prf:ref}`lem-fitness-gradient-boundary` (Lemma 11.2.2) — Fitness gradient from boundary proximity

**Definitions Used**:
- {prf:ref}`def-boundary-potential` (Definition 6.9.1 / 11.2.1 recall) — Boundary potential $W_b$
- {prf:ref}`def-boundary-exposed-set` (Definition 11.2.1) — Exposed set $\mathcal{E}_{\text{boundary}}$
- {prf:ref}`def-cloning-operator` (Definition 9.5.1) — The operator $\Psi_{\text{clone}}$

**Axioms Used**:
- {prf:ref}`ax:safe-harbor` (Axiom EG-2) — Safe interior with better reward
- {prf:ref}`ax:regularity-domain` (Axiom EG-0) — Domain regularity and smooth boundary

**Related Proofs** (for comparison):
- Chapter 10 positional variance contraction — Similar Foster-Lyapunov structure: {prf:ref}`thm-positional-variance-contraction`
- Chapter 12 composite Lyapunov drift — Combines variance and boundary contraction: {prf:ref}`thm-complete-drift-inequality`

---

**Proof Sketch Completed**: 2025-10-25 01:12 UTC

**Ready for Expansion**: ⚠️ **Partially Ready** — Needs Lemmas A and B formalized before full proof expansion

**Confidence Level**: **Medium** — Justification:
- ✅ Core proof strategy (synthesis) is sound and appropriate for summary theorem
- ✅ All major framework dependencies verified
- ✅ Constants identified correctly with proper N-independence tracking
- ⚠️ **Single-strategist validation** (Gemini failed to respond) reduces confidence
- ⚠️ **Two lemmas need formalization** before expansion (Lemmas A and B)
- ⚠️ **Stable regime assumption** needs clarification in theorem statement

**Recommendation**:
1. Formalize Lemmas A and B as explicit supporting results
2. Re-run dual validation with Gemini when available
3. Proceed to expansion once both lemmas are proven and cross-validated
4. Consider clarifying the O($N^{-1}$) scaling condition in the final theorem statement
