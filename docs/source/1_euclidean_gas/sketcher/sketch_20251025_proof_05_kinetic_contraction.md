# Proof Sketch for cor-net-velocity-contraction

**Document**: /home/guillem/fragile/docs/source/1_euclidean_gas/05_kinetic_contraction.md
**Theorem**: cor-net-velocity-contraction
**Generated**: 2025-10-25
**Agent**: Proof Sketcher v1.0

---

## I. Theorem Statement

:::{prf:corollary} Net Velocity Variance Contraction for Composed Operator
:label: cor-net-velocity-contraction

From 03_cloning.md, the cloning operator satisfies:

$$
\mathbb{E}_{\text{clone}}[\Delta V_{\text{Var},v}] \leq C_v
$$

Combining with the kinetic dissipation:

$$
\mathbb{E}_{\text{clone} \circ \text{kin}}[\Delta V_{\text{Var},v}] \leq -2\gamma V_{\text{Var},v} \tau + (d\sigma_{\max}^2 \tau + C_v)
$$

**For net contraction, we need:**

$$
2\gamma V_{\text{Var},v} \tau > d\sigma_{\max}^2 \tau + C_v
$$

**This holds when:**

$$
V_{\text{Var},v} > \frac{d\sigma_{\max}^2}{2\gamma} + \frac{C_v}{2\gamma\tau}
$$

**Equilibrium bound:**
At equilibrium where $\mathbb{E}[\Delta V_{\text{Var},v}] = 0$:

$$
V_{\text{Var},v}^{\text{eq}} \approx \frac{d\sigma_{\max}^2}{2\gamma} + \frac{C_v}{2\gamma\tau}
$$

**Interpretation:** The equilibrium velocity variance is determined by the balance between:
- Thermal noise injection ($\sigma_{\max}^2$)
- Friction dissipation ($\gamma$)
- Cloning perturbations ($C_v$)
:::

**Informal Restatement**: When the kinetic operator (Langevin dynamics with friction) and cloning operator are composed together, the net effect on velocity variance can still be contractive if the variance is large enough. The kinetic operator provides linear contraction at rate $2\gamma$, while cloning provides bounded expansion by $C_v$. For sufficiently large velocity variance, the contraction dominates, pulling the system toward an equilibrium variance that balances thermal noise, friction dissipation, and cloning perturbations.

---

## II. Proof Strategy Comparison

### Strategy A: Gemini's Approach

**Status**: ⚠️ GEMINI DID NOT RESPOND

Gemini 2.5 Pro returned an empty response on both attempts. This limits our ability to perform cross-validation.

**Limitations**:
- No cross-validation from second strategist
- Lower confidence in chosen approach
- Recommend re-running sketch when Gemini is available

---

### Strategy B: GPT-5's Approach

**Method**: Lyapunov method with tower property

**Key Steps**:
1. Fix the Lyapunov component and notation (work with $V(S) = V_{\text{Var},v}(S)$)
2. Decompose the composed-step drift using tower property: $\Delta V_{\text{total}} = \Delta V_{\text{kin}} + \Delta V_{\text{clone}}$
3. Apply the kinetic drift inequality from Theorem 5.3
4. Apply cloning bounded expansion from 03_cloning.md
5. Combine bounds and extract contraction region
6. Derive equilibrium (stationary) bound by setting $\mathbb{E}[\Delta V] = 0$

**Strengths**:
- Direct and clean: uses established one-step bounds for each operator
- Tower property rigorously justifies summing drifts across composition
- Lyapunov framework naturally handles contraction regions and equilibrium analysis
- All constants explicitly tracked and shown to be N-uniform
- Algebraic rearrangement is straightforward with no hidden technicalities

**Weaknesses**:
- Composition order ambiguity: document mentions both $\Psi_{\text{kin}} \circ \Psi_{\text{clone}}$ and $\Psi_{\text{clone}} \circ \Psi_{\text{kin}}$ in different places
- The $C_v/(2\gamma\tau)$ term in threshold grows as $\tau \to 0$, which may need interpretation
- Equilibrium bound uses "≈" in corollary but should be rigorous upper bound

**Framework Dependencies**:
- Theorem 5.3 (thm-velocity-variance-contraction-kinetic): kinetic drift bound
- Lemma from 03_cloning.md: bounded velocity variance expansion under cloning
- Tower property for conditional expectations (standard probability)
- Velocity variance definition from 03_cloning.md Definition 3.3.1

---

### Strategy Synthesis: Claude's Recommendation

**Chosen Method**: Lyapunov method with tower property (GPT-5's approach)

**Rationale**:
With only one strategist's response available, I adopt GPT-5's approach with the following justification:
- ✅ **Advantage 1**: Directly uses established framework results (Theorem 5.3 + cloning bound) without reproving them
- ✅ **Advantage 2**: Tower property is standard probability theory—no additional lemmas needed
- ✅ **Advantage 3**: Algebraically clean—all steps are elementary rearrangements
- ✅ **Advantage 4**: Explicitly tracks N-uniformity of all constants
- ⚠ **Trade-off**: Must carefully address composition order ambiguity (see Technical Challenges)

**Sources**:
- Primary approach from: GPT-5's strategy (verified against framework)
- All steps: From GPT-5's strategy (no second opinion available)
- Composition order resolution: Claude's addition (addresses ambiguity GPT-5 flagged)

**Verification Status**:
- ✅ All framework dependencies verified against 03_cloning.md and 05_kinetic_contraction.md
- ✅ No circular reasoning detected
- ✅ All constants verified as state-independent and N-uniform
- ⚠ Composition order needs explicit statement (resolved below)
- ⚠ Gemini cross-validation unavailable (recommend re-running)

---

## III. Framework Dependencies

### Verified Dependencies

**Theorems** (from earlier documents):
| Label | Document | Statement | Used in Step | Verified |
|-------|----------|-----------|--------------|----------|
| thm-velocity-variance-contraction-kinetic | 05_kinetic_contraction.md § 5.3 | $\mathbb{E}_{\text{kin}}[\Delta V_{\text{Var},v}] \leq -2\gamma V_{\text{Var},v} \tau + \sigma_{\max}^2 d \tau$ | Step 3 | ✅ |
| Bounded Velocity Variance Expansion | 03_cloning.md § 10.4 | $\mathbb{E}_{\text{clone}}[\Delta V_{\text{Var},v}] \leq C_v$ | Step 4 | ✅ |

**Definitions**:
| Label | Document | Definition | Used for |
|-------|----------|------------|----------|
| def-velocity-variance-recall | 05_kinetic_contraction.md § 5.2 | $V_{\text{Var},v}(S_1, S_2) = \frac{1}{N}\sum_{k=1,2} \sum_{i \in \mathcal{A}(S_k)} \|\delta_{v,k,i}\|^2$ | Lyapunov component |

**Constants**:
| Symbol | Definition | Value/Bound | Properties |
|--------|------------|-------------|------------|
| $\gamma$ | Friction coefficient | $> 0$ | Framework parameter, N-uniform |
| $\sigma_{\max}^2$ | Maximum eigenvalue of $\Sigma\Sigma^T$ | $> 0$ | Framework parameter, N-uniform |
| $d$ | Spatial dimension | Positive integer | Framework parameter, N-uniform |
| $\tau$ | Time step | $> 0$ | Algorithmic parameter, N-uniform |
| $C_v$ | Cloning expansion bound | $4(\alpha_{\text{restitution}} + 1)^2 V_{\max}^2 + C_{\text{bary}}$ | State-independent, N-uniform |

**Standard Results**:
- Tower property for conditional expectations: $\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X \mid \mathcal{F}]]$
- Linearity of expectation: $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$

### Missing/Uncertain Dependencies

**None Required**: This corollary is a direct algebraic consequence of combining two established bounds.

**Composition Order Clarification Needed**:
- The document mentions $\Psi_{\text{total}} = \Psi_{\text{kin}} \circ \Psi_{\text{clone}}$ in the overview (line 128)
- The corollary explicitly uses $\Psi_{\text{clone}} \circ \Psi_{\text{kin}}$ (Section 5.5, lines 2265-2310)
- Resolution: The proof follows the corollary statement's order ($\text{clone} \circ \text{kin}$)
- Alternative order yields slightly different constant: $(1 - 2\gamma\tau) C_v$ instead of $C_v$

---

## IV. Detailed Proof Sketch

### Overview

The proof is a direct application of the tower property to decompose the drift of the composed operator into the sum of the kinetic drift and the cloning drift. The kinetic operator (Theorem 5.3) provides linear contraction at rate $2\gamma$ but adds thermal noise at rate $d\sigma_{\max}^2$. The cloning operator (03_cloning.md) adds bounded expansion $C_v$. By combining these, we obtain a net contraction when $V_{\text{Var},v}$ is large enough to overcome both noise sources. Setting the total drift to zero gives the equilibrium bound.

The key insight is that the kinetic contraction is **linear** in $V_{\text{Var},v}$, while both expansion terms are **constant**. Therefore, for sufficiently large variance, contraction dominates, creating a drift toward equilibrium.

### Proof Outline (Top-Level)

The proof proceeds in 6 main stages:

1. **Fix Notation**: Establish $V(S) = V_{\text{Var},v}(S)$ as the Lyapunov component of interest
2. **Decompose Drift**: Use tower property to write total drift as sum of kinetic + cloning drifts
3. **Apply Kinetic Bound**: Use Theorem 5.3 to bound kinetic contribution
4. **Apply Cloning Bound**: Use 03_cloning.md result to bound cloning contribution
5. **Extract Contraction Region**: Combine bounds and solve for when net drift is negative
6. **Equilibrium Analysis**: Set total drift to zero to derive equilibrium variance bound

---

### Detailed Step-by-Step Sketch

#### Step 1: Fix the Lyapunov Component and Notation

**Goal**: Establish the precise definition of the velocity variance we're tracking

**Substep 1.1**: Define the Lyapunov function
- **Justification**: Use Definition from 05_kinetic_contraction.md § 5.2 (recall from 03_cloning.md Definition 3.3.1)
- **Why valid**: This is the standard velocity variance component used throughout the framework
- **Expected result**:

$$
V(S) := V_{\text{Var},v}(S) = \frac{1}{N}\sum_{k=1,2} \sum_{i \in \mathcal{A}(S_k)} \|\delta_{v,k,i}\|^2
$$

where $\delta_{v,k,i} = v_{k,i} - \mu_{v,k}$ is the centered velocity

**Substep 1.2**: Establish notation for composition
- **Justification**: Corollary explicitly states $\Psi_{\text{clone}} \circ \Psi_{\text{kin}}$ (lines 2265-2310)
- **Why valid**: Function composition—kinetic operator applied first, then cloning
- **Expected result**:

$$
S \xrightarrow{\Psi_{\text{kin}}} S_{\text{kin}} \xrightarrow{\Psi_{\text{clone}}} S_{\text{final}}
$$

**Conclusion**: We work with $V(S)$ as defined above and composition order $\text{clone} \circ \text{kin}$

**Dependencies**:
- Uses: def-velocity-variance-recall

**Potential Issues**:
- ⚠ Document overview (line 128) mentions opposite order $\Psi_{\text{kin}} \circ \Psi_{\text{clone}}$
- **Resolution**: Follow corollary statement explicitly; note alternative in remarks

---

#### Step 2: Decompose the Composed-Step Drift

**Goal**: Express the total drift as sum of kinetic and cloning contributions

**Substep 2.1**: Write total change as sum of changes
- **Action**: Define $\Delta V_{\text{total}} := V(S_{\text{final}}) - V(S)$ and decompose:

$$
\Delta V_{\text{total}} = [V(S_{\text{kin}}) - V(S)] + [V(S_{\text{final}}) - V(S_{\text{kin}})] =: \Delta V_{\text{kin}} + \Delta V_{\text{clone}}
$$

- **Justification**: Telescoping sum
- **Why valid**: Elementary algebra
- **Expected result**: Additive decomposition of total change

**Substep 2.2**: Apply tower property for expectations
- **Action**: Take expectation of both sides:

$$
\mathbb{E}[\Delta V_{\text{total}}] = \mathbb{E}[\Delta V_{\text{kin}}] + \mathbb{E}[\Delta V_{\text{clone}}]
$$

- **Justification**: Linearity of expectation
- **Why valid**: Standard probability
- **Expected result**: Expectation of total drift equals sum of expectations

**Substep 2.3**: Condition on intermediate state
- **Action**: Write expectation of cloning drift as:

$$
\mathbb{E}[\Delta V_{\text{clone}}] = \mathbb{E}\left[\mathbb{E}[\Delta V_{\text{clone}} \mid S_{\text{kin}}]\right]
$$

- **Justification**: Tower property (law of total expectation)
- **Why valid**: Standard conditional probability
- **Expected result**: Can apply cloning bound conditionally at each $S_{\text{kin}}$

**Conclusion**:

$$
\mathbb{E}_{\text{clone} \circ \text{kin}}[\Delta V_{\text{total}}] = \mathbb{E}_{\text{kin}}[\Delta V_{\text{kin}}] + \mathbb{E}_{\text{kin}}\left[\mathbb{E}_{\text{clone}}[\Delta V_{\text{clone}} \mid S_{\text{kin}}]\right]
$$

**Dependencies**:
- Uses: Tower property, linearity of expectation

**Potential Issues**:
- None—this is standard probability theory

---

#### Step 3: Apply the Kinetic Drift Inequality

**Goal**: Bound the kinetic contribution using Theorem 5.3

**Substep 3.1**: State Theorem 5.3
- **Action**: Recall from 05_kinetic_contraction.md § 5.3 (lines 1975-1995):

$$
\mathbb{E}_{\text{kin}}[\Delta V_{\text{Var},v}] \leq -2\gamma V_{\text{Var},v} \tau + \sigma_{\max}^2 d \tau
$$

- **Justification**: Theorem 5.3 (thm-velocity-variance-contraction-kinetic)
- **Why valid**: Proven in § 5.4 of same document using Itô's lemma and parallel axis theorem
- **Expected result**: Kinetic operator provides linear contraction minus thermal noise injection

**Substep 3.2**: Apply to our decomposition
- **Action**: Substitute into Step 2's decomposition:

$$
\mathbb{E}_{\text{kin}}[\Delta V_{\text{kin}}] \leq -2\gamma V(S) \tau + d\sigma_{\max}^2 \tau
$$

- **Justification**: Direct application of Theorem 5.3
- **Why valid**: Preconditions of Theorem 5.3 hold (Chapter 3 axioms: $\gamma > 0$, $\Sigma$ well-defined)
- **Expected result**: Explicit bound on kinetic contribution

**Conclusion**: Kinetic drift is bounded by linear contraction term minus thermal noise

**Dependencies**:
- Uses: thm-velocity-variance-contraction-kinetic
- Requires: Chapter 3 axioms (friction $\gamma > 0$, diffusion tensor $\Sigma$, time step $\tau$)

**Potential Issues**:
- ⚠ Theorem 5.3 proof may assume small $\tau$ (BAOAB integrator discretization)
- **Resolution**: Assume $\tau$ satisfies discretization requirements from § 5.5 (BAOAB stability)

---

#### Step 4: Apply Cloning Bounded Expansion

**Goal**: Bound the cloning contribution using 03_cloning.md result

**Substep 4.1**: State cloning bound
- **Action**: Recall from 03_cloning.md § 10.4 (lines 6670-6685):

$$
\mathbb{E}_{\text{clone}}[\Delta V_{\text{Var},v}] \leq C_v
$$

where $C_v = 4(\alpha_{\text{restitution}} + 1)^2 V_{\max}^2 + C_{\text{bary}}$ is state-independent

- **Justification**: Theorem "Bounded Velocity Variance Expansion from Cloning" (03_cloning.md)
- **Why valid**: Proven using inelastic collision analysis and barycenter stability
- **Expected result**: Cloning adds at most $C_v$ variance, independent of current state

**Substep 4.2**: Apply uniformity in state
- **Action**: The bound holds for **any** pre-clone state, including $S_{\text{kin}}$:

$$
\mathbb{E}_{\text{clone}}[\Delta V_{\text{clone}} \mid S_{\text{kin}}] \leq C_v
$$

- **Justification**: $C_v$ is explicitly state-independent (03_cloning.md, line 6750)
- **Why valid**: The lemma's proof shows $C_v$ depends only on algorithmic parameters, not on the state
- **Expected result**: Can apply bound conditionally at each $S_{\text{kin}}$

**Substep 4.3**: Take outer expectation
- **Action**: Apply expectation over kinetic randomness:

$$
\mathbb{E}_{\text{kin}}\left[\mathbb{E}_{\text{clone}}[\Delta V_{\text{clone}} \mid S_{\text{kin}}]\right] \leq \mathbb{E}_{\text{kin}}[C_v] = C_v
$$

- **Justification**: $C_v$ is constant, so expectation is identity
- **Why valid**: Linearity of expectation with constants
- **Expected result**: Cloning contribution bounded by $C_v$ in total expectation

**Conclusion**:

$$
\mathbb{E}_{\text{clone} \circ \text{kin}}[\Delta V_{\text{clone}}] \leq C_v
$$

**Dependencies**:
- Uses: Bounded Velocity Variance Expansion (03_cloning.md § 10.4)
- Requires: Collision model with $\alpha_{\text{restitution}}$, velocity bound $V_{\max}$

**Potential Issues**:
- None—state-independence is explicitly proven in 03_cloning.md

---

#### Step 5: Combine Bounds and Extract Contraction Region

**Goal**: Combine Steps 3 and 4 to derive net contraction condition

**Substep 5.1**: Add the two bounds
- **Action**: From Steps 3 and 4:

$$
\mathbb{E}_{\text{clone} \circ \text{kin}}[\Delta V_{\text{total}}] \leq \left(-2\gamma V(S) \tau + d\sigma_{\max}^2 \tau\right) + C_v
$$

- **Justification**: Inequalities sum (Step 2 decomposition)
- **Why valid**: $\mathbb{E}[X] \leq a$ and $\mathbb{E}[Y] \leq b$ implies $\mathbb{E}[X+Y] \leq a+b$
- **Expected result**: Combined drift inequality

**Substep 5.2**: Simplify to canonical form
- **Action**: Rearrange:

$$
\mathbb{E}_{\text{clone} \circ \text{kin}}[\Delta V_{\text{Var},v}] \leq -2\gamma V_{\text{Var},v} \tau + (d\sigma_{\max}^2 \tau + C_v)
$$

- **Justification**: Algebraic rearrangement
- **Why valid**: Elementary algebra
- **Expected result**: **First claim of corollary** ✅

**Substep 5.3**: Solve for net contraction condition
- **Action**: For negative drift, require:

$$
-2\gamma V_{\text{Var},v} \tau + (d\sigma_{\max}^2 \tau + C_v) < 0
$$

Rearrange:

$$
2\gamma V_{\text{Var},v} \tau > d\sigma_{\max}^2 \tau + C_v
$$

- **Justification**: Inequality manipulation
- **Why valid**: Standard algebra
- **Expected result**: **Second claim of corollary** ✅

**Substep 5.4**: Solve for threshold
- **Action**: Divide by $2\gamma\tau$:

$$
V_{\text{Var},v} > \frac{d\sigma_{\max}^2}{2\gamma} + \frac{C_v}{2\gamma\tau}
$$

- **Justification**: Division by positive constant preserves inequality
- **Why valid**: $2\gamma\tau > 0$ by framework axioms
- **Expected result**: **Third claim of corollary** ✅

**Conclusion**: Net contraction occurs when velocity variance exceeds the threshold combining thermal noise equilibrium and cloning perturbation scaled by friction.

**Dependencies**:
- Uses: Steps 3, 4, and 2
- Requires: $\gamma > 0$, $\tau > 0$

**Potential Issues**:
- ⚠ Threshold has $C_v/(2\gamma\tau)$ term that grows as $\tau \to 0$
- **Resolution**: This is physically correct for discrete-time cloning—smaller time steps mean cloning perturbations are relatively larger per-step impact. For continuous-time limit, would need $C_v \sim O(\tau)$.

---

#### Step 6: Equilibrium (Stationary) Bound

**Goal**: Derive equilibrium variance by setting total drift to zero

**Substep 6.1**: Set stationarity condition
- **Action**: At equilibrium/stationarity, the mean drift must vanish:

$$
\mathbb{E}[\Delta V_{\text{Var},v}] = 0
$$

- **Justification**: Definition of stationary distribution
- **Why valid**: If measure is invariant, expected change is zero
- **Expected result**: Equilibrium constraint

**Substep 6.2**: Apply combined drift inequality at equilibrium
- **Action**: From Substep 5.2, at equilibrium:

$$
0 = \mathbb{E}[\Delta V_{\text{Var},v}] \leq -2\gamma \mathbb{E}[V_{\text{Var},v}] \tau + (d\sigma_{\max}^2 \tau + C_v)
$$

- **Justification**: Stationarity + combined drift inequality
- **Why valid**: Expectation is linear
- **Expected result**: Inequality constraint on $\mathbb{E}[V_{\text{Var},v}]$

**Substep 6.3**: Rearrange to solve for equilibrium variance
- **Action**: From $0 \leq -2\gamma \mathbb{E}[V_{\text{Var},v}] \tau + d\sigma_{\max}^2 \tau + C_v$:

$$
2\gamma \mathbb{E}[V_{\text{Var},v}] \tau \leq d\sigma_{\max}^2 \tau + C_v
$$

Divide by $2\gamma\tau$:

$$
\mathbb{E}[V_{\text{Var},v}] \leq \frac{d\sigma_{\max}^2}{2\gamma} + \frac{C_v}{2\gamma\tau}
$$

- **Justification**: Algebraic rearrangement
- **Why valid**: Division by positive constant
- **Expected result**: Upper bound on equilibrium velocity variance

**Substep 6.4**: Interpret as approximate equality
- **Action**: Corollary uses "≈" because:
  - The drift inequality may not be tight
  - At equilibrium, the system fluctuates around this value
  - In practice, the upper bound is often achieved approximately

- **Justification**: Heuristic tightness of bounds when system is in balance
- **Why valid**: For rigorous statement, use ≤; "≈" is interpretive
- **Expected result**: **Fourth claim of corollary** ✅ (with caveat)

$$
V_{\text{Var},v}^{\text{eq}} \approx \frac{d\sigma_{\max}^2}{2\gamma} + \frac{C_v}{2\gamma\tau}
$$

**Rigorous Statement**: Any stationary distribution satisfies:

$$
\mathbb{E}_{\text{stationary}}[V_{\text{Var},v}] \leq \frac{d\sigma_{\max}^2}{2\gamma} + \frac{C_v}{2\gamma\tau}
$$

**Conclusion**: Equilibrium velocity variance is bounded above by the balance point where contraction and expansion terms cancel.

**Dependencies**:
- Uses: Stationarity definition, combined drift from Step 5

**Potential Issues**:
- ⚠ Bound is not necessarily tight (could be strict inequality)
- **Resolution**: State rigorously as upper bound; "≈" indicates expected near-saturation

---

**Q.E.D.** ∎

---

## V. Technical Deep Dives

### Challenge 1: Composition Order Consistency

**Why Difficult**: The document contains references to both operator orders:
- Line 128 (overview): $\Psi_{\text{total}} = \Psi_{\text{kin}} \circ \Psi_{\text{clone}}$ (kinetic after cloning)
- Lines 2265-2310 (this corollary): $\Psi_{\text{clone}} \circ \Psi_{\text{kin}}$ (cloning after kinetic)

These are **different compositions** and yield slightly different bounds.

**Proposed Solution**:

For **clone ∘ kin** (as stated in corollary):
- Kinetic acts first: $S \to S_{\text{kin}}$ with drift $\mathbb{E}[\Delta V_{\text{kin}}] \leq -2\gamma V \tau + d\sigma_{\max}^2 \tau$
- Cloning acts second: $S_{\text{kin}} \to S_{\text{final}}$ with drift $\mathbb{E}[\Delta V_{\text{clone}} \mid S_{\text{kin}}] \leq C_v$
- Combined: $\mathbb{E}[\Delta V_{\text{total}}] \leq -2\gamma V \tau + (d\sigma_{\max}^2 \tau + C_v)$ ✅

For **kin ∘ clone** (alternative order):
- Cloning acts first: $S \to S_{\text{clone}}$ with $V(S_{\text{clone}}) \leq V(S) + C_v$
- Kinetic acts second: $\mathbb{E}_{\text{kin}}[\Delta V \mid S_{\text{clone}}] \leq -2\gamma V(S_{\text{clone}}) \tau + d\sigma_{\max}^2 \tau$
- Using $V(S_{\text{clone}}) \leq V(S) + C_v$:

$$
\mathbb{E}[\Delta V_{\text{total}}] \leq -2\gamma(V(S) + C_v)\tau + d\sigma_{\max}^2 \tau = -2\gamma V \tau + (d\sigma_{\max}^2 - 2\gamma C_v)\tau
$$

Note: This changes the constant term from $+C_v$ to $-2\gamma C_v \tau + d\sigma_{\max}^2 \tau$.

**Resolution for This Proof**:
- Follow the corollary statement exactly: **clone ∘ kin**
- Add a remark noting the alternative order yields different constant
- For small $\tau$, the difference is $O(\tau)$ in the constant term

**Alternative Approach**: If the framework globally commits to kin ∘ clone order, the corollary statement should be adjusted, but the qualitative result (contraction region exists, equilibrium bound exists) remains valid.

---

### Challenge 2: Time-Step Scaling of Threshold

**Why Difficult**: The contraction threshold is:

$$
V_{\text{Var},v} > \frac{d\sigma_{\max}^2}{2\gamma} + \frac{C_v}{2\gamma\tau}
$$

The second term $\frac{C_v}{2\gamma\tau}$ **grows as $\tau \to 0$**, meaning smaller time steps require larger variance to achieve contraction. This seems counterintuitive.

**Proposed Solution**:

**Physical Interpretation**:
- In discrete time, cloning is a one-shot operator that adds $C_v$ variance per step
- The kinetic operator contracts at rate $2\gamma\tau$ per step
- As $\tau \to 0$, each kinetic step removes only $2\gamma V \tau$ variance (small)
- But cloning still adds $C_v$ variance (not scaled by $\tau$)
- Therefore, need larger $V$ to overcome the per-step cloning impact

**Alternative Modeling**:
If the framework instead models cloning as a continuous-time process with intensity $\lambda$ (number of cloning events per unit time), then:
- Per-step cloning impact scales as $C_v(\tau) \sim \lambda \tau \tilde{C}_v$
- The threshold becomes $V_{\text{Var},v} > \frac{d\sigma_{\max}^2}{2\gamma} + \frac{\lambda \tilde{C}_v}{2\gamma}$ (no $1/\tau$ term)

**Resolution for This Proof**:
- The discrete-time convention is **mathematically correct** for per-step cloning
- The $1/\tau$ scaling is a feature, not a bug
- For practical algorithms, $\tau$ is fixed (not taken to limit), so threshold is well-defined
- Document this as expected behavior for discrete-time composition

**Remark**: If continuous-time limit is desired, must reformulate cloning operator to have $\tau$-dependent impact.

---

### Challenge 3: Equilibrium "≈" vs Rigorous Bound

**Why Difficult**: The corollary uses "≈" for equilibrium variance, but mathematical rigor requires precise statements.

**Proposed Solution**:

**Rigorous Statement**:
Any stationary distribution $\pi$ satisfies:

$$
\mathbb{E}_{\pi}[V_{\text{Var},v}] \leq \frac{d\sigma_{\max}^2}{2\gamma} + \frac{C_v}{2\gamma\tau}
$$

This is an **upper bound**, not necessarily an equality.

**Why "≈" is Used**:
1. **Heuristic Tightness**: If the drift inequalities are nearly tight (i.e., the bounds in Theorem 5.3 and cloning expansion are close to actual drifts), then the equilibrium variance will saturate the bound.

2. **Fluctuations**: In any stochastic system, the instantaneous variance fluctuates around the mean. The "≈" acknowledges these fluctuations.

3. **Pedagogical Clarity**: For intuition, thinking of equilibrium as "balance point" where contraction equals expansion is clearer than "upper bound on equilibrium".

**Resolution for This Proof**:
- **Rigorous proof**: Derive the upper bound $\leq$
- **Interpretation**: Note that tight bounds suggest near-saturation, justifying "≈"
- **Final statement**: Use $\leq$ in theorem, use "≈" in interpretation/remarks

**Alternative Approach**: To prove equality (or lower bound), would need:
- Tightness analysis of Theorem 5.3 bound
- Analysis of when cloning expansion saturates $C_v$
- Proof that stationary distribution exists and is unique
- This is beyond the scope of a corollary—would be a separate theorem

---

## VI. Proof Validation Checklist

- [x] **Logical Completeness**: All steps follow from previous steps via tower property and algebra
- [x] **Hypothesis Usage**: All framework results (Theorem 5.3, cloning bound) are cited and applied correctly
- [x] **Conclusion Derivation**: All four claims of the corollary are explicitly derived:
  - Combined drift inequality ✅
  - Net contraction condition ✅
  - Threshold formula ✅
  - Equilibrium bound ✅
- [x] **Framework Consistency**: All dependencies verified against 03_cloning.md and 05_kinetic_contraction.md
- [x] **No Circular Reasoning**: Proof uses only prior results (Theorem 5.3 proven in § 5.4, cloning bound proven in 03_cloning.md § 10.4)
- [x] **Constant Tracking**: All constants ($\gamma$, $\sigma_{\max}^2$, $d$, $\tau$, $C_v$) are explicitly defined and shown to be:
  - State-independent ✅
  - N-uniform ✅
  - Finite ✅
- [x] **Edge Cases**:
  - $\tau \to 0$ behavior: Threshold grows (addressed in Challenge 2) ✅
  - $\gamma \to 0$ behavior: Threshold diverges (no contraction without friction—expected) ✅
  - $C_v = 0$ behavior: Reduces to pure kinetic contraction (correct) ✅
- [x] **Regularity Verified**: No additional smoothness assumptions needed—purely algebraic
- [x] **Measure Theory**: All probabilistic operations (expectations, conditioning) are well-defined for discrete swarms

---

## VII. Alternative Approaches (Not Chosen)

### Alternative 1: Direct Coupling at the Micro-Level

**Approach**: Instead of using existing operator-level bounds, construct a detailed coupling between two swarms evolving under clone ∘ kin and track velocity variance difference directly.

**Pros**:
- Can sometimes yield sharper constants by exploiting cancellations
- Provides micro-level insight into how individual walkers contribute
- Could potentially prove tightness of bounds

**Cons**:
- Significantly more technical—requires reproducing arguments from Theorem 5.3 and 03_cloning.md
- No additional value for this corollary, which is meant to be a direct consequence
- Would obscure the simple algebraic nature of the result
- Duplicates work already done in earlier theorems

**When to Consider**:
- If Theorem 5.3 or cloning bounds were not available
- If seeking tight constants for optimization
- If proving lower bounds or necessity of conditions

---

### Alternative 2: Generator-Based Small-τ Expansion

**Approach**: Derive the combined drift via a Trotter-like expansion of the composed operator's generator, treating composition as $e^{\tau \mathcal{L}_{\text{kin}}} \circ \mathcal{K}_{\text{clone}}$ and expanding in powers of $\tau$.

**Pros**:
- Clarifies $\tau$-scaling systematically via Taylor expansion
- Can track higher-order corrections ($O(\tau^2)$, $O(\tau^3)$, etc.)
- Provides continuous-time limit interpretation
- More elegant from PDE perspective

**Cons**:
- Requires additional regularity assumptions (smoothness of generator action on $V_{\text{Var},v}$)
- Heavy bookkeeping of commutators $[\mathcal{L}_{\text{kin}}, \mathcal{K}_{\text{clone}}]$
- Overkill given that one-step inequalities are already established
- Obscures simplicity: this is an elementary consequence, not a deep PDE result

**When to Consider**:
- If deriving continuous-time limit theorems
- If $\tau$-scaling analysis is critical for numerics
- If higher-order corrections are needed for accuracy

---

### Alternative 3: Direct Lyapunov Equation Solving

**Approach**: Instead of bounding drifts, try to solve the Lyapunov equation $\mathbb{E}[\Delta V] = 0$ directly for the equilibrium distribution and compute $V_{\text{Var},v}^{\text{eq}}$ exactly.

**Pros**:
- Would give exact equilibrium variance (no "≈")
- Could prove existence and uniqueness of stationary distribution
- Provides full characterization, not just bound

**Cons**:
- Requires existence proof for stationary distribution (non-trivial)
- Requires detailed understanding of composed operator's spectrum
- May not have closed-form solution
- Far beyond scope of a corollary—would be a major theorem

**When to Consider**:
- If proving QSD (quasi-stationary distribution) convergence theorems
- If exact equilibrium properties are needed
- If characterizing long-time behavior is the main goal

---

## VIII. Open Questions and Future Work

### Remaining Gaps

1. **Composition Order Ambiguity**: Document should clarify global convention—is it kin ∘ clone or clone ∘ kin?
   - **How critical**: Medium—affects constant terms but not qualitative behavior
   - **Resolution**: Add explicit statement in framework axioms (Chapter 1 or 2)

2. **Tightness of Bounds**: Are the bounds in Theorem 5.3 and cloning expansion tight?
   - **How critical**: Low for this corollary (upper bound sufficient), High for optimization
   - **Resolution**: Separate analysis of when equalities hold (e.g., Gaussian distributions)

3. **Continuous-Time Limit**: How does the $1/\tau$ term in threshold behave as $\tau \to 0$?
   - **How critical**: Medium for understanding algorithmic scaling
   - **Resolution**: Reformulate cloning as continuous-time process with intensity $\lambda$

### Conjectures

1. **Tight Equilibrium**: For Gaussian potentials and isotropic diffusion, the equilibrium bound is saturated (equality holds)
   - **Why plausible**: Gaussian systems often saturate Lyapunov bounds due to quadratic structure
   - **How to test**: Compute equilibrium variance explicitly for harmonic oscillator case

2. **Optimal Composition Order**: The order clone ∘ kin yields better constants than kin ∘ clone for typical parameter regimes
   - **Why plausible**: Cloning first "mixes" velocities, then kinetic dissipation removes excess variance efficiently
   - **How to test**: Compare constants for both orders with realistic parameters

### Extensions

1. **Anisotropic Diffusion**: Extend to position-dependent $\Sigma(x)$ (addressed elsewhere in document § 6)
2. **Adaptive Friction**: Allow $\gamma = \gamma(x, v)$ position/velocity-dependent
3. **Multiple Cloning Steps**: Analyze $\Psi_{\text{kin}} \circ \Psi_{\text{clone}}^k$ (multiple cloning events per kinetic step)
4. **Coupling to Position Variance**: How does velocity variance contraction interact with position variance dynamics (addressed in Chapter 6 hypocoercivity)

---

## IX. Expansion Roadmap

### Phase 1: Rigorous Formalization (Estimated: 2-4 hours)

No additional lemmas needed—this is a direct consequence. Expansion consists of:

1. **Formalize tower property application** (30 min)
   - Write out conditional expectation machinery explicitly
   - Verify measurability of all random variables
   - State Markov property clearly

2. **Verify Theorem 5.3 preconditions** (1 hour)
   - Check that all Chapter 3 axioms are stated in document
   - Verify BAOAB integrator assumptions
   - Confirm $\tau$ regime where discretization is valid

3. **Verify cloning bound preconditions** (1 hour)
   - Check collision model assumptions from 03_cloning.md
   - Verify $\alpha_{\text{restitution}}$, $V_{\max}$, $C_{\text{bary}}$ are well-defined
   - Confirm N-uniformity claims

4. **Add composition order remark** (30 min)
   - Derive alternative bound for kin ∘ clone order
   - Compare constants between two orders
   - Recommend global convention

### Phase 2: Tightness Analysis (Optional) (Estimated: 4-8 hours)

1. **Analyze when Theorem 5.3 bound is tight** (2-3 hours)
   - For what distributions is equality achieved?
   - Construct examples showing near-saturation
   - Derive lower bound on kinetic drift

2. **Analyze when cloning bound is tight** (2-3 hours)
   - When does cloning variance increase by exactly $C_v$?
   - Construct worst-case scenarios
   - Derive lower bound on cloning drift

3. **Prove equilibrium bound saturation** (2-3 hours)
   - Show that for Gaussian case, "≈" becomes "="
   - Compute equilibrium variance explicitly
   - Prove uniqueness of stationary distribution (may require additional analysis)

### Phase 3: Numerical Verification (Optional) (Estimated: 2-3 hours)

1. **Implement combined operator in code** (1 hour)
2. **Run simulations with various parameter regimes** (1 hour)
3. **Compare empirical equilibrium variance with theoretical bound** (1 hour)
4. **Validate $1/\tau$ scaling of threshold** (30 min)

### Phase 4: Integration with Full Framework (Estimated: 1 hour)

1. **Cross-reference with Chapter 6 (hypocoercivity)** (30 min)
   - How does velocity contraction combine with position contraction?
   - Does this corollary contribute to full Lyapunov drift?

2. **Cross-reference with convergence theorems** (30 min)
   - Does equilibrium bound feed into QSD convergence rates?
   - Is this used in mean-field limit theorems?

**Total Estimated Expansion Time**: 3-7 hours (core formalization) + 6-12 hours (optional deep dives)

**Priority**: Phase 1 is essential. Phases 2-4 are valuable but not critical for proof validity.

---

## X. Cross-References

**Theorems Used**:
- {prf:ref}`thm-velocity-variance-contraction-kinetic` (Theorem 5.3, 05_kinetic_contraction.md)
- Bounded Velocity Variance Expansion from Cloning (03_cloning.md § 10.4)

**Definitions Used**:
- {prf:ref}`def-velocity-variance-recall` (Definition 5.2, 05_kinetic_contraction.md, recall from 03_cloning.md)
- Velocity Variance Component (03_cloning.md Definition 3.3.1)

**Related Proofs** (for comparison):
- Theorem 5.3 proof (same document § 5.4) - establishes kinetic contraction via Itô's lemma
- Cloning expansion proof (03_cloning.md § 10.4) - establishes bounded expansion via collision analysis
- Hypocoercivity theorem (Chapter 6) - combines position and velocity Lyapunov components

**Dependent Results** (what uses this corollary):
- Chapter 6 convergence theorems likely use this to show full Lyapunov drift is negative
- Mean-field limit theorems may use equilibrium bound for uniform-in-N estimates

---

## XI. Special Notes

### ⚠️ Gemini Cross-Validation Unavailable

**Issue**: Gemini 2.5 Pro returned empty responses on both query attempts. This proof sketch is based solely on GPT-5's strategy with high reasoning effort.

**Implications**:
- Lower confidence in strategy choice (no second opinion to compare)
- Potential blind spots or alternative approaches may be missed
- Cannot perform cross-validation to detect hallucinations

**Mitigation**:
- GPT-5's strategy is straightforward and uses only established results
- All framework dependencies manually verified against source documents
- Proof is elementary (tower property + algebra)—low risk of deep errors
- Strategy naturally aligns with Lyapunov framework established in earlier chapters

**Recommendation**:
- Re-run proof sketch when Gemini is available to obtain cross-validation
- If Gemini proposes different approach, compare and synthesize
- Current sketch is likely correct but would benefit from independent verification

---

### Composition Order Clarification

**Critical Note**: This corollary explicitly uses $\Psi_{\text{clone}} \circ \Psi_{\text{kin}}$ (cloning after kinetic), as stated in lines 2265-2310. However, the document overview (line 128) mentions $\Psi_{\text{total}} = \Psi_{\text{kin}} \circ \Psi_{\text{clone}}$ (opposite order).

**Resolution in This Proof**: We follow the corollary statement exactly. The alternative order yields a different constant:
- **clone ∘ kin**: Combined drift $\leq -2\gamma V \tau + (d\sigma_{\max}^2 \tau + C_v)$
- **kin ∘ clone**: Combined drift $\leq -2\gamma V \tau + (d\sigma_{\max}^2 \tau - 2\gamma C_v \tau)$

For small $\tau$, the difference is $O(\tau)$. Both orders yield qualitatively similar results (contraction region exists, equilibrium bound exists).

**Recommendation**: Framework should adopt a global convention and state it explicitly in foundational axioms.

---

**Proof Sketch Completed**: 2025-10-25
**Ready for Expansion**: Yes (with caveat about Gemini cross-validation)
**Confidence Level**: High - Proof is elementary combination of established results. All dependencies verified. Only concern is lack of second strategist cross-validation, but GPT-5's strategy is sound and well-justified.
