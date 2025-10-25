# Proof Sketch: Synergistic Rate Derivation from Component Drifts

**Theorem Label:** thm-synergistic-rate-derivation
**Document:** docs/source/1_euclidean_gas/06_convergence.md
**Type:** Theorem
**Created:** 2025-10-25 09:31:20

---

## 1. Theorem Statement

:::{prf:theorem} Synergistic Rate Derivation from Component Drifts
:label: thm-synergistic-rate-derivation

The total drift inequality combines component-wise drift bounds from cloning and kinetic operators to yield explicit synergistic convergence.

**Component Drift Structure:**

From the cloning operator {prf:ref}`thm-positional-variance-contraction` and kinetic operator {prf:ref}`thm-velocity-variance-contraction`, each Lyapunov component satisfies:

$$
\begin{aligned}
\mathbb{E}_{\text{clone}}[\Delta V_{\text{Var},x}] &\leq -\kappa_x V_{\text{Var},x} + C_x + C_{xv} V_{\text{Var},v} + C_{xW} V_W \\
\mathbb{E}_{\text{kin}}[\Delta V_{\text{Var},v}] &\leq -\kappa_v V_{\text{Var},v} + C_v + C_{vx} V_{\text{Var},x} \\
\mathbb{E}_{\text{clone}}[\Delta V_W] &\leq -\kappa_W V_W + C_W \\
\mathbb{E}_{\text{clone}}[\Delta W_b] &\leq -\kappa_b W_b + C_b
\end{aligned}
$$

where cross-component coupling terms $C_{xv}, C_{xW}, C_{vx}$ arise from expansion by the complementary operator.

**Weighted Combination:**

Define the weighted Lyapunov function:

$$
V_{\text{total}} = V_{\text{Var},x} + \alpha_v V_{\text{Var},v} + \alpha_W V_W + \alpha_b W_b
$$

Taking expectations over a full step (kinetic + cloning):

$$
\begin{aligned}
\mathbb{E}[\Delta V_{\text{total}}] &= \mathbb{E}[\Delta V_{\text{Var},x}] + \alpha_v \mathbb{E}[\Delta V_{\text{Var},v}] + \alpha_W \mathbb{E}[\Delta V_W] + \alpha_b \mathbb{E}[\Delta W_b] \\
&\leq (-\kappa_x V_{\text{Var},x} + C_x + C_{xv} V_{\text{Var},v} + C_{xW} V_W) \\
&\quad + \alpha_v(-\kappa_v V_{\text{Var},v} + C_v + C_{vx} V_{\text{Var},x}) \\
&\quad + \alpha_W(-\kappa_W V_W + C_W) \\
&\quad + \alpha_b(-\kappa_b W_b + C_b)
\end{aligned}
$$

**Weight Selection for Coupling Domination:**

Choose weights to ensure coupling terms are dominated by contraction:

$$
\alpha_v \geq \frac{C_{xv}}{\kappa_v V_{\text{Var},v}^{\text{eq}}}, \quad
\alpha_W \geq \frac{C_{xW}}{\kappa_W V_W^{\text{eq}}}, \quad
\alpha_v \kappa_v \geq C_{vx} / V_{\text{Var},x}^{\text{eq}}
$$

With these weights, the coupling terms satisfy:

$$
C_{xv} V_{\text{Var},v} - \alpha_v \kappa_v V_{\text{Var},v} \leq -\epsilon_v \alpha_v \kappa_v V_{\text{Var},v}
$$

and similarly for other cross terms, where $\epsilon_v, \epsilon_W \ll 1$ are small positive fractions.

**Synergistic Rate:**

After cancellation of dominated coupling terms:

$$
\mathbb{E}[\Delta V_{\text{total}}] \leq -\kappa_{\text{total}} V_{\text{total}} + C_{\text{total}}
$$

where:

$$
\kappa_{\text{total}} = \min(\kappa_x, \alpha_v \kappa_v, \alpha_W \kappa_W, \alpha_b \kappa_b) \cdot (1 - \epsilon_{\text{coupling}})
$$

$$
C_{\text{total}} = C_x + \alpha_v C_v + \alpha_W C_W + \alpha_b C_b
$$

and $\epsilon_{\text{coupling}} = \max(\epsilon_v, \epsilon_W, \ldots)$ is the residual coupling ratio after weight balancing.

**Physical Interpretation:**

The synergistic rate $\kappa_{\text{total}}$ is determined by:
1. **Bottleneck principle**: The weakest contraction rate dominates (min over components)
2. **Coupling penalty**: $\epsilon_{\text{coupling}}$ reduces the effective rate due to energy transfer between components
3. **Weight balancing**: Optimal $\alpha_i$ maximize $\alpha_i \kappa_i$ subject to coupling domination

When $\epsilon_{\text{coupling}} \ll 1$, the total rate approaches the bottleneck component rate. The equilibrium variance is:

$$
V_{\text{total}}^{\text{QSD}} = \frac{C_{\text{total}}}{\kappa_{\text{total}}}
$$

:::

---

## 2. Dependencies

### 2.1 Required Theorems

**From 03_cloning.md:**
- {prf:ref}`thm-positional-variance-contraction` (Theorem 10.3.1): Establishes positional variance contraction under cloning operator with drift inequality:

$$
\mathbb{E}_{\text{clone}}[\Delta V_{\text{Var},x}] \leq -\kappa_x V_{\text{Var},x} + C_x
$$

where $\kappa_x > 0$ and $C_x < \infty$ are N-uniform.

- {prf:ref}`thm-bounded-velocity-expansion-cloning` (Theorem 10.4.1): Establishes bounded velocity expansion under cloning:

$$
\mathbb{E}_{\text{clone}}[\Delta V_{\text{Var},v}] \leq C_v
$$

**From 05_kinetic_contraction.md:**
- {prf:ref}`thm-velocity-variance-contraction-kinetic` (inferred label): Establishes velocity variance contraction under kinetic operator:

$$
\mathbb{E}_{\text{kin}}[\Delta V_{\text{Var},v}] \leq -\kappa_v V_{\text{Var},v} + C_v'
$$

where $\kappa_v = 2\gamma\tau$ and $C_v' = \sigma_{\max}^2 d \tau$.

- {prf:ref}`thm-positional-variance-bounded-expansion` (Section 05:4.3): Bounded positional expansion from kinetic operator

- {prf:ref}`thm-inter-swarm-contraction-kinetic` (Section 05:2.3): Hypocoercive contraction of inter-swarm error $V_W$

- {prf:ref}`thm-boundary-potential-contraction-kinetic` (Section 05:5.3): Boundary potential contraction via confining potential

### 2.2 Required Definitions

- **Lyapunov components** (Section 6 of 06_convergence.md):
  - $V_{\text{Var},x}$: Positional variance
  - $V_{\text{Var},v}$: Velocity variance
  - $V_W$: Inter-swarm Wasserstein distance
  - $W_b$: Boundary potential

- **Contraction rates**: $\kappa_x, \kappa_v, \kappa_W, \kappa_b$
- **Expansion constants**: $C_x, C_v, C_W, C_b$
- **Coupling terms**: $C_{xv}, C_{xW}, C_{vx}$

### 2.3 Required Concepts

- **Foster-Lyapunov drift condition**: $\mathbb{E}[\Delta V] \leq -\kappa V + C$
- **Operator composition**: Kinetic operator $\Psi_{\text{kin}}$ followed by cloning operator $\Psi_{\text{clone}}$
- **Hypocoercivity**: Different operators contract different components synergistically
- **Equilibrium values**: $V^{\text{eq}}$ defined by $\mathbb{E}[\Delta V] = 0$

---

## 3. Proof Strategy

### 3.1 High-Level Approach

This theorem synthesizes individual component drift bounds into a unified total convergence rate using the **hypocoercive Lyapunov method**. The key insight is that while each operator (cloning or kinetic) may expand some components, the complementary operator contracts them, leading to overall convergence when properly weighted.

**Main Steps:**
1. **Linearity**: Use linearity of expectation to combine individual component drifts
2. **Regrouping**: Collect contraction and expansion terms for each component
3. **Weight selection**: Choose weights $\alpha_i$ to ensure coupling terms are dominated
4. **Algebraic simplification**: Cancel dominated terms to obtain net contraction
5. **Rate extraction**: Identify the bottleneck rate and total constant

### 3.2 Detailed Strategy

**STEP 1: Component Drift Expansion (Algebraic)**

Starting from the weighted Lyapunov function:

$$
V_{\text{total}} = V_{\text{Var},x} + \alpha_v V_{\text{Var},v} + \alpha_W V_W + \alpha_b W_b
$$

Apply linearity of expectation to express:

$$
\mathbb{E}[\Delta V_{\text{total}}] = \mathbb{E}[\Delta V_{\text{Var},x}] + \alpha_v \mathbb{E}[\Delta V_{\text{Var},v}] + \alpha_W \mathbb{E}[\Delta V_W] + \alpha_b \mathbb{E}[\Delta W_b]
$$

Each component receives contributions from both operators:

$$
\mathbb{E}[\Delta V_i] = \mathbb{E}_{\text{kin}}[\Delta V_i] + \mathbb{E}_{\text{clone}}[\Delta V_i]
$$

Substitute the drift bounds from prerequisite theorems to obtain the expanded form.

**STEP 2: Regrouping by Lyapunov Component**

Collect all terms proportional to each $V$ component:

For $V_{\text{Var},x}$:
- Contraction from cloning: $-\kappa_x V_{\text{Var},x}$
- Expansion from kinetic via velocity coupling: $+\alpha_v C_{vx} V_{\text{Var},x}$

For $V_{\text{Var},v}$:
- Contraction from kinetic: $-\alpha_v \kappa_v V_{\text{Var},v}$
- Expansion from cloning: $+C_{xv} V_{\text{Var},v}$

For $V_W$:
- Contraction from kinetic: $-\alpha_W \kappa_W V_W$
- Expansion from cloning: $+C_{xW} V_W$

For $W_b$:
- Contraction from both operators: $-\alpha_b \kappa_b W_b$

**STEP 3: Weight Selection for Domination**

The critical step is choosing weights $\alpha_i$ such that for each component, the contraction dominates the expansion.

For velocity variance, we require:

$$
C_{xv} V_{\text{Var},v} < \alpha_v \kappa_v V_{\text{Var},v}
$$

This motivates the weight selection:

$$
\alpha_v \geq \frac{C_{xv}}{\kappa_v V_{\text{Var},v}^{\text{eq}}}
$$

where $V_{\text{Var},v}^{\text{eq}}$ is the equilibrium value (known from prerequisite theorems).

Similarly for other components. The key is that at equilibrium, the coupling term must be strictly less than the contraction.

**STEP 4: Dominated Term Cancellation**

With weights satisfying the domination conditions, we can write:

$$
C_{xv} V_{\text{Var},v} - \alpha_v \kappa_v V_{\text{Var},v} \leq -\epsilon_v \alpha_v \kappa_v V_{\text{Var},v}
$$

where $\epsilon_v := 1 - \frac{C_{xv}}{\alpha_v \kappa_v V_{\text{Var},v}^{\text{eq}}} > 0$ is the **domination margin**.

After substituting and simplifying, the total drift becomes:

$$
\mathbb{E}[\Delta V_{\text{total}}] \leq -(1 - \epsilon_{\text{coupling}}) \min(\kappa_x, \alpha_v \kappa_v, \alpha_W \kappa_W, \alpha_b \kappa_b) V_{\text{total}} + C_{\text{total}}
$$

**STEP 5: Rate and Constant Identification**

Define:

$$
\kappa_{\text{total}} := \min(\kappa_x, \alpha_v \kappa_v, \alpha_W \kappa_W, \alpha_b \kappa_b) \cdot (1 - \epsilon_{\text{coupling}})
$$

$$
C_{\text{total}} := C_x + \alpha_v C_v + \alpha_W C_W + \alpha_b C_b
$$

This gives the Foster-Lyapunov form:

$$
\mathbb{E}[\Delta V_{\text{total}}] \leq -\kappa_{\text{total}} V_{\text{total}} + C_{\text{total}}
$$

---

## 4. Key Steps (Numbered Outline)

1. **Invoke linearity of expectation** to express total drift as sum of component drifts

2. **Substitute component drift bounds** from prerequisite theorems:
   - Positional variance: cloning contracts, kinetic expands
   - Velocity variance: kinetic contracts, cloning expands
   - Inter-swarm distance: kinetic contracts, cloning bounded
   - Boundary potential: both operators contract

3. **Identify coupling terms** that couple different components:
   - $C_{xv} V_{\text{Var},v}$: velocity variance appears in positional drift
   - $C_{vx} V_{\text{Var},x}$: positional variance appears in velocity drift
   - $C_{xW} V_W$: inter-swarm distance appears in positional drift

4. **Determine equilibrium values** for each component from individual drift equations:
   - $V_{\text{Var},x}^{\text{eq}} = C_x / \kappa_x$
   - $V_{\text{Var},v}^{\text{eq}} = C_v' / \kappa_v$ (from kinetic)
   - $V_W^{\text{eq}} = C_W / \kappa_W$
   - $W_b^{\text{eq}} = C_b / \kappa_b$

5. **Choose weights $\alpha_i$** such that:
   - For each coupled pair, the contraction dominates expansion at equilibrium
   - Weights are chosen as $\alpha_i = \frac{C_{ji}}{\kappa_i V_i^{\text{eq}}} + \delta_i$ with small $\delta_i > 0$

6. **Verify domination** by showing:
   - $C_{xv} V_{\text{Var},v} - \alpha_v \kappa_v V_{\text{Var},v} \leq -\epsilon_v \alpha_v \kappa_v V_{\text{Var},v}$
   - Similar inequalities for other coupling terms
   - $\epsilon_{\text{coupling}} = \max_i \epsilon_i \ll 1$

7. **Regroup all terms** proportional to each Lyapunov component:
   - Net coefficient of $V_{\text{Var},x}$: $-(\kappa_x - \alpha_v C_{vx}/V_{\text{Var},x})$
   - Net coefficient of $V_{\text{Var},v}$: $-(\alpha_v \kappa_v - C_{xv})$
   - Net coefficient of $V_W$: $-(\alpha_W \kappa_W - C_{xW})$
   - Net coefficient of $W_b$: $-\alpha_b \kappa_b$

8. **Extract minimum rate** (bottleneck principle):
   - All net coefficients are positive (by domination condition)
   - The weakest contraction determines overall rate
   - Use $\kappa_{\text{total}} \leq$ minimum net coefficient

9. **Collect constant terms**:
   - $C_{\text{total}} = C_x + \alpha_v C_v + \alpha_W C_W + \alpha_b C_b$
   - This is state-independent and finite

10. **Verify Foster-Lyapunov form**:
    - $\mathbb{E}[\Delta V_{\text{total}}] \leq -\kappa_{\text{total}} V_{\text{total}} + C_{\text{total}}$
    - $\kappa_{\text{total}} > 0$ (synergistic contraction)
    - $C_{\text{total}} < \infty$ (bounded source)

---

## 5. Critical Estimates and Bounds

### 5.1 Required Bounds

**Individual Contraction Rates** (from prerequisite theorems):
- $\kappa_x > 0$: N-uniform positional contraction from cloning (Keystone Principle)
- $\kappa_v = 2\gamma\tau$: Velocity friction dissipation
- $\kappa_W > 0$: Hypocoercive inter-swarm contraction
- $\kappa_b > 0$: Boundary potential contraction

**Individual Expansion Constants** (from prerequisite theorems):
- $C_x < \infty$: N-uniform positional expansion (cloning noise + boundary)
- $C_v < \infty$: Velocity expansion (inelastic collisions)
- $C_v' = \sigma_{\max}^2 d \tau$: Velocity expansion (thermal noise)
- $C_W < \infty$: Inter-swarm expansion (cloning noise)
- $C_b < \infty$: Boundary expansion (thermal kicks)

**Coupling Constants** (to be estimated):
- $C_{xv}$: How velocity variance affects positional drift
- $C_{vx}$: How positional variance affects velocity drift
- $C_{xW}$: How inter-swarm distance affects positional drift

### 5.2 Key Estimates

**Equilibrium Values:**

$$
V_{\text{Var},x}^{\text{eq}} = \frac{C_x}{\kappa_x}, \quad
V_{\text{Var},v}^{\text{eq}} = \frac{d\sigma_{\max}^2}{2\gamma}, \quad
V_W^{\text{eq}} = \frac{C_W}{\kappa_W}, \quad
W_b^{\text{eq}} = \frac{C_b}{\kappa_b}
$$

**Weight Lower Bounds:**

$$
\alpha_v \geq \frac{C_{xv}}{\kappa_v V_{\text{Var},v}^{\text{eq}}} = \frac{2\gamma C_{xv}}{d\sigma_{\max}^2 \kappa_v}
$$

$$
\alpha_W \geq \frac{C_{xW}}{\kappa_W V_W^{\text{eq}}} = \frac{\kappa_W C_{xW}}{C_W}
$$

**Domination Margins:**

$$
\epsilon_v = 1 - \frac{C_{xv}}{\alpha_v \kappa_v V_{\text{Var},v}^{\text{eq}}}
$$

$$
\epsilon_W = 1 - \frac{C_{xW}}{\alpha_W \kappa_W V_W^{\text{eq}}}
$$

**Coupling Penalty:**

$$
\epsilon_{\text{coupling}} = \max(\epsilon_v, \epsilon_W, \ldots) \ll 1
$$

This must be small for synergy to be effective.

### 5.3 Technical Challenges

**Challenge 1: Estimating Coupling Constants $C_{xv}, C_{vx}, C_{xW}$**

These are not given explicitly in prerequisite theorems. We need to:
- Use the proof details of the component drift theorems
- Extract how one component's variance enters the drift bound of another
- Show these are state-independent and finite

**Potential approach:**
- From positional expansion under kinetic operator: displacement $\Delta x \sim v \tau$, so variance grows as $\mathbb{E}[\|v\|^2] \tau^2 \sim V_{\text{Var},v} \tau^2$
- This suggests $C_{xv} \sim \tau^2$ (scale factor)
- Similarly, from velocity drift, force depends on position gradient

**Challenge 2: Verifying Weight Feasibility**

We need to show that weights satisfying all domination conditions simultaneously exist and are positive.

**Approach:**
- Check that the system of weight inequalities is consistent
- Verify no circular dependencies (e.g., $\alpha_v$ needs $\alpha_W$ which needs $\alpha_v$)
- Show explicit construction of valid weights

**Challenge 3: Proving Bottleneck Formula**

After regrouping, we claim the minimum rate controls convergence. This requires:
- Showing all net contraction coefficients are positive
- Proving the minimum dominates (no positive feedback loops)
- Handling the $\epsilon_{\text{coupling}}$ correction carefully

**Approach:**
- Use that $V_{\text{total}}$ is a sum with positive weights
- Each component must individually satisfy $\mathbb{E}[\Delta V_i] \leq -\kappa_i' V_i + C_i'$
- The minimum rate ensures all components are bounded

---

## 6. Potential Difficulties

### 6.1 Coupling Constant Estimation

**Difficulty:** The coupling terms $C_{xv}, C_{xW}, C_{vx}$ are not explicitly stated in prerequisite theorems.

**Resolution Strategy:**
- Review proofs of {prf:ref}`thm-positional-variance-bounded-expansion` to extract how $V_{\text{Var},v}$ enters
- Review proofs of velocity drift theorems to extract how $V_{\text{Var},x}$ enters
- Use dimensional analysis: $[C_{xv}] = \text{length}^2 / (\text{velocity}^2)$ suggests $C_{xv} \sim \tau^2$

**Backup approach:**
- If explicit values unavailable, parametrize as $C_{xv} = O(\tau^2)$, $C_{vx} = O(1)$
- Show qualitatively that domination is achievable for sufficiently small $\tau$

### 6.2 Weight Selection Consistency

**Difficulty:** Multiple constraints on weights may be incompatible.

**Resolution Strategy:**
- Write out all domination conditions as a system of inequalities
- Check for contradictions (e.g., $\alpha_v > A$ and $\alpha_v < B$ with $A > B$)
- If consistent, construct explicit weights (e.g., $\alpha_i = 2C_{ji}/(\kappa_i V_i^{\text{eq}})$)

**Verification:**
- Substitute chosen weights back into drift inequality
- Verify all coupling terms are indeed dominated

### 6.3 Small Coupling Assumption

**Difficulty:** The theorem assumes $\epsilon_{\text{coupling}} \ll 1$, which may not always hold.

**Resolution Strategy:**
- Show that $\epsilon_{\text{coupling}} \to 0$ as $\tau \to 0$ (time step refinement)
- Alternatively, show it's small for typical parameter regimes (e.g., high friction $\gamma$)
- If necessary, state as an assumption on parameter regimes

**Fallback:**
- Even if $\epsilon_{\text{coupling}}$ is not small, as long as $(1 - \epsilon_{\text{coupling}}) > 0$, we still have contraction
- Rate is reduced but convergence is preserved

### 6.4 Minimum Rate Extraction

**Difficulty:** Proving the minimum formula requires careful algebra after regrouping.

**Resolution Strategy:**
- After substituting weights and simplifying, factor out common $V_{\text{total}}$ term
- Use that $V_{\text{total}} = \sum \alpha_i V_i$ with $\alpha_1 = 1$
- Show the coefficient of $V_{\text{total}}$ is at most the minimum of individual coefficients

**Algebraic technique:**
- Use weighted average inequality: $\sum \alpha_i (-\kappa_i' V_i) \leq -\min_i(\kappa_i') \sum \alpha_i V_i$
- Requires all $\kappa_i' > 0$ (ensured by domination)

### 6.5 Physical Interpretation Justification

**Difficulty:** The physical interpretation claims are intuitive but need mathematical backing.

**Resolution Strategy:**
- **Bottleneck principle**: Prove rigorously using weighted average argument
- **Coupling penalty**: Show $\epsilon_{\text{coupling}}$ arises from incomplete cancellation of cross terms
- **Weight balancing**: Formulate as an optimization problem and characterize the solution

**Pedagogical value:**
- Even if full rigor is challenging, the interpretation aids understanding
- Can be stated as a remark rather than part of the proof

---

## 7. Proof Outline Summary

**Structure of Full Proof:**

1. **Setup**: Define weighted Lyapunov function and state prerequisite drift bounds

2. **Drift Expansion**: Use linearity to expand total drift into component contributions

3. **Substitution**: Plug in drift bounds from theorems, separating contraction and expansion

4. **Coupling Extraction**: Identify and estimate all coupling constants $C_{ij}$

5. **Weight Determination**: Solve for weights satisfying domination conditions

6. **Algebraic Simplification**: Regroup terms by Lyapunov component, cancel dominated terms

7. **Rate Extraction**: Identify the minimum rate and prove it controls total drift

8. **Constant Aggregation**: Sum all expansion constants into $C_{\text{total}}$

9. **Foster-Lyapunov Verification**: Confirm the final inequality has the required form

10. **Physical Interpretation**: Relate mathematical result to algorithmic behavior

**Expected Length:** 3-4 pages of detailed algebra with clear step-by-step justification.

**Main Techniques:**
- Linearity of expectation
- Prerequisite theorem invocation
- Algebraic regrouping and factorization
- Weighted average inequalities
- Dimensional analysis

---

## 8. Conclusion

**Proof Viability:** HIGH

This theorem is primarily an **algebraic synthesis** of existing component drift bounds. The prerequisite theorems provide all the necessary contraction and expansion estimates. The main challenge is careful bookkeeping of coupling terms and weight selection, which is technically involved but straightforward.

**Key Success Factors:**
- All component drifts already established in prerequisite theorems
- Linearity of expectation makes combination rigorous
- Weight selection is a solvable system of inequalities
- Foster-Lyapunov framework is well-established

**Potential Issues:**
- Coupling constants may need extraction from prerequisite proofs
- Weight consistency needs verification
- $\epsilon_{\text{coupling}} \ll 1$ may require parameter assumptions

**Recommended Approach:**
1. Start with complete expansion using linearity
2. Carefully track all coupling terms
3. Use dimensional analysis to estimate unknown coupling constants
4. Construct explicit weights satisfying all constraints
5. Perform algebraic simplification step-by-step
6. Verify Foster-Lyapunov form is achieved

**Next Steps:**
- Proceed to full proof development
- Verify coupling constant estimates against prerequisite theorem proofs
- Construct explicit numerical examples for validation
