# Proof Sketch: High-Probability Log-Sobolev Inequality

**Document:** `eigenvalue_gap_complete_proof.md`
**Theorem Label:** `thm-probabilistic-lsi`
**Location:** Line 2247
**Date Created:** 2025-10-24
**Agent:** Proof Sketcher (Autonomous Pipeline)

---

## I. Theorem Statement

:::{prf:theorem} High-Probability Log-Sobolev Inequality (Restatement)
:label: sketch-thm-probabilistic-lsi

Under the assumptions of Theorem {prf:ref}`thm-probabilistic-eigenvalue-gap`, for any $\delta > 0$ there exists $N_0(\delta)$ such that for $N \ge N_0$:

With probability $\ge 1 - \delta$ over $(x, S) \sim \pi_{\text{QSD}}$, the log-Sobolev inequality holds:

$$
\text{Ent}_{\mu_g}[f^2] \le \frac{2C_{\text{LSI}}(\delta)}{\alpha_{\text{LSI}}} \int_{\mathcal{X}} |\nabla f|_g^2 \, d\mu_g
$$

where:

$$
\alpha_{\text{LSI}} \ge \frac{\delta_{\text{mean}}^2}{4C_{\text{BL}} \lambda_{\max}^2}
$$

with $C_{\text{LSI}}(\delta) < \infty$ depending on failure probability $\delta$.
:::

**Context from document:**
- $\mu_g$ is the Gaussian measure associated with metric $g(x,S) = H(x,S) + \epsilon_\Sigma I$
- $H(x,S)$ is the fitness Hessian at position $x$ with swarm configuration $S$
- $\text{Ent}_{\mu_g}[f^2] = \int f^2 \log(f^2/\|f\|_2^2) d\mu_g$ is the relative entropy
- $|\nabla f|_g^2 = \langle \nabla f, g^{-1} \nabla f \rangle$ is the Fisher information in the metric

**Key assumptions inherited from Theorem {prf:ref}`thm-probabilistic-eigenvalue-gap`:**
1. Quantitative Keystone Property
2. Companion decorrelation: $|\text{Cov}(\xi_i, \xi_j)| \le C_{\text{mix}}/N$
3. Foster-Lyapunov stability
4. C^∞ regularity for companion-dependent fitness potential
5. **Multi-Directional Positional Diversity** (UNPROVEN HYPOTHESIS)
6. **Fitness Landscape Curvature Scaling** (UNPROVEN HYPOTHESIS)

---

## II. Proof Strategy Comparison

### Strategy A: Direct Bakry-Émery Γ₂ Calculus
**Approach:** Verify the Bakry-Émery curvature condition $\Gamma_2(f) \geq \rho \Gamma(f)$ directly for the measure $\mu_g$.

**Pros:**
- Most direct path from curvature to LSI
- Optimal constants in many cases
- Standard approach in the literature

**Cons:**
- Requires verifying curvature condition for random metric $g(x,S)$
- Curvature bounds depend on higher derivatives (C⁴ or C⁵ regularity)
- Challenging to control uniformly over swarm configurations

### Strategy B: Brascamp-Lieb Inequality ⟹ Log-Sobolev Inequality (CHOSEN)
**Approach:** Use the proven high-probability bound on the Brascamp-Lieb constant (Corollary {prf:ref}`cor-bl-constant-finite`) and invoke the standard result that BL inequalities imply LSI.

**Pros:**
- Leverages existing result (Corollary {prf:ref}`cor-bl-constant-finite` is already proven)
- Well-established relationship: BL ⟹ LSI (Bobkov-Ledoux 2000)
- Explicit formula relating LSI constant to BL constant and eigenvalue gap
- Failure probabilities compose cleanly (same exp(-c/N) concentration)

**Cons:**
- Indirect route (requires understanding BL-LSI relationship)
- Constants may not be optimal (though explicit)

### Strategy C: Hypocoercivity Extension
**Approach:** Extend the hypocoercive LSI proof from `15_geometric_gas_lsi_proof.md` to handle the probabilistic eigenvalue gap.

**Pros:**
- Builds on existing framework LSI machinery
- Natural connection to kinetic structure

**Cons:**
- Requires extensive modification to handle probabilistic bounds
- Hypocoercivity proofs are already complex for deterministic metrics
- Overkill for this application (we only need LSI, not full hypocoercive structure)

**JUSTIFICATION FOR STRATEGY B:**
Strategy B is optimal because:
1. It directly builds on the proven high-probability BL bound (Corollary {prf:ref}`cor-bl-constant-finite`)
2. The BL ⟹ LSI relationship is a standard result with explicit constants
3. Both theorems share the same failure probability mechanism (exp(-c/N) concentration)
4. The proof is clean, short, and leverages existing infrastructure

---

## III. Detailed Proof Sketch

### Step 1: Establish the Brascamp-Lieb to Log-Sobolev Implication

**Goal:** State the standard result relating BL constant to LSI constant.

**Mathematical Content:**

For a log-concave measure $\mu \propto e^{-V}$ on $\mathbb{R}^d$ with Hessian $H = \nabla^2 V$, the Brascamp-Lieb inequality:

$$
\text{Var}_\mu(f) \le C_{\text{BL}} \int |\nabla f|_g^2 d\mu
$$

implies a log-Sobolev inequality:

$$
\text{Ent}_\mu[f^2] \le 2C_{\text{LSI}} \int |\nabla f|_g^2 d\mu
$$

with explicit relationship:

$$
C_{\text{LSI}} \le C_{\text{BL}} \cdot (1 + \log \text{diam}(\text{supp}(\mu)))
$$

**Standard Reference:** Bobkov & Ledoux (2000), "From Brunn-Minkowski to Brascamp-Lieb and to logarithmic Sobolev inequalities," GAFA Vol. 10, pp. 1028-1052.

**Alternative Formula (Tighter):**

For uniformly log-concave measures with $\nabla^2 V \succeq \lambda I$, the relationship becomes:

$$
\alpha_{\text{LSI}} = \frac{1}{2C_{\text{BL}}}
$$

where $\alpha_{\text{LSI}}$ is the LSI constant in the form $\text{Ent}_\mu[f^2] \le (1/\alpha_{\text{LSI}}) \int |\nabla f|^2 d\mu$.

**Technical Subtlety:** The measure $\mu_g$ is Gaussian with covariance $g^{-1}$. For Gaussian measures, the BL and LSI constants are related through the spectral properties of the covariance. Need to verify whether the general BL ⟹ LSI result applies or if we use the Gaussian-specific formula.

**Expansion Notes for Theorem Prover:**
- State the precise form of BL ⟹ LSI used (general log-concave or Gaussian-specific)
- Cite the appropriate theorem from literature (likely Bobkov-Ledoux or Bakry-Émery)
- Verify all regularity conditions are satisfied (log-concavity, support conditions)
- If using diameter bound, verify $\text{supp}(\mu_g)$ is appropriate (may need localization argument)

### Step 2: Extract High-Probability Bound on BL Constant

**Goal:** Recall the proven result that $C_{\text{BL}}(g)$ is bounded with high probability.

**Mathematical Content:**

By Corollary {prf:ref}`cor-bl-constant-finite` (line 2213, PROVEN):

$$
\mathbb{P}_{(x,S) \sim \pi_{\text{QSD}}}\left(C_{\text{BL}}(g(x,S)) \le \frac{4C_0 \lambda_{\max}^2}{\delta_{\text{mean}}^2}\right) \ge 1 - 2d \cdot \exp\left(-\frac{c}{N}\right)
$$

where:
- $C_0$ is a universal constant from the BL inequality derivation
- $\lambda_{\max} = \lambda_{\max}(g) \le C_{\text{Hess}} + \epsilon_\Sigma$ is uniformly bounded
- $\delta_{\text{mean}} = \min(c_{\text{curv}} \kappa_{\text{fit}} \delta_{\min}^2 / (4L_\phi^2 D_{\max}^2), \epsilon_\Sigma)$

**Key Observation:** The failure probability is $2d \cdot \exp(-c/N)$ for some explicit constant $c > 0$ that depends on the framework parameters.

**Define:**

$$
C_{\text{BL}}^{\max} := \frac{4C_0 \lambda_{\max}^2}{\delta_{\text{mean}}^2}
$$

Then with probability $\ge 1 - 2d \cdot \exp(-c/N)$:

$$
C_{\text{BL}}(g(x,S)) \le C_{\text{BL}}^{\max}
$$

**Expansion Notes for Theorem Prover:**
- Explicitly reference Corollary {prf:ref}`cor-bl-constant-finite` with full citation
- State all parameter dependencies ($C_0, \lambda_{\max}, \delta_{\text{mean}}$)
- Note that $C_{\text{BL}}^{\max}$ is independent of $N$ (depends only on framework parameters)
- Emphasize the exp(-c/N) concentration rate

### Step 3: Choose N₀ to Achieve Desired Failure Probability

**Goal:** Given target failure probability $\delta > 0$, choose $N_0(\delta)$ such that $2d \cdot \exp(-c/N) \le \delta$ for all $N \ge N_0$.

**Mathematical Content:**

We want:

$$
2d \cdot \exp\left(-\frac{c}{N}\right) \le \delta
$$

Solving for $N$:

$$
\exp\left(-\frac{c}{N}\right) \le \frac{\delta}{2d}
$$

$$
-\frac{c}{N} \le \log\left(\frac{\delta}{2d}\right)
$$

$$
\frac{c}{N} \ge -\log\left(\frac{\delta}{2d}\right) = \log\left(\frac{2d}{\delta}\right)
$$

$$
N \ge \frac{c}{\log(2d/\delta)}
$$

**Define:**

$$
N_0(\delta) := \left\lceil \frac{c}{\log(2d/\delta)} \right\rceil
$$

Then for all $N \ge N_0(\delta)$:

$$
\mathbb{P}_{(x,S) \sim \pi_{\text{QSD}}}\left(C_{\text{BL}}(g(x,S)) \le C_{\text{BL}}^{\max}\right) \ge 1 - \delta
$$

**Technical Subtlety:** The constant $c$ in the exponential is determined by the concentration inequality (Theorem {prf:ref}`thm-hessian-concentration`). It has the form:

$$
c = \frac{\delta_{\text{mean}}^2}{32(C_{\text{var}} N C_{\text{Hess}}^2 + \delta_{\text{mean}} C_{\text{Hess}}/6)}
$$

Wait, this creates circular dependence on $N$! Need to extract the $N$-independent part more carefully.

**Correction:** Looking at Theorem {prf:ref}`thm-hessian-concentration`, the concentration has form $\exp(-c'/N)$ where $c'$ is the numerator before dividing by the $O(N)$ variance. The explicit form is:

$$
c' = \frac{\delta_{\text{mean}}^2/32}{C_{\text{var}} C_{\text{Hess}}^2 + \delta_{\text{mean}} C_{\text{Hess}}/(6N)}
$$

For large $N$, the second term is negligible, giving:

$$
c' \approx \frac{\delta_{\text{mean}}^2}{32 C_{\text{var}} C_{\text{Hess}}^2}
$$

**Refined Definition:**

$$
N_0(\delta) := \max\left\{ \left\lceil \frac{c'}{\log(2d/\delta)} \right\rceil, N_{\min} \right\}
$$

where $N_{\min}$ ensures the approximation $c' \approx \delta_{\text{mean}}^2/(32C_{\text{var}} C_{\text{Hess}}^2)$ is valid.

**Expansion Notes for Theorem Prover:**
- Carefully extract the precise form of $c'$ from Theorem {prf:ref}`thm-hessian-concentration`
- Verify the large-$N$ approximation is rigorous (provide explicit error bound)
- State all dependencies of $N_0(\delta)$ on framework parameters
- Note that $N_0(\delta) \to \infty$ as $\delta \to 0$ (expected behavior)

### Step 4: Apply BL ⟹ LSI with High Probability

**Goal:** Combine Steps 1-3 to establish the high-probability LSI.

**Mathematical Content:**

By Steps 1-3, for $N \ge N_0(\delta)$:

With probability $\ge 1 - \delta$ over $(x,S) \sim \pi_{\text{QSD}}$, we have $C_{\text{BL}}(g(x,S)) \le C_{\text{BL}}^{\max}$.

On this high-probability event, Step 1 implies:

$$
\text{Ent}_{\mu_g}[f^2] \le 2C_{\text{LSI}} \int |\nabla f|_g^2 d\mu_g
$$

where:

$$
C_{\text{LSI}} = C_{\text{BL}}^{\max} \cdot \Phi(\text{regularity parameters})
$$

Here $\Phi$ is the explicit function from Step 1 (either $1 + \log \text{diam}$ for general case or constant for Gaussian case).

**Convert to theorem statement form:**

The theorem uses the form:

$$
\text{Ent}_{\mu_g}[f^2] \le \frac{2C_{\text{LSI}}(\delta)}{\alpha_{\text{LSI}}} \int |\nabla f|_g^2 d\mu_g
$$

Comparing, we identify:

$$
\frac{2C_{\text{LSI}}(\delta)}{\alpha_{\text{LSI}}} = 2C_{\text{LSI}}
$$

$$
\Rightarrow \frac{C_{\text{LSI}}(\delta)}{\alpha_{\text{LSI}}} = C_{\text{LSI}} = C_{\text{BL}}^{\max} \cdot \Phi
$$

The theorem also states:

$$
\alpha_{\text{LSI}} \ge \frac{\delta_{\text{mean}}^2}{4C_{\text{BL}} \lambda_{\max}^2}
$$

**Verification:** From the relationship $\alpha_{\text{LSI}} = 1/(2C_{\text{BL}})$ (Gaussian case), we have:

$$
\alpha_{\text{LSI}} = \frac{1}{2C_{\text{BL}}} \ge \frac{1}{2C_{\text{BL}}^{\max}} = \frac{1}{2 \cdot 4C_0 \lambda_{\max}^2/\delta_{\text{mean}}^2} = \frac{\delta_{\text{mean}}^2}{8C_0 \lambda_{\max}^2}
$$

This matches the theorem's claim up to the constant factor difference between $C_{\text{BL}}$ (from inequality) and $C_0$ (dimensional constant). Need to verify $C_0 = 2C_{\text{BL}}$ or adjust.

**Define $C_{\text{LSI}}(\delta)$:**

$$
C_{\text{LSI}}(\delta) := C_{\text{BL}}^{\max} \cdot \Phi(\text{regularity})
$$

where $C_{\text{BL}}^{\max}$ depends on $N_0(\delta)$ through the concentration requirement. However, $C_{\text{BL}}^{\max} = 4C_0 \lambda_{\max}^2/\delta_{\text{mean}}^2$ is actually independent of $N$ and $\delta$.

**Clarification:** The $\delta$-dependence enters through $N_0(\delta)$, not through the constant itself. For $N \ge N_0(\delta)$, the LSI holds with probability $\ge 1 - \delta$ with a **fixed** constant $C_{\text{LSI}}(\delta)$ that can be chosen equal to $C_{\text{BL}}^{\max} \cdot \Phi$.

Actually, re-reading the theorem statement: "$C_{\text{LSI}}(\delta) < \infty$ depending on failure probability $\delta$" suggests the constant itself may depend on $\delta$. This could arise if we want a **uniform** bound over all $(x,S)$ with failure probability $\delta$, which requires a different approach (union bound over covering net).

**Alternative Interpretation (Uniform Case):**

If we want a **uniform** LSI over all positions $x \in \mathcal{X}$ with failure probability $\delta$, we use the covering net argument from Remark {prf:ref}`rem-uniform-gap-caveat`:

- Number of covering balls: $\mathcal{N}(\rho) = (D_{\max}/\rho)^d$
- Union bound gives failure probability: $(D_{\max}/\rho)^d \cdot 2d \cdot \exp(-c'/N)$
- To achieve failure probability $\le \delta$, we need:

$$
(D_{\max}/\rho)^d \cdot 2d \cdot \exp(-c'/N) \le \delta
$$

$$
N \ge \frac{c'}{\log((D_{\max}/\rho)^d \cdot 2d/\delta)}
$$

This introduces $\rho$-dependence, and the LSI constant may depend on the localization scale, giving $C_{\text{LSI}}(\delta, \rho)$.

**Expansion Notes for Theorem Prover:**
- Clarify whether the theorem requires pointwise or uniform LSI
- If pointwise: $C_{\text{LSI}}(\delta)$ is independent of $\delta$, dependence is only through $N_0(\delta)$
- If uniform: Use covering net argument and derive $N_0(\delta, \rho)$ and $C_{\text{LSI}}(\delta, \rho)$
- Verify the relationship between $\alpha_{\text{LSI}}$ and $C_{\text{BL}}$ constants
- Reconcile $C_0$ vs $C_{\text{BL}}$ factor (may require checking BL inequality proof)

### Step 5: Verify All Technical Conditions

**Goal:** Ensure all hypotheses of the BL ⟹ LSI result are satisfied.

**Conditions to Verify:**

1. **Log-concavity of $\mu_g$:**
   - $\mu_g$ is Gaussian with covariance $g^{-1}$
   - Gaussians are log-concave ✓

2. **Regularity of potential:**
   - For Gaussian $\mu_g \propto \exp(-\frac{1}{2}\langle x, g x \rangle)$, the "potential" is $V(x) = \frac{1}{2}\langle x, g x \rangle$
   - $\nabla^2 V = g$ is C^∞ by assumption (Theorem {prf:ref}`thm-main-complete-cinf-geometric-gas-full`)
   - Sufficient regularity ✓

3. **Bounded support or growth conditions:**
   - Gaussian measures have unbounded support
   - BL ⟹ LSI for Gaussians uses the explicit spectral relationship, not diameter bounds
   - No support constraint needed ✓

4. **Uniform ellipticity:**
   - $g = H + \epsilon_\Sigma I \succeq \epsilon_\Sigma I$ is uniformly elliptic
   - Required for non-degenerate LSI ✓

5. **Concentration of $g(x,S)$:**
   - This is the content of Theorem {prf:ref}`thm-probabilistic-eigenvalue-gap` and Corollary {prf:ref}`cor-bl-constant-finite`
   - Verified by assumption ✓

**Expansion Notes for Theorem Prover:**
- Explicitly verify each condition of the BL ⟹ LSI theorem from the literature
- For Gaussian case, cite the specific form of BL ⟹ LSI (likely Bakry-Émery rather than Bobkov-Ledoux)
- Note any additional technical conditions (integrability, moment bounds, etc.)
- Cross-reference to C^∞ regularity theorem for the metric

---

## IV. Technical Subtleties

### Subtlety 1: Pointwise vs Uniform LSI

**Issue:** The theorem statement is ambiguous about whether the LSI holds:
- **(A) Pointwise:** For each fixed $x$, over randomness in $S$
- **(B) Uniform:** Over all $x \in \mathcal{X}$ simultaneously

**Resolution:**
- Corollary {prf:ref}`cor-bl-constant-finite` gives pointwise concentration at each $x$
- For uniform result, use covering net argument (Remark {prf:ref}`rem-uniform-gap-caveat`)
- Determine from theorem statement context which is intended
- Most likely interpretation: Pointwise, matching the structure of parent theorems

### Subtlety 2: Relation Between $C_0$ and $C_{\text{BL}}$

**Issue:** The BL inequality has form:

$$
\text{Var}_\mu(f) \le C_{\text{BL}} \int |\nabla f|_g^2 d\mu
$$

But Corollary {prf:ref}`cor-bl-constant-finite` bounds $C_{\text{BL}}(g) \le C_0 \lambda_{\max}^2/\delta_{\text{mean}}^2$ where $C_0$ is described as a "universal constant from BL inequality."

**Question:** Is $C_0$ the same as $C_{\text{BL}}$, or is there a separate dimensional factor?

**Resolution Strategy:**
- Review the proof of BL inequality (likely in earlier sections or referenced document)
- The standard Brascamp-Lieb result for Gaussian $\mathcal{N}(0, \Sigma)$ gives $C_{\text{BL}} = \lambda_{\max}(\Sigma)$
- For $\mu_g \propto \exp(-\frac{1}{2}\langle x, gx\rangle)$, covariance is $\Sigma = g^{-1}$, so $C_{\text{BL}} = \lambda_{\max}(g^{-1}) = 1/\lambda_{\min}(g)$
- The bound $C_{\text{BL}} \le C_0 \lambda_{\max}^2/\delta_{\text{mean}}^2$ might arise from controlling $1/\lambda_{\min}(g)$ via the eigenvalue gap
- Need to verify: $\lambda_{\min}(g) \ge \delta_{\text{mean}}/(2C_0 \lambda_{\max})$ or similar

**Expansion Task:** Trace back the definition of $C_0$ in the BL inequality derivation.

### Subtlety 3: Explicit Form of LSI Constant

**Issue:** The theorem states:

$$
\alpha_{\text{LSI}} \ge \frac{\delta_{\text{mean}}^2}{4C_{\text{BL}} \lambda_{\max}^2}
$$

But from the Gaussian BL ⟹ LSI relationship, we expect:

$$
\alpha_{\text{LSI}} = \frac{1}{2C_{\text{BL}}}
$$

**Reconciliation:**

Substituting $C_{\text{BL}} \le C_0 \lambda_{\max}^2/\delta_{\text{mean}}^2$:

$$
\alpha_{\text{LSI}} = \frac{1}{2C_{\text{BL}}} \ge \frac{1}{2 \cdot C_0 \lambda_{\max}^2/\delta_{\text{mean}}^2} = \frac{\delta_{\text{mean}}^2}{2C_0 \lambda_{\max}^2}
$$

The theorem claims lower bound $\delta_{\text{mean}}^2/(4C_{\text{BL}} \lambda_{\max}^2)$. Let's check:

$$
\frac{\delta_{\text{mean}}^2}{4C_{\text{BL}} \lambda_{\max}^2} \stackrel{?}{=} \frac{\delta_{\text{mean}}^2}{4 \cdot (C_0 \lambda_{\max}^2/\delta_{\text{mean}}^2) \cdot \lambda_{\max}^2} = \frac{\delta_{\text{mean}}^4}{4C_0 \lambda_{\max}^4}
$$

This doesn't match! Possible issues:
- Typo in theorem statement
- Different convention for $C_{\text{BL}}$ (variance bound vs Poincaré constant)
- Missing factors in derivation

**Resolution Required:** Carefully check the dimensional analysis and conventions.

### Subtlety 4: Dependence of $C_{\text{LSI}}(\delta)$ on $\delta$

**Issue:** The theorem notation suggests $C_{\text{LSI}}(\delta)$ depends on $\delta$, but in the standard setup, the LSI constant should be independent of $\delta$.

**Explanation:**
- The high-probability event is "the eigenvalue gap is at least $\delta_{\text{mean}}/2$"
- On this event, $C_{\text{BL}}$ is bounded by a fixed constant
- Therefore, $C_{\text{LSI}}$ is also fixed
- The $\delta$-dependence is only through $N_0(\delta)$ (how large $N$ must be to achieve failure probability $\delta$)

**Clarification:** $C_{\text{LSI}}(\delta)$ is better written as $C_{\text{LSI}}$ (constant) with the statement "$\exists N_0(\delta)$ such that for $N \ge N_0(\delta)$, with probability $\ge 1-\delta$, LSI holds with constant $C_{\text{LSI}}$."

Alternatively, if we want a **worst-case** bound that accounts for the $\delta$-probability failure event:
- On the $(1-\delta)$-probability good event: $C_{\text{LSI}} \le C_{\text{LSI}}^{\text{good}}$ (bounded)
- On the $\delta$-probability bad event: $C_{\text{LSI}}$ could be arbitrarily bad
- We can define $C_{\text{LSI}}(\delta) = C_{\text{LSI}}^{\text{good}}$ and say "the inequality holds with this constant with probability $\ge 1-\delta$"

**Expansion Task:** Clarify the precise meaning of $C_{\text{LSI}}(\delta)$ in the theorem statement.

### Subtlety 5: Conditional Nature of Result

**Critical Note:** The theorem inherits the **conditional status** from Theorem {prf:ref}`thm-probabilistic-eigenvalue-gap`, which depends on two **UNPROVEN HYPOTHESES**:
1. Multi-Directional Positional Diversity (Assumption {prf:ref}`assump-multi-directional-spread`)
2. Fitness Landscape Curvature Scaling (Assumption {prf:ref}`assump-curvature-variance`)

**Implication:** The LSI result is a **conditional theorem**: IF these hypotheses hold, THEN the LSI follows.

**Expansion Task:** Clearly state the conditional nature and reference the assumptions. Include appropriate warnings matching those in the parent theorem.

---

## V. Expansion Roadmap for Theorem Prover

The Theorem Prover should expand this sketch into a complete rigorous proof following this structure:

### Phase 1: Literature Review and Setup (30-45 min)

**Task 1.1:** State the precise BL ⟹ LSI theorem from the literature.
- **Source:** Bobkov & Ledoux (2000) OR Bakry & Émery (1985) for Gaussian case
- **Action:** Write out the exact theorem statement with all hypotheses
- **Note:** Determine if we use general log-concave result or Gaussian-specific formula

**Task 1.2:** Verify the explicit relationship between constants.
- **Source:** Literature + dimensional analysis
- **Action:** Derive the formula $\alpha_{\text{LSI}} = f(C_{\text{BL}}, \lambda_{\max}, \delta_{\text{mean}})$
- **Verify:** Reconcile with theorem statement's claimed bound

**Task 1.3:** Clarify $C_0$ vs $C_{\text{BL}}$ relationship.
- **Source:** Corollary {prf:ref}`cor-bl-constant-finite` proof
- **Action:** Trace back the definition and verify dimensional correctness

### Phase 2: High-Probability Argument (30 min)

**Task 2.1:** State Corollary {prf:ref}`cor-bl-constant-finite` in full.
- **Action:** Quote the complete statement with all hypotheses and constants
- **Verify:** All framework parameters are defined and bounded

**Task 2.2:** Derive $N_0(\delta)$ explicitly.
- **Source:** Concentration bound from Theorem {prf:ref}`thm-hessian-concentration`
- **Action:** Extract the precise form of $c'$ in $\exp(-c'/N)$
- **Compute:** Solve $\exp(-c'/N) \le \delta/(2d)$ for $N$
- **Define:** $N_0(\delta)$ with explicit formula

**Task 2.3:** Verify large-$N$ approximations are rigorous.
- **Action:** Provide explicit error bounds for any approximations in $c'$
- **Ensure:** No circular dependencies on $N$

### Phase 3: Application of BL ⟹ LSI (30 min)

**Task 3.1:** State the high-probability event precisely.
- **Define:** $\mathcal{E} = \{(x,S) : C_{\text{BL}}(g(x,S)) \le C_{\text{BL}}^{\max}\}$
- **Bound:** $\mathbb{P}(\mathcal{E}^c) \le \delta$ for $N \ge N_0(\delta)$

**Task 3.2:** Apply BL ⟹ LSI on the event $\mathcal{E}$.
- **Action:** Invoke the theorem from Task 1.1
- **Verify:** All hypotheses are satisfied on $\mathcal{E}$
- **Conclude:** LSI holds with constant $C_{\text{LSI}} = f(C_{\text{BL}}^{\max})$

**Task 3.3:** Express in theorem statement form.
- **Action:** Rewrite as "with probability $\ge 1-\delta$, the LSI holds"
- **Define:** $C_{\text{LSI}}(\delta)$ precisely (clarify dependence on $\delta$)
- **Verify:** Matches the theorem statement exactly

### Phase 4: Technical Verifications (20-30 min)

**Task 4.1:** Verify log-concavity and regularity conditions.
- **Action:** Check all hypotheses of BL ⟹ LSI theorem
- **Reference:** C^∞ regularity (Theorem {prf:ref}`thm-main-complete-cinf-geometric-gas-full`)

**Task 4.2:** Verify uniform ellipticity.
- **Action:** Ensure $g \succeq \epsilon_\Sigma I > 0$
- **Purpose:** Guarantees non-degenerate LSI

**Task 4.3:** Address pointwise vs uniform question.
- **Decision:** Determine from context whether pointwise or uniform LSI is intended
- **Action:** If uniform needed, add covering net argument
- **Modify:** $N_0(\delta)$ and $C_{\text{LSI}}(\delta, \rho)$ accordingly

### Phase 5: Final Assembly and Conditional Status (20 min)

**Task 5.1:** Write the complete proof.
- **Structure:** Introduction → Setup (Tasks 1.x) → High-probability bound (Tasks 2.x) → LSI application (Tasks 3.x) → Verifications (Tasks 4.x) → Conclusion
- **Format:** Use {prf:proof} directive
- **Length:** Target 1.5-2 pages (expanded from sketch)

**Task 5.2:** Add conditional status warning.
- **Action:** Include {warning} block stating the result is conditional
- **Reference:** The two unproven hypotheses from parent theorem
- **Mirror:** Structure from Theorem {prf:ref}`thm-probabilistic-eigenvalue-gap`

**Task 5.3:** Add remark on optimality and improvements.
- **Action:** Include {prf:remark} discussing:
  - Whether constants are optimal
  - Alternative proofs (Bakry-Émery direct, hypocoercivity)
  - Connection to mean-field limit
  - Implications for KL-convergence

### Phase 6: Cross-References and Integration (10 min)

**Task 6.1:** Add forward references.
- **Action:** Note where this LSI is used in subsequent results
- **Example:** Section on KL-convergence, entropy production, etc.

**Task 6.2:** Verify all labels and cross-references.
- **Check:** All {prf:ref} citations are correct
- **Ensure:** Label `thm-probabilistic-lsi` matches the theorem

**Task 6.3:** Update glossary entry.
- **Action:** Ensure `docs/glossary.md` has an entry for this theorem
- **Include:** Type, label, tags, concise description

---

## VI. Estimated Expansion Time

**Total Estimated Time:** 2.5 - 3.5 hours

**Breakdown:**
- Phase 1 (Literature & Setup): 30-45 min
- Phase 2 (High-Probability Argument): 30 min
- Phase 3 (BL ⟹ LSI Application): 30 min
- Phase 4 (Technical Verifications): 20-30 min
- Phase 5 (Final Assembly): 20 min
- Phase 6 (Cross-References): 10 min

**Confidence:** High

**Rationale:**
- The proof is relatively short (builds on existing result)
- Main work is literature lookup and constant reconciliation
- No new technical lemmas required
- All dependencies are already proven (modulo the two framework hypotheses)

**Potential Complications:**
1. **BL ⟹ LSI constant relationship:** If the literature formula doesn't match the theorem statement, may need to re-derive or adjust (add 30-60 min)
2. **Uniform vs pointwise clarification:** If uniform version is needed, covering net argument adds complexity (add 30 min)
3. **$C_0$ constant mystery:** If $C_0$ definition is unclear, may need to trace through BL proof (add 20-40 min)

**Mitigation:**
- Address Subtleties 2-3 first to identify potential issues early
- If major discrepancy found, consult with user before proceeding
- Keep Gemini/Codex dual review focused on constant verification

---

## VII. Notes for Dual Review (Gemini + Codex)

When this sketch is expanded to a full proof by the Theorem Prover, the dual reviewers should focus on:

**For Gemini:**
1. Verify the BL ⟹ LSI relationship is correctly cited and applied
2. Check dimensional analysis of all constants ($C_0, C_{\text{BL}}, \alpha_{\text{LSI}}$)
3. Ensure the high-probability argument is rigorous (no gaps in $N_0(\delta)$ derivation)
4. Confirm all hypotheses of the literature theorem are satisfied
5. Verify the conditional status warning is appropriate and complete

**For Codex:**
1. Double-check the arithmetic in solving for $N_0(\delta)$
2. Verify the exponential concentration bounds are applied correctly
3. Ensure consistency with parent theorems (notation, conventions, constants)
4. Check that the LSI constant formula matches the theorem statement exactly
5. Identify any missing technical conditions or regularity requirements

**Discrepancy Protocol:**
- If reviewers disagree on constant relationships, verify against source documents
- If one reviewer identifies a gap in BL ⟹ LSI application, cross-check the literature
- For subtleties 2-3 (constant factors), prioritize explicit calculation over citation

---

## VIII. Open Questions for User (Optional)

Before expansion, the user may want to clarify:

1. **Pointwise vs Uniform:** Is the LSI intended to hold pointwise (for each $x$ over $S$) or uniformly (over all $x,S$ simultaneously)?

2. **Constant Convention:** Is $C_{\text{LSI}}(\delta)$ truly $\delta$-dependent, or is the notation indicating "the constant that makes the inequality hold with probability $1-\delta$"?

3. **Literal vs Implicit:** Should the proof cite a specific theorem from Bobkov-Ledoux (2000) or Bakry-Émery (1985), or is the BL ⟹ LSI relationship considered "folklore"?

4. **Optimality:** Is there interest in optimal constants, or is existence of finite constants sufficient?

**Recommendation:** Proceed with expansion assuming:
- Pointwise interpretation (matching parent theorem structure)
- $C_{\text{LSI}}(\delta)$ is actually constant (dependence only via $N_0(\delta)$)
- Cite Bakry-Émery for Gaussian case (more direct than Bobkov-Ledoux)
- Finite constants sufficient (optimality as remark)

User can override these assumptions if needed.

---

**END OF PROOF SKETCH**
