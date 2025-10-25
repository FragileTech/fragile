# Proof Sketch: Revival Operator is KL-Expansive

**Theorem Label**: `thm-revival-kl-expansive`

**Source Document**: [16_convergence_mean_field.md § 7.1](../16_convergence_mean_field.md)

**Theorem Statement** (from source):

:::{prf:theorem} Revival Operator is KL-Expansive (VERIFIED)
:label: thm-revival-kl-expansive-sketch

The mean-field revival operator $\mathcal{R}[\rho, m_d]$ **increases** the KL-divergence to the invariant measure $\pi$:

$$
\frac{d}{dt} D_{\text{KL}}(\rho \| \pi) \Big|_{\text{revival}} > 0 \quad \text{for all } \rho \neq \pi, \, m_d > 0
$$

where the entropy production has the explicit form:

$$
\frac{d}{dt} D_{\text{KL}}(\rho \| \pi) \Big|_{\text{revival}} = \lambda m_d \left( 1 + \frac{D_{\text{KL}}(\rho \| \pi)}{\|\rho\|_{L^1}} \right)
$$

**Status**: PROVEN (verified by Gemini 2025-01-08)
:::

---

## I. Context and Significance

**Physical Interpretation**: The revival operator models the resurrection of dead walkers by sampling proportionally from the current alive distribution $\rho$. This theorem establishes that revival, when considered in isolation, **increases** the KL-divergence to the quasi-stationary distribution (QSD) $\pi$. This is a critical negative result: it means convergence to the QSD cannot rely on the revival operator being KL-contractive.

**Strategic Implications**:
- This result **refutes** the hypothesis that the mean-field revival operator $\mathcal{R}$ alone is KL-contractive
- It **validates** the kinetic dominance approach: KL-convergence must come from the Langevin diffusion operator overwhelming the revival expansion
- It **completes Stage 0** of the mean-field convergence roadmap by resolving the GO/NO-GO question

**Dependencies**:
- {prf:ref}`def-revival-operator-formal` (Mean-Field Revival Operator definition, 16_convergence_mean_field.md § 1.2)
- Standard KL-divergence properties
- Infinitesimal generator method for entropy production

---

## II. Full Hypotheses

**Hypothesis 1: Revival Operator Well-Defined**

The mean-field revival operator is defined as:

$$
\mathcal{R}[\rho, m_d](x,v) := \lambda_{\text{revive}} \cdot m_d \cdot \frac{\rho(x,v)}{\|\rho\|_{L^1}}
$$

where:
- $\rho \in \mathcal{P}(\Omega)$ is the alive density with $\|\rho\|_{L^1} > 0$
- $m_d \in [0,1]$ is the dead mass
- $\lambda_{\text{revive}} > 0$ is the revival rate parameter
- $\Omega = \mathcal{X}_{\text{valid}} \times \mathbb{R}^d$ is the phase space

**Hypothesis 2: QSD Existence**

There exists a quasi-stationary distribution $\pi \in \mathcal{P}(\Omega)$ satisfying the stationary equation (proved in Stage 0.5 of the source document, {prf:ref}`thm-qsd-existence-corrected`).

**Hypothesis 3: Finite Initial KL-Divergence**

The initial density $\rho$ satisfies $D_{\text{KL}}(\rho \| \pi) < \infty$ and $\rho \neq \pi$ (non-equilibrium).

**Hypothesis 4: Non-Degeneracy**

We assume $m_d > 0$ (some dead mass exists to be revived) and $\lambda_{\text{revive}} > 0$.

---

## III. Proof Strategy Outline

The proof uses the **infinitesimal generator method** for computing entropy production. The strategy is:

### Step 1: Set Up Entropy Production Formula

Start with the definition of entropy production under the revival operator:

$$
\frac{d}{dt} D_{\text{KL}}(\rho \| \pi) \Big|_{\text{revival}} = \int_\Omega \frac{\partial \rho}{\partial t} \Big|_{\text{revival}} \left(1 + \log \frac{\rho}{\pi}\right) \, dx dv
$$

This follows from the standard entropy production identity:

$$
\frac{d}{dt} \int \rho \log \frac{\rho}{\pi} = \int \frac{\partial \rho}{\partial t} \left(1 + \log \frac{\rho}{\pi}\right)
$$

(derivation: chain rule + integration by parts, noting that $\pi$ is stationary)

### Step 2: Substitute Revival Dynamics

Use the revival operator definition:

$$
\frac{\partial \rho}{\partial t} \Big|_{\text{revival}} = \mathcal{R}[\rho, m_d] = \lambda m_d \frac{\rho}{\|\rho\|_{L^1}}
$$

Substitute into the entropy production formula:

$$
\frac{d}{dt} D_{\text{KL}}(\rho \| \pi) \Big|_{\text{revival}} = \int_\Omega \lambda m_d \frac{\rho}{\|\rho\|_{L^1}} \left(1 + \log \frac{\rho}{\pi}\right) \, dx dv
$$

### Step 3: Factor Out Constants

Pull out the constant factors $\lambda m_d / \|\rho\|_{L^1}$:

$$
\frac{d}{dt} D_{\text{KL}}(\rho \| \pi) \Big|_{\text{revival}} = \frac{\lambda m_d}{\|\rho\|_{L^1}} \int_\Omega \rho \left(1 + \log \frac{\rho}{\pi}\right) \, dx dv
$$

### Step 4: Separate Integration Terms

Split the integral into two parts:

$$
\int_\Omega \rho \left(1 + \log \frac{\rho}{\pi}\right) \, dx dv = \int_\Omega \rho \, dx dv + \int_\Omega \rho \log \frac{\rho}{\pi} \, dx dv
$$

Recognize the terms:
- First term: $\int_\Omega \rho \, dx dv = \|\rho\|_{L^1}$ (total alive mass)
- Second term: $\int_\Omega \rho \log \frac{\rho}{\pi} \, dx dv = D_{\text{KL}}(\rho \| \pi)$ (KL-divergence)

### Step 5: Obtain Explicit Formula

Substitute the recognized terms:

$$
\frac{d}{dt} D_{\text{KL}}(\rho \| \pi) \Big|_{\text{revival}} = \frac{\lambda m_d}{\|\rho\|_{L^1}} \left( \|\rho\|_{L^1} + D_{\text{KL}}(\rho \| \pi) \right)
$$

Simplify:

$$
\frac{d}{dt} D_{\text{KL}}(\rho \| \pi) \Big|_{\text{revival}} = \lambda m_d \left( 1 + \frac{D_{\text{KL}}(\rho \| \pi)}{\|\rho\|_{L^1}} \right)
$$

### Step 6: Establish Strict Positivity

Verify that each factor is strictly positive under the hypotheses:
- $\lambda > 0$ (revival rate parameter, positive by definition)
- $m_d > 0$ (dead mass, assumed non-zero by Hypothesis 4)
- $1 > 0$ (trivial)
- $D_{\text{KL}}(\rho \| \pi) \ge 0$ (KL-divergence non-negativity)
- $\|\rho\|_{L^1} > 0$ (alive mass, positive by Hypothesis 1)

Therefore:

$$
\frac{d}{dt} D_{\text{KL}}(\rho \| \pi) \Big|_{\text{revival}} = \lambda m_d \left( 1 + \frac{D_{\text{KL}}(\rho \| \pi)}{\|\rho\|_{L^1}} \right) > \lambda m_d \cdot 1 = \lambda m_d > 0
$$

This proves strict positivity (KL-expansion).

---

## IV. Key Technical Lemmas Needed

### Lemma 1: Entropy Production Identity

**Statement**: For any smooth evolution $\partial_t \rho = F[\rho]$ and reference measure $\pi$:

$$
\frac{d}{dt} D_{\text{KL}}(\rho_t \| \pi) = \int_\Omega F[\rho] \left(1 + \log \frac{\rho}{\pi}\right) \, dx dv
$$

**Justification**: Standard calculus of variations.

**Proof sketch**:
$$
\begin{aligned}
\frac{d}{dt} \int \rho \log \frac{\rho}{\pi} &= \int \frac{\partial \rho}{\partial t} \log \frac{\rho}{\pi} + \int \frac{\partial \rho}{\partial t} \\
&= \int F[\rho] \left( \log \frac{\rho}{\pi} + 1 \right)
\end{aligned}
$$

The second term uses $\partial_t \rho \cdot 1 / \rho = \partial_t \log \rho$ and chain rule.

**Status**: Standard result (see e.g., Otto-Villani "Generalization of an Inequality by Talagrand", Section 2.2)

### Lemma 2: KL-Divergence Non-Negativity

**Statement**: For any probability measures $\rho, \pi$ with $\rho \ll \pi$:

$$
D_{\text{KL}}(\rho \| \pi) := \int \rho \log \frac{\rho}{\pi} \ge 0
$$

with equality if and only if $\rho = \pi$ almost everywhere.

**Justification**: Gibb's inequality / Jensen's inequality applied to the convex function $x \log x$.

**Status**: Standard result (information theory, e.g., Cover & Thomas Theorem 2.6.3)

### Lemma 3: Mass Conservation Under Integration

**Statement**: For $\rho \in L^1(\Omega)$ with $\rho \ge 0$:

$$
\int_\Omega \rho \, dx dv = \|\rho\|_{L^1}
$$

**Justification**: Definition of $L^1$ norm.

**Status**: Trivial (definition)

---

## V. Technical Difficulties and Subtleties

### Difficulty 1: Normalization Singularity (LOW)

**Issue**: The revival operator contains the term $1/\|\rho\|_{L^1}$, which could be singular if $\|\rho\|_{L^1} \to 0$.

**Resolution**: Hypothesis 1 assumes $\|\rho\|_{L^1} > 0$ (positive alive mass). In the mean-field model, this is guaranteed by conservation: $\|\rho\|_{L^1} + m_d = 1$ with $m_d < 1$ (not all mass dead).

**Rigor check**: The proof explicitly requires $\|\rho\|_{L^1} > 0$ as a hypothesis. If $\|\rho\|_{L^1} = 0$, the revival operator is undefined, and the theorem statement does not apply.

### Difficulty 2: Non-Stationarity of $\pi$ Under Revival (MEDIUM)

**Issue**: The proof uses $\pi$ as the reference measure in the KL-divergence, but does not explicitly verify that $\pi$ is stationary under the revival operator alone.

**Resolution**: The theorem computes entropy production for the **revival operator in isolation**, not the full dynamics. The QSD $\pi$ is stationary under the **full generator** $\mathcal{L} = \mathcal{L}_{\text{kin}} + \mathcal{L}_{\text{jump}}$, which includes both killing and revival. The revival operator alone is **not** stationarity-preserving, which is precisely why it's KL-expansive.

**Rigor check**: The proof does not claim $\mathcal{R}[\pi, m_{d,\infty}] = 0$. It correctly computes entropy production with respect to $\pi$ as a **reference measure**, not as a fixed point of $\mathcal{R}$ alone.

### Difficulty 3: Dependence on Dead Mass $m_d$ (LOW)

**Issue**: The dead mass $m_d$ is time-varying in the full dynamics, which could affect the calculation.

**Resolution**: The theorem statement and proof use the **instantaneous** value of $m_d$ at time $t$. The entropy production formula is valid pointwise in time. The time evolution of $m_d$ itself does not enter the calculation (it's an external parameter to $\mathcal{R}$ at each instant).

**Rigor check**: The proof treats $m_d$ as a parameter, not a dynamical variable. This is consistent with the operator-splitting interpretation: at each instant, $\mathcal{R}[\rho, m_d]$ acts with the current value of $m_d$.

---

## VI. Verification Against Gemini's Calculation (2025-01-08)

The source document states that Gemini verified this calculation on 2025-01-08. The Gemini-verified formula is:

$$
\frac{d}{dt} D_{\text{KL}}(\rho \| \pi) \Big|_{\text{revival}} = \lambda m_d \left( 1 + \frac{D_{\text{KL}}(\rho \| \pi)}{\|\rho\|_{L^1}} \right)
$$

**Comparison with our derivation**: Our Step 5 produces exactly this formula. ✅

**Gemini's conclusion** (quoted from source): "Since $D_{\text{KL}} \ge 0$, $\lambda > 0$, $m_d > 0$, and $\|\rho\|_{L^1} > 0$, the entropy production is **strictly positive**."

**Agreement**: Our Step 6 reaches the identical conclusion. ✅

**Confidence**: The derivation is algebraically straightforward (no integration by parts, no boundary terms, no functional inequalities). The only subtlety is recognizing that $\int \rho = \|\rho\|_{L^1}$ and $\int \rho \log(\rho/\pi) = D_{\text{KL}}$.

---

## VII. Connections to Broader Framework

### Connection 1: Contrast with Finite-N Cloning Operator

**Finite-N result** ({prf:ref}`thm-cloning-lsi-preservation`, 09_kl_convergence.md): The discrete cloning operator **preserves** the Log-Sobolev inequality (LSI) and is KL-contractive in discrete time.

**Mean-field result** (this theorem): The continuous-time revival operator is KL-**expansive**.

**Reconciliation**:
- The finite-N cloning operator includes **inelastic collisions** and **momentum conservation**, which create dissipation
- The mean-field revival operator is a **pure proportional resampling** without collision dynamics
- The limiting process $N \to \infty$ eliminates the microscopic collision structure, leaving only the macroscopic mass transfer

**Implication**: The LSI-preservation property does **not** survive the mean-field limit for the revival operator in isolation. This justifies the kinetic dominance approach.

### Connection 2: Joint Operator Analysis (Stage 0, § 7.2)

**Follow-up theorem** ({prf:ref}`thm-joint-not-contractive`, 16_convergence_mean_field.md § 7.2): The combined killing + revival operator is **NOT** unconditionally KL-contractive either.

**Mechanism**: The joint operator entropy production is:

$$
\frac{d}{dt} D_{\text{KL}}(\rho \| \pi) \Big|_{\text{jump}} = (\lambda m_d - \int \kappa_{\text{kill}} \rho) + \int \left(\frac{\lambda m_d}{\|\rho\|_{L^1}} - \kappa_{\text{kill}}(x)\right) \rho \log \frac{\rho}{\pi}
$$

The sign depends on whether the system is above or below equilibrium mass, **not** on the KL-divergence itself.

**Conclusion**: KL-convergence requires the **kinetic operator** to dominate both the killing and revival expansions.

### Connection 3: Two-State Discrete Model (Stage 0, § 4.1)

**Simplified model** ({prf:ref}`obs-revival-rate-constraint`, 16_convergence_mean_field.md § 4.1): In a two-state system, the revival operator is KL-non-expansive if and only if:

$$
\lambda_{\text{revive}} \le 1
$$

**Physical interpretation**: Revival rate must not exceed the death rate for stability.

**Extension to continuous case**: The continuous theorem does not assume $\lambda \le 1$. Instead, it shows unconditional expansion for $\lambda > 0$. This suggests the two-state constraint does not extend to the full continuum setting (the geometry of the state space matters).

---

## VIII. Difficulty Assessment

**Overall Difficulty**: **LOW**

**Justification**:
- The proof is a **direct calculation** using the infinitesimal generator method
- No functional inequalities (LSI, Poincaré, etc.) are required
- No integration by parts or boundary term analysis
- No hypocoercivity theory or coupling arguments
- The only non-trivial step is recognizing the definitions of $\|\rho\|_{L^1}$ and $D_{\text{KL}}(\rho \| \pi)$

**Prerequisites**:
- Basic calculus of variations
- Definition of KL-divergence
- Standard entropy production formula (Lemma 1)

**Comparison to related results**:
- Easier than proving the kinetic operator is KL-contractive (which requires LSI and hypocoercivity)
- Easier than analyzing the joint jump operator (which requires QSD equilibrium conditions)
- Similar difficulty to verifying mass conservation (algebraic manipulation)

---

## IX. Expansion Time Estimate

**Full Proof Expansion**: **2-4 hours**

**Breakdown**:
1. **Introduction and context** (30 min): State the theorem, explain significance, outline proof strategy
2. **Lemma proofs** (30 min): Prove or cite Lemmas 1-3 (mostly standard results)
3. **Main calculation** (1 hour): Steps 1-6 with detailed algebraic steps
4. **Verification and cross-checks** (30 min): Compare to Gemini's calculation, check hypotheses
5. **Discussion and connections** (30-60 min): Relate to finite-N results, joint operator, two-state model
6. **Formatting and cross-references** (30 min): Add proper MyST directives, labels, citations

**Confidence**: High. This is a straightforward calculation that has already been verified by Gemini (2025-01-08). The expansion is primarily a matter of **writing out the details** that are currently implicit in the source document's condensed presentation.

**Dependencies for full proof**:
- None (self-contained)
- Optional: Cite standard references for entropy production formula (e.g., Otto-Villani, Villani "Topics in Optimal Transportation" Ch 23)

---

## X. Cross-References to Framework Documents

### Definitions Used

| Label | Document | Statement | Role in Proof |
|-------|----------|-----------|---------------|
| `def-revival-operator-formal` | 16_convergence_mean_field.md § 1.2 | $\mathcal{R}[\rho, m_d] = \lambda m_d \rho / \|\rho\|_{L^1}$ | Main object of study (Step 2) |
| KL-divergence (standard) | Information theory | $D_{\text{KL}}(\rho \| \pi) = \int \rho \log(\rho/\pi)$ | Quantity being tracked (Steps 1, 4, 5) |
| $L^1$ norm | Functional analysis | $\|\rho\|_{L^1} = \int \rho$ | Normalization factor (Steps 2, 4, 5) |

### Theorems Used

| Label | Document | Statement | Role in Proof |
|-------|----------|-----------|---------------|
| Entropy production identity | Standard (Otto-Villani) | $\frac{d}{dt} D_{\text{KL}} = \int F[\rho] (1 + \log \rho/\pi)$ | Foundational formula (Step 1, Lemma 1) |
| KL non-negativity | Standard (Gibb's inequality) | $D_{\text{KL}}(\rho \| \pi) \ge 0$ | Strict positivity (Step 6, Lemma 2) |

### Related Results (Context)

| Label | Document | Statement | Relationship to This Theorem |
|-------|----------|-----------|------------------------------|
| `thm-cloning-lsi-preservation` | 09_kl_convergence.md | Finite-N cloning preserves LSI | Contrasts with mean-field expansion (our result) |
| `thm-joint-not-contractive` | 16_convergence_mean_field.md § 7.2 | Joint jump operator not contractive | Extends this result to killing+revival |
| `obs-revival-rate-constraint` | 16_convergence_mean_field.md § 4.1 | Two-state model requires $\lambda \le 1$ | Simplified discrete analog |
| `thm-stage0-complete` | 16_convergence_mean_field.md § 8.1 | Revival expansive → kinetic dominance strategy | Uses this theorem as Component 1 |

### Framework Context

- **Chapter**: 2 (Geometric Gas / Mean-Field Regime)
- **Stage in Roadmap**: Stage 0 (GO/NO-GO feasibility study)
- **Proof Technique Category**: Infinitesimal generator / entropy production
- **Functional Analytic Tools**: KL-divergence calculus (no LSI, Poincaré, or hypocoercivity in this proof)

---

## XI. Open Questions and Future Directions

### Question 1: Revival Rate Constraint

**Issue**: Does the two-state model's constraint $\lambda_{\text{revive}} \le 1$ have any analog in the continuous setting?

**Observation**: The continuous theorem shows unconditional expansion (no constraint on $\lambda$). However, the **magnitude** of the expansion depends on $\lambda$:

$$
\frac{d}{dt} D_{\text{KL}} = \lambda m_d (\ldots) \propto \lambda
$$

**Conjecture**: While expansion occurs for all $\lambda > 0$, the convergence of the **full system** (kinetic + jump) may require $\lambda < \lambda_{\text{crit}}$ for some critical threshold depending on the diffusion strength $\sigma^2$.

**Investigation**: This is addressed in Stage 1-3 of the roadmap via the kinetic dominance condition:

$$
\sigma^2 > \sigma_{\text{crit}}^2 \propto \lambda
$$

### Question 2: Non-Proportional Revival

**Issue**: What if revival is **not** proportional to $\rho$ but uses a different resampling law (e.g., uniform in space, peaked near boundaries)?

**Speculation**: The KL-expansion is a consequence of the **proportional** nature of $\mathcal{R}$. Non-proportional revival could be KL-contractive if it preferentially samples regions where $\rho/\pi$ is large (reducing divergence).

**Relevance**: Not applicable to the current Euclidean Gas model, but interesting for generalizations.

### Question 3: Higher-Order Entropy Measures

**Issue**: The theorem studies KL-divergence (first-order entropy). What about higher-order information measures (e.g., $\chi^2$-divergence, Rényi divergence)?

**Observation**: The entropy production formula involves only first derivatives of $\rho$. Higher-order divergences would introduce additional geometry-dependent terms.

**Conjecture**: The revival operator is likely expansive for all Rényi divergences $D_\alpha(\rho \| \pi)$ with $\alpha \ge 1$, but this requires verification.

---

## XII. Final Summary

**Theorem Statement (Informal)**: The mean-field revival operator, which resurrects dead walkers by sampling proportionally from the current alive distribution, **increases** the KL-divergence to the quasi-stationary distribution. This expansion is strict and unconditional (holds for all $\lambda > 0$, $m_d > 0$, $\rho \neq \pi$).

**Proof Strategy**: Direct calculation via infinitesimal generator method. Compute $\frac{d}{dt} D_{\text{KL}}(\rho \| \pi) = \int \mathcal{R}[\rho] (1 + \log \rho/\pi)$, recognize $\int \rho = \|\rho\|_{L^1}$ and $\int \rho \log(\rho/\pi) = D_{\text{KL}}$, factor to obtain explicit formula, verify strict positivity.

**Difficulty**: LOW (algebraic manipulation, no functional inequalities)

**Expansion Time**: 2-4 hours (mostly writing details of standard calculation)

**Significance**:
- **Refutes** hypothesis that mean-field revival is KL-contractive
- **Validates** kinetic dominance approach (convergence via diffusion dominating expansion)
- **Completes** Stage 0 GO/NO-GO decision (proceed with revised strategy)

**Dependencies**: Self-contained (uses only standard KL-divergence calculus)

**Review Status**:
- ✅ Verified by Gemini (2025-01-08, see source document § 7.1)
- ⚠️ Single-strategist review only (GPT-5/Codex) due to Gemini MCP issues → **FLAG AS LOWER CONFIDENCE**

---

## XIII. Recommendation for Full Proof Development

**Priority**: HIGH (foundational result for Stage 0 completion)

**Next Steps**:
1. Expand Steps 1-6 into detailed prose with all algebraic manipulations shown explicitly
2. Prove or cite Lemmas 1-3 with full references
3. Add pedagogical remarks explaining why normalization by $\|\rho\|_{L^1}$ creates expansion
4. Include numerical example (two-state model from § 4.1) to illustrate the formula
5. Cross-reference to {prf:ref}`thm-joint-not-contractive` and {prf:ref}`thm-stage0-complete`
6. Format with proper MyST directives ({prf:proof}, numbered equations, etc.)

**Quality Check**:
- Verify every algebraic step can be followed by a graduate student in PDE/probability
- Check that all hypotheses are explicitly stated and used
- Ensure the proof is self-contained (no "it is easy to see" without justification)

**Target Audience**: Researchers in McKean-Vlasov equations, quasi-stationary distributions, and stochastic control

**Publication Readiness**: Once expanded, this proof is suitable for inclusion in a journal article on mean-field KL-convergence (e.g., SIAM Journal on Mathematical Analysis, Journal of Functional Analysis, Probability Theory and Related Fields)

---

**End of Proof Sketch**
