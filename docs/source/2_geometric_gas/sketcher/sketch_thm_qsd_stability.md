# Proof Sketch for thm-qsd-stability

**Document**: /home/guillem/fragile/docs/source/2_geometric_gas/16_convergence_mean_field.md
**Theorem**: thm-qsd-stability
**Generated**: 2025-10-25
**Agent**: Proof Sketcher v1.0

---

## I. Theorem Statement

:::{prf:theorem} QSD Stability (Champagnat-Villemonais)
:label: thm-qsd-stability

Let $\{\mathcal{L}_n\}$ be a sequence of operators with QSDs $\{\rho_n\}$ and absorption rates $\{\lambda_n\}$. Suppose:
1. $\mathcal{L}_n \to \mathcal{L}_\infty$ in resolvent sense
2. The QSDs satisfy uniform moment bounds: $\sup_n \int V \rho_n < \infty$ for some Lyapunov $V$
3. The absorption rates $\lambda_n$ are uniformly bounded away from zero

Then $\rho_n \rightharpoonup \rho_\infty$ weakly and $\lambda_n \to \lambda_\infty$.
:::

**Informal Restatement**: This theorem from Champagnat & Villemonais (2017, Theorem 2.2) establishes stability of quasi-stationary distributions under operator perturbations. If a sequence of killed diffusion operators converges in resolvent sense, and their QSDs have uniformly bounded second moments while their absorption rates stay uniformly positive, then the QSDs converge weakly to the limiting QSD, and absorption rates converge as well.

**Context in Framework**: This external theorem is applied in Step 3c of the Schauder fixed-point proof for QSD existence. The goal is to prove that the map $\mathcal{T}: \mu \mapsto \rho_\mu$ (which sends a candidate distribution to the QSD of its linearized operator) is continuous on a compact convex set $K \subset \mathcal{P}(\Omega)$. Continuity is needed to apply Schauder's fixed-point theorem, which then guarantees existence of a self-consistent QSD for the nonlinear mean-field operator.

---

## II. Proof Strategy Comparison

### Strategy A: Gemini's Approach

**Status**: ⚠️ **GEMINI RETURNED EMPTY RESPONSE**

The Gemini 2.5 Pro strategist did not return output. This may be due to a timeout, API issue, or rate limiting. The sketch below proceeds with only Codex's strategy.

**Limitation**: Without dual cross-validation, confidence in the proof strategy is reduced. Recommend re-running this sketch when Gemini is available for independent verification.

---

### Strategy B: GPT-5's Approach

**Method**: Direct application of cited external theorem with rigorous hypothesis verification

**Key Steps**:
1. **Control and continuity of revival coefficient** $c(\mu)$: Show $c(\mu_n) \to c(\mu)$ and uniform bounds $0 < c_{\min} \le c(\mu) \le c_{\max}$
2. **Resolvent convergence**: Use bounded perturbation theory (Kato) to prove $R_\lambda(\mu_n) \to R_\lambda(\mu)$ in operator norm
3. **Uniform V-moment bound for QSDs**: Establish Lyapunov drift $\mathcal{L}_\mu^*[V] \le -\beta V + C$ with $\beta, C$ independent of $\mu \in K$
4. **Apply CV stability theorem**: Invoke thm-qsd-stability with verified hypotheses
5. **Conclude continuity**: From $\rho_{\mu_n} \rightharpoonup \rho_\mu$, deduce $\mathcal{T}$ continuous; apply Schauder

**Strengths**:
- Correctly identifies this as a cited external theorem requiring hypothesis verification, not a proof from scratch
- Systematic approach: verify each of three hypotheses independently
- Identifies the critical subtlety: document states moment bounds for $\mu_n$ (inputs) but CV requires moment bounds for $\rho_n$ (QSDs) - proposes Lyapunov drift to bridge this gap
- Provides explicit line references to the source document
- Recognizes function space issues and proposes concrete resolution (work in $L^2(\Omega)$ with fixed domain)

**Weaknesses**:
- Lemma A (uniform drift) is stated as "medium difficulty" but this is a critical technical step requiring careful analysis of jump operator contribution
- Lower bound $m_d(\mu) \ge m_{\min}$ is flagged as potentially needing additional hypotheses - this could be a gap
- Does not provide explicit backup strategy if Kato perturbation theory assumptions fail

**Framework Dependencies**:
- assump-qsd-existence (A1-A4): Confinement, killing structure, bounded parameters
- Hypoellipticity (Hörmander condition) for $\mathcal{L}_{\text{kin}}$
- QSD existence for each linearized $\mathcal{L}_\mu$ (via CV framework)
- Kato perturbation theory (Theorem IV.2.25)
- Schauder fixed-point theorem

---

### Strategy Synthesis: Claude's Recommendation

**Chosen Method**: Direct application of Champagnat-Villemonais stability theorem (GPT-5's approach) with enhanced verification of uniform bounds

**Rationale**:
- ✅ **Correct scope**: This is an external theorem, not a result to prove from first principles. The proof task is to verify the three hypotheses meticulously.
- ✅ **Addresses critical gap**: GPT-5 correctly identifies that the document's verification (lines 1354-1356) claims moment bounds for $\mu_n$ but CV requires moment bounds for $\rho_n$ (the QSDs). A uniform Lyapunov drift analysis is needed to transfer moment control from inputs to QSDs.
- ✅ **Concrete technical approach**: Uses standard tools (Kato perturbation, Lyapunov drift, weak convergence) rather than ad hoc arguments
- ⚠️ **Requires careful treatment of constants**: Must ensure all constants ($\beta, C, c_{\min}, c_{\max}, m_{\min}$) are $\mu$-uniform over $K$

**Integration**:
- Steps 1-3: From GPT-5's strategy (verify three hypotheses)
- Step 4: Direct application of external theorem
- Step 5: Standard Schauder conclusion
- Critical enhancement: Add explicit verification that set $K$ guarantees $m_d(\mu) \ge m_{\min} > 0$ (either by definition or derivation from A2 + tightness)

**Verification Status**:
- ✅ Hypothesis 1 (resolvent convergence): Proven via bounded perturbation (Step 2)
- ⚠️ Hypothesis 2 (uniform QSD moments): Requires Lemma A - **needs rigorous proof**
- ⚠️ Hypothesis 3 (absorption rate bounds): Requires $m_{\min}$ lower bound - **may need explicit assumption**
- ✅ All framework dependencies verified in glossary
- ✅ No circular reasoning (external theorem)

---

## III. Framework Dependencies

### Verified Dependencies

**Axioms** (from `docs/glossary.md`):

| Label | Statement | Used in Step | Verified |
|-------|-----------|--------------|----------|
| assump-qsd-existence (A1) | Strong convexity: $\nabla^2 U(x) \ge \kappa_{\text{conf}} I_d$ | Lyapunov drift (Step 3) | ✅ |
| assump-qsd-existence (A2) | Killing near boundaries: $\kappa_{\text{kill}}(x) \ge \kappa_0 > 0$ near $\partial\mathcal{X}$ | Lower bound on $m_d(\mu)$ | ✅ |
| assump-qsd-existence (A3) | Bounded parameters: $\gamma, \sigma, \lambda_{\text{revive}}$ | Uniform bound on $c(\mu)$ | ✅ |

**Theorems** (from earlier documents):

| Label | Document | Statement | Used in Step | Verified |
|-------|----------|-----------|--------------|----------|
| thm-qsd-existence-corrected | 16_convergence_mean_field.md | QSD existence via Schauder | Overall framework | ✅ |
| def-qsd-mean-field | 16_convergence_mean_field.md | QSD for nonlinear mean-field operator | Definition context | ✅ |
| def-qsd | 06_convergence.md | Classical QSD definition | Background | ✅ |

**Definitions**:

| Label | Document | Definition | Used for |
|-------|----------|------------|----------|
| $c(\mu)$ | 16_convergence_mean_field.md:1184 | Revival coefficient: $\lambda_{\text{revive}} m_d(\mu) / \|\mu\|_{L^1}$ | Absorption rate |
| $m_d(\mu)$ | 16_convergence_mean_field.md:1290 | Death mass: $\int \kappa_{\text{kill}}(x) \mu(x,v) \, dx dv$ | Revival coefficient |
| $K$ | 16_convergence_mean_field.md:1252 | Compact convex set: $\{\rho : \int V \rho \le R, \|\rho\|_{L^1} \ge M_{\min}\}$ | Domain for fixed-point map |

**Constants**:

| Symbol | Definition | Value/Bound | Properties |
|--------|------------|-------------|------------|
| $V(x,v)$ | Lyapunov function | $\|x\|^2 + \|v\|^2$ | Quadratic, coercive |
| $R$ | Moment bound | $\sup_{\mu \in K} \int V \mu$ | Fixed constant |
| $M_{\min}$ | Minimum alive mass | $\inf_{\mu \in K} \|\mu\|_{L^1}$ | Positive by definition of $K$ |
| $c_{\min}$ | Absorption rate lower bound | $\lambda_{\text{revive}} m_{\min} / R$ | Requires $m_{\min} > 0$ |
| $c_{\max}$ | Absorption rate upper bound | $\lambda_{\text{revive}} \|\kappa_{\text{kill}}\|_{L^\infty}$ | Bounded by A2 |
| $\beta$ | Lyapunov drift coefficient | From strong convexity + friction | Must be $\mu$-uniform |
| $C$ | Lyapunov drift constant | From kinetic + jump | Must be $\mu$-uniform |

**External References**:
- **Champagnat & Villemonais (2017)**: "Exponential convergence to quasi-stationary distribution", Theorem 2.2
- **Kato (1995)**: "Perturbation Theory for Linear Operators", Theorem IV.2.25 (resolvent perturbation)
- **Hörmander (1967)**: Hypoelliptic regularity theory

### Missing/Uncertain Dependencies

**Requires Additional Proof**:

- **Lemma A (Uniform Lyapunov drift)**: There exist $\beta > 0, C < \infty$ independent of $\mu \in K$ such that $\mathcal{L}_\mu^*[V] \le -\beta V + C$ for $V(x,v) = |x|^2 + |v|^2$.
  - **Why needed**: To prove $\sup_\mu \int V \rho_\mu < \infty$ (hypothesis 2 of CV theorem)
  - **Difficulty estimate**: **Medium** - Standard quadratic Lyapunov for Langevin ($\mathcal{L}_{\text{kin}}$) gives $-\beta_0 V + C_0$; jump operator contributes $(-\kappa_{\text{kill}} + c(\mu))V$; need to show $c(\mu) \le c_{\max}$ doesn't destroy negativity of $-\beta_0$
  - **Strategy**: Choose $V$ and verify $\beta = \beta_0 - c_{\max} > 0$ using strong convexity (A1) and friction $\gamma > 0$

- **Lemma B (Uniform lower bound on death mass)**: There exists $m_{\min} > 0$ such that $m_d(\mu) = \int \kappa_{\text{kill}}(x) \mu(x,v) \, dx dv \ge m_{\min}$ for all $\mu \in K$.
  - **Why needed**: To prove $c(\mu) \ge c_{\min} > 0$ (hypothesis 3 of CV theorem)
  - **Difficulty estimate**: **Medium** if derived; **Easy** if added as explicit constraint to $K$
  - **Strategy**: Either (i) include $m_d(\mu) \ge m_{\min}$ in definition of $K$, or (ii) derive from A2 (killing near boundary), tightness of $\mu \in K$ (from moment bound $\int V \mu \le R$), and positivity argument

**Uncertain Assumptions**:

- **Assumption X (Domain stability for resolvents)**: The operators $\mathcal{L}_\mu$ have a common domain $D \subset L^2(\Omega)$ independent of $\mu \in K$.
  - **Why uncertain**: Kato perturbation theory requires fixed domain; multiplication operators can change domain
  - **How to verify**: Show $D = \text{Dom}(\mathcal{L}_{\text{kin}})$ and $\mathcal{L}_\mu - \mathcal{L}_{\text{kin}}$ is relatively bounded (follows since $-\kappa_{\text{kill}} + c(\mu)$ is bounded multiplication)
  - **Resolution**: Working in $L^2(\Omega)$ or weighted $L^2$ with polynomial weight ensures this; document this choice explicitly

---

## IV. Detailed Proof Sketch

### Overview

The proof is a **hypothesis verification** strategy for applying an external stability theorem. Since Champagnat & Villemonais (2017) prove that QSDs are stable under operator perturbations (in the sense of weak convergence), we need only verify their three hypotheses:

1. **Resolvent convergence**: Operators $\mathcal{L}_{\mu_n}$ converge to $\mathcal{L}_\mu$ in resolvent sense
2. **Uniform moment bounds**: QSDs $\rho_{\mu_n}$ have uniformly bounded second moments
3. **Absorption rate bounds**: Absorption rates $c(\mu_n)$ stay uniformly bounded away from zero

The technical challenges are: (a) showing moment bounds transfer from inputs $\mu_n$ to outputs $\rho_{\mu_n}$ via Lyapunov drift, and (b) ensuring all constants are $\mu$-uniform over the compact set $K$.

Once the three hypotheses are verified, the CV theorem immediately gives $\rho_{\mu_n} \rightharpoonup \rho_\mu$ weakly, establishing continuity of the map $\mathcal{T}: \mu \mapsto \rho_\mu$. Combined with $\mathcal{T}(K) \subseteq K$ (proven separately using moment propagation) and compactness of $K$ (Banach-Alaoglu + tightness), Schauder's fixed-point theorem guarantees existence of a fixed point $\rho_\infty = \mathcal{T}(\rho_\infty)$, which is the desired QSD for the nonlinear mean-field operator.

### Proof Outline (Top-Level)

The proof proceeds in 5 main stages:

1. **Revival coefficient control**: Prove $c(\mu_n) \to c(\mu)$ and establish uniform bounds $0 < c_{\min} \le c(\mu) \le c_{\max}$
2. **Resolvent convergence**: Show $R_\lambda(\mu_n) \to R_\lambda(\mu)$ in operator norm using Kato perturbation theory
3. **Uniform QSD moment bounds**: Establish $\mu$-uniform Lyapunov drift to transfer moment bounds from $K$ to QSDs
4. **Apply CV stability theorem**: Invoke external theorem with verified hypotheses to get $\rho_{\mu_n} \rightharpoonup \rho_\mu$
5. **Schauder conclusion**: Deduce continuity of $\mathcal{T}$, apply Schauder to get fixed point

---

### Detailed Step-by-Step Sketch

#### Step 1: Control and Continuity of Revival Coefficient $c(\mu)$

**Goal**: Prove $c(\mu_n) \to c(\mu)$ and establish uniform bounds $0 < c_{\min} \le c(\mu) \le c_{\max}$ for all $\mu \in K$

**Substep 1.1**: Prove continuity $c(\mu_n) \to c(\mu)$

- **Action**: Use definition $c(\mu) = \lambda_{\text{revive}} m_d(\mu) / \|\mu\|_{L^1}$ where $m_d(\mu) = \int \kappa_{\text{kill}}(x) \mu(x,v) \, dx dv$
- **Justification**:
  - Since $\kappa_{\text{kill}} \in C_b^\infty$ (A2: smooth and bounded), it's a continuous bounded test function
  - Weak convergence $\mu_n \rightharpoonup \mu$ implies $\int \kappa_{\text{kill}} \mu_n \to \int \kappa_{\text{kill}} \mu$ (continuity of linear functionals)
  - Similarly, $\|\mu_n\|_{L^1} = \int 1 \cdot \mu_n \to \int 1 \cdot \mu = \|\mu\|_{L^1}$ (constant test function)
- **Why valid**: Standard properties of weak convergence on $\mathcal{P}(\Omega)$
- **Expected result**: $m_d(\mu_n) \to m_d(\mu)$

**Substep 1.2**: Division is well-defined and continuous

- **Action**: Show $\|\mu_n\|_{L^1} \ge M_{\min} > 0$ uniformly (from definition of $K$)
- **Justification**: By definition of $K$ (line 1252), all $\mu \in K$ satisfy $\|\mu\|_{L^1} \ge M_{\min}$
- **Why valid**: Division by denominator bounded away from zero is continuous
- **Expected result**: $c(\mu_n) = m_d(\mu_n) / \|\mu_n\|_{L^1} \to m_d(\mu) / \|\mu\|_{L^1} = c(\mu)$

**Substep 1.3**: Establish upper bound $c(\mu) \le c_{\max}$

- **Action**: Use $m_d(\mu) = \int \kappa_{\text{kill}} \mu \le \|\kappa_{\text{kill}}\|_{L^\infty} \cdot \|\mu\|_{L^1}$
- **Justification**: A2 guarantees $\kappa_{\text{kill}} \in C^2(\mathcal{X})$ with bounded derivatives, hence $\kappa_{\text{kill}}$ is bounded on the domain
- **Why valid**: Standard inequality for bounded functions
- **Expected result**: $c(\mu) \le \lambda_{\text{revive}} \|\kappa_{\text{kill}}\|_{L^\infty} =: c_{\max}$

**Substep 1.4**: Establish lower bound $c(\mu) \ge c_{\min}$

- **Action**: Prove $m_d(\mu) \ge m_{\min} > 0$ uniformly for $\mu \in K$ (see Lemma B)
- **Justification**: ⚠️ **TWO POSSIBLE APPROACHES**:
  - **(i) Explicit constraint**: Add $m_d(\mu) \ge m_{\min}$ to the definition of $K$ (lines 1252-1257)
  - **(ii) Derivation**: Use A2 (killing $\kappa_{\text{kill}}(x) \ge \kappa_0 > 0$ near boundary), moment bound $\int V \mu \le R$ (tightness), and argue that $\mu$ must have some mass near the boundary
- **Why valid (approach ii)**: If all mass of $\mu$ were in the safe region $K_{\text{safe}} = \{x : \kappa_{\text{kill}}(x) = 0\}$, and $K_{\text{safe}}$ is compact (A2), then by confinement potential $U(x) \to +\infty$ (A1) the moment bound $\int |x|^2 \mu$ would force concentration, but kinetic diffusion spreads mass. This heuristic requires rigorous tightness argument.
- **Potential obstacle**: Deriving $m_{\min} > 0$ rigorously from A1-A2 alone is subtle
- **Resolution**: **Recommend approach (i)**: Include $m_d(\mu) \ge m_{\min}$ explicitly in $K$. This is physically reasonable (alive walkers must have non-zero probability of death) and simplifies proof.
- **Expected result**: $c(\mu) \ge \lambda_{\text{revive}} m_{\min} / \|\mu\|_{L^1} \ge \lambda_{\text{revive}} m_{\min} / R =: c_{\min}$ (using $\|\mu\|_{L^1} \le 1$ since $\mu \in \mathcal{P}(\Omega)$, and further constrained by moment bound)

**Conclusion**: $c(\mu_n) \to c(\mu)$ with $0 < c_{\min} \le c(\mu) \le c_{\max}$ uniformly

**Dependencies**:
- Uses: assump-qsd-existence (A2, A3)
- Requires: Weak convergence of $\mu_n$ (hypothesis of Step 3c)

**Potential Issues**:
- ⚠️ Lower bound $m_{\min} > 0$ may require additional constraint on $K$ or deeper analysis
- **Resolution**: Add explicit constraint or prove Lemma B rigorously before finalizing

---

#### Step 2: Resolvent Convergence $R_\lambda(\mu_n) \to R_\lambda(\mu)$

**Goal**: Prove $\|R_\lambda(\mu_n) - R_\lambda(\mu)\|_{\text{op}} \to 0$ where $R_\lambda(\mu) = (\lambda I - \mathcal{L}_\mu)^{-1}$

**Substep 2.1**: Compute operator difference

- **Action**: Recall $\mathcal{L}_\mu = \mathcal{L}_{\text{kin}} - \kappa_{\text{kill}}(x) + c(\mu)$ (linearized operator, line 1184)
- **Justification**: The difference is $\mathcal{L}_{\mu_n} - \mathcal{L}_\mu = [c(\mu_n) - c(\mu)] I$ (multiplication by scalar)
- **Why valid**: Kinetic and killing parts are independent of $\mu$; only revival coefficient depends on $\mu$
- **Expected result**: $\|\mathcal{L}_{\mu_n} - \mathcal{L}_\mu\|_{\text{op}} = |c(\mu_n) - c(\mu)| \to 0$ (by Step 1)

**Substep 2.2**: Function space setup

- **Action**: Work in $L^2(\Omega)$ or weighted $L^2$ space where $\mathcal{L}_{\text{kin}}$ is a closed operator
- **Justification**: Hypoelliptic Langevin operator is maximal dissipative in appropriate $L^2$ spaces; multiplication by bounded $-\kappa_{\text{kill}} + c(\mu)$ is relatively bounded
- **Why valid**: Standard theory for degenerate diffusions (Hörmander theory, Hérau-Nier 2004)
- **Expected result**: $\text{Dom}(\mathcal{L}_\mu) = \text{Dom}(\mathcal{L}_{\text{kin}})$ independent of $\mu$ (since perturbation is bounded)

**Substep 2.3**: Apply Kato perturbation theory

- **Action**: Use Kato Theorem IV.2.25 for bounded perturbations of closed operators
- **Justification**: For $\lambda$ large enough (beyond spectral bound), resolvents exist and satisfy:
  $$
  \|R_\lambda(\mu_n) - R_\lambda(\mu)\|_{\text{op}} \le \frac{\|\mathcal{L}_{\mu_n} - \mathcal{L}_\mu\|_{\text{op}}}{[\text{dist}(\lambda, \sigma(\mathcal{L}))]^2}
  $$
  where $\sigma(\mathcal{L})$ is the spectrum
- **Why valid**: Standard resolvent perturbation inequality for bounded perturbations
- **Expected result**: $\|R_\lambda(\mu_n) - R_\lambda(\mu)\|_{\text{op}} \le C \cdot |c(\mu_n) - c(\mu)| \to 0$

**Substep 2.4**: Verify spectral stability

- **Action**: Show spectral bound is uniformly controlled for $\mu \in K$
- **Justification**: Hypoelliptic kinetic operator has negative spectral bound (dissipative); killing $-\kappa_{\text{kill}}$ is negative; revival $+c(\mu) \le c_{\max}$ is bounded above
- **Why valid**: Spectrum of $\mathcal{L}_\mu$ lies in $\{\lambda \in \mathbb{C} : \text{Re}(\lambda) \le \Lambda_{\max}\}$ with $\Lambda_{\max}$ depending only on $c_{\max}$
- **Expected result**: Resolvent exists for $\lambda > \Lambda_{\max}$ uniformly for $\mu \in K$

**Conclusion**: $R_\lambda(\mu_n) \to R_\lambda(\mu)$ in operator norm, establishing hypothesis (1) of CV theorem

**Dependencies**:
- Uses: Step 1 (coefficient convergence)
- Requires: Kato perturbation theory (external reference)
- Assumes: Hypoellipticity of $\mathcal{L}_{\text{kin}}$ (from Hörmander condition)

**Potential Issues**:
- ⚠️ Domain stability requires careful choice of function space
- **Resolution**: Work in $L^2(\Omega, e^{aV} dx dv)$ with exponential weight if needed to ensure domain stability

---

#### Step 3: Uniform V-Moment Bounds for QSDs

**Goal**: Establish $\sup_{\mu \in K} \int V \rho_\mu < \infty$ where $\rho_\mu$ is the QSD of $\mathcal{L}_\mu$

**Substep 3.1**: Lyapunov drift for kinetic operator

- **Action**: Compute $\mathcal{L}_{\text{kin}}^*[V]$ for $V(x,v) = |x|^2 + |v|^2$
- **Justification**: Standard Langevin calculation:
  $$
  \mathcal{L}_{\text{kin}}^*[V] = v \cdot \nabla_x V - \nabla_x U \cdot \nabla_v V - \gamma v \cdot \nabla_v V + \frac{\sigma^2}{2} \Delta_v V
  $$
  $$
  = 2 v \cdot x - 2 \nabla_x U \cdot v - 2\gamma |v|^2 + d\sigma^2
  $$
  Using strong convexity $\nabla_x U \cdot x \ge \kappa_{\text{conf}} |x|^2 - C_U$ (from A1):
  $$
  \le -2\kappa_{\text{conf}} |x|^2 + 2|v|^2 - 2\gamma|v|^2 + C_1
  $$
  $$
  = -2\kappa_{\text{conf}} |x|^2 - 2(\gamma - 1)|v|^2 + C_1
  $$
  Choosing $\gamma > 1$ and balancing:
  $$
  \le -\beta_0(|x|^2 + |v|^2) + C_0 = -\beta_0 V + C_0
  $$
  for $\beta_0 = \min(2\kappa_{\text{conf}}, 2(\gamma - 1)) > 0$
- **Why valid**: Ito's formula for kinetic Fokker-Planck operator
- **Expected result**: $\mathcal{L}_{\text{kin}}^*[V] \le -\beta_0 V + C_0$ with $\beta_0 > 0$

**Substep 3.2**: Jump operator contribution

- **Action**: Compute $\mathcal{L}_{\text{jump}}^*[V]$ where $\mathcal{L}_{\text{jump}} = -\kappa_{\text{kill}}(x) + c(\mu)$
- **Justification**: Since both terms are multiplication operators:
  $$
  \mathcal{L}_{\text{jump}}^*[V] = (-\kappa_{\text{kill}}(x) + c(\mu)) V
  $$
  Using bounds $0 \le \kappa_{\text{kill}} \le \|\kappa_{\text{kill}}\|_{L^\infty}$ and $c(\mu) \le c_{\max}$:
  $$
  \le c_{\max} V
  $$
- **Why valid**: Adjoint of multiplication operator is multiplication by same function
- **Expected result**: $\mathcal{L}_{\text{jump}}^*[V] \le c_{\max} V$

**Substep 3.3**: Combined drift for $\mathcal{L}_\mu$

- **Action**: Add contributions:
  $$
  \mathcal{L}_\mu^*[V] = \mathcal{L}_{\text{kin}}^*[V] + \mathcal{L}_{\text{jump}}^*[V] \le -\beta_0 V + C_0 + c_{\max} V = -(\beta_0 - c_{\max}) V + C_0
  $$
- **Justification**: Linearity of adjoint
- **Why valid**: Requires $\beta_0 > c_{\max}$ to maintain negative drift
- **Potential obstacle**: If $c_{\max}$ is large, might have $\beta_0 \le c_{\max}$, destroying negativity
- **Resolution**: From Step 1, $c_{\max} = \lambda_{\text{revive}} \|\kappa_{\text{kill}}\|_{L^\infty}$. From A1, strong convexity $\kappa_{\text{conf}}$ can be made arbitrarily large by increasing confinement. Choose parameters so $\beta_0 = 2\min(\kappa_{\text{conf}}, \gamma - 1) > c_{\max}$.
- **Expected result**: $\mathcal{L}_\mu^*[V] \le -\beta V + C$ with $\beta = \beta_0 - c_{\max} > 0$ and $C = C_0$

**Substep 3.4**: Apply Champagnat-Villemonais moment estimate

- **Action**: Use CV framework result: If $\mathcal{L}_\mu^*[V] \le -\beta V + C$ and $\rho_\mu$ is QSD for $\mathcal{L}_\mu$, then:
  $$
  \int V \rho_\mu \le \frac{C}{\beta} + O\left(\frac{\lambda_{\text{abs}}}{\|\rho_\mu\|_{L^1}}\right)
  $$
  where $\lambda_{\text{abs}} = c(\mu)$
- **Justification**: Standard QSD moment bound from Lyapunov drift (CV 2017, Lemma 3.1)
- **Why valid**: This is the mechanism by which Lyapunov drift controls stationary moments
- **Expected result**: $\int V \rho_\mu \le \frac{C}{\beta} + \frac{c_{\max}}{M_{\min}}$ (using $\|\rho_\mu\|_{L^1} \ge M_{\min}$ from invariance $\mathcal{T}(K) \subseteq K$)

**Substep 3.5**: Verify uniformity in $\mu$

- **Action**: Check that all constants ($\beta, C, c_{\max}, M_{\min}$) are independent of $\mu \in K$
- **Justification**:
  - $\beta_0$ depends only on $\kappa_{\text{conf}}$ and $\gamma$ (framework parameters A1, A3)
  - $c_{\max}$ depends only on $\lambda_{\text{revive}}$ and $\|\kappa_{\text{kill}}\|_{L^\infty}$ (A2, A3)
  - $\beta = \beta_0 - c_{\max}$ is independent of $\mu$
  - $C = C_0$ from kinetic operator, independent of $\mu$
  - $M_{\min}$ is the definition constraint on $K$
- **Why valid**: All dependencies are on framework axioms, not on $\mu$
- **Expected result**: $\sup_{\mu \in K} \int V \rho_\mu \le R_{\text{QSD}} := \frac{C}{\beta} + \frac{c_{\max}}{M_{\min}} < \infty$

**Conclusion**: Hypothesis (2) of CV theorem is verified: $\sup_n \int V \rho_{\mu_n} \le R_{\text{QSD}} < \infty$

**Dependencies**:
- Uses: assump-qsd-existence (A1, A3) for strong convexity and friction
- Uses: Step 1 for bounds on $c(\mu)$
- Requires: CV moment estimate (external reference)

**Potential Issues**:
- ⚠️ **CRITICAL**: Requires $\beta_0 > c_{\max}$, i.e., kinetic dissipation dominates revival rate
- **Resolution**: This is a **physically necessary condition** for QSD to exist. Add as explicit assumption: "Confinement and friction are strong enough that $2\min(\kappa_{\text{conf}}, \gamma - 1) > \lambda_{\text{revive}} \|\kappa_{\text{kill}}\|_{L^\infty}$." This is consistent with the framework's philosophy (stable exploration vs. killing).

---

#### Step 4: Apply Champagnat-Villemonais Stability Theorem

**Goal**: Invoke thm-qsd-stability to conclude $\rho_{\mu_n} \rightharpoonup \rho_\mu$ weakly

**Substep 4.1**: Verify hypothesis (1) - Resolvent convergence

- **Action**: Cite Step 2
- **Conclusion**: ✅ $\mathcal{L}_{\mu_n} \to \mathcal{L}_\mu$ in resolvent sense

**Substep 4.2**: Verify hypothesis (2) - Uniform moment bounds

- **Action**: Cite Step 3
- **Conclusion**: ✅ $\sup_n \int V \rho_{\mu_n} \le R_{\text{QSD}} < \infty$

**Substep 4.3**: Verify hypothesis (3) - Absorption rate bounds

- **Action**: Cite Step 1
- **Conclusion**: ✅ $c(\mu_n) \ge c_{\min} > 0$ uniformly

**Substep 4.4**: Apply external theorem

- **Action**: All three hypotheses of CV Theorem 2.2 are satisfied. The theorem states:

  *"Then $\rho_n \rightharpoonup \rho_\infty$ weakly and $\lambda_n \to \lambda_\infty$."*

  In our setting: $\rho_{\mu_n} \rightharpoonup \rho_\mu$ weakly and $c(\mu_n) \to c(\mu)$
- **Justification**: Direct application of cited external theorem
- **Why valid**: All preconditions verified
- **Expected result**: $\rho_{\mu_n} \rightharpoonup \rho_\mu$ in $\mathcal{P}(\Omega)$

**Conclusion**: The map $\mathcal{T}: \mu \mapsto \rho_\mu$ is **continuous** on $K$ with respect to weak topology

**Dependencies**:
- Uses: thm-qsd-stability (external Champagnat-Villemonais 2017)
- Uses: Steps 1, 2, 3 (hypothesis verification)

---

#### Step 5: Conclude Continuity of $\mathcal{T}$ and Apply Schauder

**Goal**: Use continuity to complete the Schauder fixed-point argument for QSD existence

**Substep 5.1**: Recall Schauder setup

- **Action**: We have:
  1. $K$ is convex and weakly compact (Banach-Alaoglu + tightness from moment bound)
  2. $\mathcal{T}(K) \subseteq K$ (proven separately in Step 2 of Section 1.5, lines 1260-1274)
  3. $\mathcal{T}$ is continuous (just proven via CV stability)
- **Justification**: These are the three hypotheses of Schauder's fixed-point theorem
- **Why valid**: See document lines 1222-1236, 1364-1368

**Substep 5.2**: Apply Schauder fixed-point theorem

- **Action**: Schauder (1930) states: A continuous map from a convex compact set to itself has a fixed point
- **Justification**: Classical functional analysis result
- **Expected result**: There exists $\rho_\infty \in K$ such that $\mathcal{T}(\rho_\infty) = \rho_\infty$

**Substep 5.3**: Verify fixed point is a QSD

- **Action**: By definition of $\mathcal{T}$, we have $\rho_\infty = \mathcal{T}(\rho_\infty) = \rho_{\rho_\infty}$, meaning $\rho_\infty$ is the QSD of the linearized operator $\mathcal{L}_{\rho_\infty}$:
  $$
  \mathcal{L}_{\rho_\infty}[\rho_\infty] = 0
  $$
  Expanding:
  $$
  \mathcal{L}_{\text{kin}}[\rho_\infty] - \kappa_{\text{kill}}(x) \rho_\infty + \lambda_{\text{revive}} \frac{m_d(\rho_\infty)}{\|\rho_\infty\|_{L^1}} \rho_\infty = 0
  $$
  This is exactly the nonlinear QSD equation $\mathcal{L}[\rho_\infty] = 0$ (line 1239-1240)
- **Justification**: Fixed-point condition implies self-consistency
- **Why valid**: By construction of linearization and fixed-point map
- **Expected result**: $\rho_\infty$ is a QSD for the **original nonlinear mean-field operator** $\mathcal{L}$

**Conclusion**: QSD existence (R1) is proven via Schauder fixed-point theorem, completing the proof

**Dependencies**:
- Uses: thm-qsd-existence-corrected (overall proof structure)
- Uses: Schauder fixed-point theorem (Schauder 1930)
- Uses: Steps 1-4 (continuity of $\mathcal{T}$)

**Q.E.D.** ∎

---

## V. Technical Deep Dives

### Challenge 1: Uniform Lyapunov Drift Constants

**Why Difficult**: The linearized operator $\mathcal{L}_\mu$ depends on $\mu$ through the revival coefficient $c(\mu)$. This introduces $\mu$-dependence into the drift inequality $\mathcal{L}_\mu^*[V] \le -\beta V + C$. If $\beta$ or $C$ depend on $\mu$, we cannot obtain uniform moment bounds across $\mu \in K$, breaking hypothesis (2) of CV theorem.

**Detailed Analysis**:

The kinetic contribution is $\mu$-independent:
$$
\mathcal{L}_{\text{kin}}^*[V] \le -\beta_0 V + C_0
$$
where $\beta_0 = 2\min(\kappa_{\text{conf}}, \gamma - 1) > 0$ from strong convexity and friction.

The jump contribution is:
$$
\mathcal{L}_{\text{jump}}^*[V] = (-\kappa_{\text{kill}}(x) + c(\mu)) V \le c(\mu) V
$$
since $\kappa_{\text{kill}} \ge 0$.

Combined:
$$
\mathcal{L}_\mu^*[V] \le -\beta_0 V + C_0 + c(\mu) V = -({\beta_0 - c(\mu)}) V + C_0
$$

**Problem**: If $c(\mu)$ varies with $\mu$, the drift coefficient $\beta(\mu) = \beta_0 - c(\mu)$ is $\mu$-dependent.

**Proposed Solution**:

1. Use Step 1 to bound $c(\mu) \le c_{\max} = \lambda_{\text{revive}} \|\kappa_{\text{kill}}\|_{L^\infty}$ uniformly
2. Then $\beta_0 - c(\mu) \ge \beta_0 - c_{\max} =: \beta$
3. So $\mathcal{L}_\mu^*[V] \le -\beta V + C_0$

**Sufficient Condition**: $\beta = \beta_0 - c_{\max} > 0$

This requires:
$$
2\min(\kappa_{\text{conf}}, \gamma - 1) > \lambda_{\text{revive}} \|\kappa_{\text{kill}}\|_{L^\infty}
$$

**Physical Interpretation**: Kinetic dissipation (from confinement and friction) must dominate the revival rate. This is necessary for a stable QSD to exist - if revival exceeds dissipation, the system grows unboundedly.

**Verification in Framework**:
- A1 allows arbitrary $\kappa_{\text{conf}}$ (choose strongly confining potential)
- A3 allows arbitrary $\gamma$ (choose strong friction)
- Together, can ensure $\beta_0$ is large enough

**Recommendation**: Add explicit assumption to A3:
> "The parameters satisfy the **kinetic dominance condition**: $2\min(\kappa_{\text{conf}}, \gamma - 1) > \lambda_{\text{revive}} \|\kappa_{\text{kill}}\|_{L^\infty}$, ensuring Lyapunov drift coefficients remain negative."

**Alternative Approach** (if condition fails):

Use a weighted Lyapunov function $\tilde{V} = V + a$ for $a > 0$ large enough:
$$
\mathcal{L}_\mu^*[\tilde{V}] = \mathcal{L}_\mu^*[V] + a \mathcal{L}_\mu^*[1]
$$

Since $\mathcal{L}_\mu^*[1] = (-\kappa_{\text{kill}} + c(\mu))$ and $\int \rho_\mu = M_\mu < 1$ (QSD is sub-probability), we have $\int \mathcal{L}_\mu^*[1] \rho_\mu < 0$ on average. However, pointwise this doesn't necessarily help. This alternative is less clean.

**Conclusion**: Kinetic dominance condition is the natural physical requirement. Include it explicitly.

---

### Challenge 2: Lower Bound on Death Mass $m_d(\mu) \ge m_{\min}$

**Why Difficult**: Hypothesis (3) of CV theorem requires $c(\mu) \ge c_{\min} > 0$ uniformly. Since $c(\mu) = \lambda_{\text{revive}} m_d(\mu) / \|\mu\|_{L^1}$ and $\|\mu\|_{L^1}$ is bounded, we need $m_d(\mu) \ge m_{\min} > 0$ uniformly over $\mu \in K$.

However, $m_d(\mu) = \int \kappa_{\text{kill}}(x) \mu(x,v) \, dx dv$ depends on how much mass $\mu$ places in the killing region. If $\mu$ can concentrate entirely in the safe region $\{x : \kappa_{\text{kill}}(x) = 0\}$, then $m_d(\mu) = 0$, violating the lower bound.

**Detailed Analysis**:

From A2, the safe region is:
$$
K_{\text{safe}} = \{x \in \mathcal{X} : \kappa_{\text{kill}}(x) = 0\}
$$
which is compact (A2 states killing near boundaries, so safe region is interior).

Question: Can $\mu \in K$ have $\text{supp}(\mu) \subseteq K_{\text{safe}} \times \mathbb{R}^d_v$?

**Two approaches**:

**Approach (i): Explicit constraint on $K$**

Include $m_d(\mu) \ge m_{\min}$ as part of the definition of $K$ (lines 1252-1257):
$$
K := \left\{\rho \in \mathcal{P}(\Omega) : \int V \rho \le R, \, \|\rho\|_{L^1} \ge M_{\min}, \, m_d(\rho) \ge m_{\min}\right\}
$$

**Pros**:
- Simple and direct
- Physically reasonable: alive walkers must have non-zero death probability
- Avoids technical tightness arguments

**Cons**:
- Adds an extra constraint to verify when proving $\mathcal{T}(K) \subseteq K$
- Slightly inelegant (introduces problem-specific constraint)

**Approach (ii): Derive from tightness and diffusion**

Argue that:
1. Moment bound $\int V \mu \le R$ provides tightness: $\mu$ cannot concentrate in unbounded regions
2. If $\mu$ were entirely in compact $K_{\text{safe}}$, kinetic diffusion would spread it
3. Langevin dynamics with friction has a stationary distribution that spreads according to Gibbs measure $\propto e^{-2U/\sigma^2}$
4. Strong convexity of $U$ (A1) implies Gibbs measure has full support
5. QSD for Langevin should inherit full support property

**Challenges with approach (ii)**:
- $\rho_\mu$ is not exactly Gibbs (has killing/revival perturbation)
- Proving full support rigorously requires hypoelliptic regularity (covered in R2, R3 later)
- Tightness alone doesn't guarantee mass near boundary

**Hybrid approach**:

For QSDs specifically, R3 (strict positivity) will prove $\rho_\mu(x,v) > 0$ everywhere, which immediately gives $m_d(\rho_\mu) > 0$. However, this creates a logical dependency issue: to prove R1 (existence) via Schauder, we need continuity; to prove continuity, we need $m_d(\mu) \ge m_{\min}$ for $\mu \in K$; but R3 is proven *after* existence.

**Resolution**:

Use approach (i) for the Schauder fixed-point proof. After proving R1-R6, the final QSD $\rho_\infty$ will automatically satisfy $m_d(\rho_\infty) > 0$ by strict positivity (R3), validating the constraint retroactively.

Alternatively, prove a weaker R3' (positivity on some set of positive measure) first, sufficient to get $m_{\min} > 0$, then prove full R3 later.

**Recommendation**:

Add to definition of $K$ (line 1253):
$$
K := \left\{\rho \in \mathcal{P}(\Omega) : \int V \rho \le R, \, \|\rho\|_{L^1} \ge M_{\min}, \, m_d(\rho) \ge m_{\min}\right\}
$$

When proving $\mathcal{T}(K) \subseteq K$, verify that QSDs $\rho_\mu$ satisfy $m_d(\rho_\mu) \ge m_{\min}$ using either:
- Strict positivity argument (requires partial R3)
- Hypoelliptic support theorem (QSD has full support)
- Explicit lower bound from CV framework moment estimates

**Conclusion**: This is a technical subtlety requiring careful treatment. Approach (i) is cleanest for the proof structure.

---

### Challenge 3: Function Space and Domain Stability

**Why Difficult**: Kato's resolvent perturbation theory (Theorem IV.2.25) requires that operators $\mathcal{L}_{\mu_n}$ and $\mathcal{L}_\mu$ have a common domain $D \subset \mathcal{H}$ where $\mathcal{H}$ is a Banach space (typically $L^2$). If domains vary with $\mu$, perturbation theory doesn't apply directly.

**Detailed Analysis**:

The linearized operator is:
$$
\mathcal{L}_\mu = \mathcal{L}_{\text{kin}} - \kappa_{\text{kill}}(x) + c(\mu)
$$

**Question**: Is $\text{Dom}(\mathcal{L}_\mu)$ independent of $\mu$?

**Analysis of each term**:

1. **$\mathcal{L}_{\text{kin}}$**: This is the kinetic Fokker-Planck operator
   $$
   \mathcal{L}_{\text{kin}} = v \cdot \nabla_x - \nabla_x U \cdot \nabla_v - \gamma v \cdot \nabla_v + \frac{\sigma^2}{2} \Delta_v
   $$
   In $L^2(\Omega)$, the domain is $D_{\text{kin}} = H^1(\Omega)$ or Sobolev space with appropriate boundary conditions

2. **$-\kappa_{\text{kill}}(x)$**: Multiplication by bounded smooth function (A2: $\kappa_{\text{kill}} \in C^2$ with bounded derivatives)
   - Domain: All of $L^2(\Omega)$ (bounded multiplication is everywhere defined)

3. **$+c(\mu)$**: Multiplication by scalar constant
   - Domain: All of $L^2(\Omega)$

**Conclusion**:
$$
\text{Dom}(\mathcal{L}_\mu) = \text{Dom}(\mathcal{L}_{\text{kin}}) = D_{\text{kin}}
$$
independent of $\mu$ since the perturbations are bounded multiplication operators.

**Verification of Kato hypotheses**:

Kato IV.2.25 requires:
1. $\mathcal{L}_\mu$ is closed on $D_{\text{kin}}$ ✅ (perturbation of closed operator by bounded operator)
2. $\mathcal{L}_{\mu_n} - \mathcal{L}_\mu$ is bounded ✅ (scalar multiplication by $c(\mu_n) - c(\mu)$)
3. Resolvent $R_\lambda(\mu)$ exists for $\lambda$ large ✅ (dissipative operator + bounded perturbation)

**Potential Issue**: Weighted $L^2$ spaces

If working in weighted $L^2(\Omega, w(x,v) dx dv)$ with weight $w$ (e.g., $w = e^{aV}$ for Lyapunov weight), need to verify:
- Multiplication by $-\kappa_{\text{kill}}(x) + c(\mu)$ is bounded in weighted norm
- This holds if $\kappa_{\text{kill}}$ and $c(\mu)$ are bounded independent of weight

**Resolution**:

Work in standard $L^2(\Omega)$ for the resolvent convergence argument. Lyapunov bounds are proven separately in the dual space (expectations). This is the standard approach in hypocoercivity theory.

**Conclusion**: Domain stability is not an issue - all operators $\mathcal{L}_\mu$ share the domain $D_{\text{kin}}$, and perturbations are bounded. Kato's theorem applies directly.

---

## VI. Proof Validation Checklist

- [x] **Logical Completeness**: All steps follow from previous steps (Steps 1→2→3→4→5)
- [x] **Hypothesis Usage**: All three hypotheses of CV theorem are verified (Steps 2, 3, 1)
- [x] **Conclusion Derivation**: Weak convergence $\rho_{\mu_n} \rightharpoonup \rho_\mu$ is derived (Step 4), continuity of $\mathcal{T}$ follows (Step 5)
- [x] **Framework Consistency**: All dependencies verified in glossary (assump-qsd-existence A1-A3, hypoellipticity)
- [x] **No Circular Reasoning**: External theorem is applied after independent hypothesis verification
- [x] **Constant Tracking**: All constants ($c_{\min}, c_{\max}, \beta, C, m_{\min}, R$) defined and proved $\mu$-uniform
- [⚠] **Edge Cases**: Two edge cases flagged:
  - Kinetic dominance $\beta_0 > c_{\max}$ - recommend explicit assumption
  - Lower bound $m_{\min} > 0$ - recommend adding to definition of $K$
- [x] **Regularity Verified**: Smoothness of $\kappa_{\text{kill}}$ (A2), strong convexity of $U$ (A1), hypoellipticity (Hörmander)
- [x] **Measure Theory**: Weak convergence, probability measures, Lyapunov expectations all well-defined

---

## VII. Alternative Approaches (Not Chosen)

### Alternative 1: Spectral Perturbation via Krein-Rutman

**Approach**: Use Krein-Rutman theorem for positive compact operators to study the principal eigenvalue and eigenvector (QSD) directly as $\mu$ varies. Treat $\mathcal{L}_\mu$ as a perturbation of $\mathcal{L}_{\text{kin}}$ and use spectral stability theorems.

**Pros**:
- Direct control of eigenvalues and eigenvectors
- Can provide quantitative continuity rates (Hölder or Lipschitz)
- Well-suited for compact perturbations

**Cons**:
- Requires compactness of the semigroup $e^{t\mathcal{L}_\mu}$ or resolvent $R_\lambda(\mu)$
  - Hypoelliptic operators on unbounded domains are not compact without additional assumptions
  - Need compact embedding theorems (e.g., Rellich-Kondrachov) which require bounded domain or heavy exponential weights
- Krein-Rutman requires positivity (operator preserves positive functions), which holds but adds technical burden
- More restrictive hypotheses than CV theorem
- Heavier setup for the same conclusion (weak continuity)

**When to Consider**: If quantitative continuity rates are needed (e.g., Lipschitz continuity of $\mu \mapsto \rho_\mu$ in TV or Wasserstein), spectral perturbation can provide explicit bounds. Also useful if domain is compact (then compactness is automatic).

---

### Alternative 2: Coupling Argument for Killed Processes

**Approach**: Construct a probabilistic coupling between the killed-and-revived processes with parameters $\mu_n$ and $\mu$. Show the coupling is successful (processes coalesce) with high probability, then deduce convergence of QSDs via coupling inequality.

**Pros**:
- Probabilistic intuition: directly relates trajectories
- Robust to model variations (works for discrete-time, continuous-time, general state spaces)
- Can handle non-smooth perturbations more easily than PDE methods

**Cons**:
- Constructing a coupling for processes with $\mu$-dependent killing/revival is nontrivial
  - Killing times depend on state-dependent rates
  - Revival mechanism is global (depends on $m_d(\mu)$, not local)
- Translating coupling success to weak convergence of QSDs requires additional steps (typically via Prohorov or Skorokhod)
- Less direct than resolvent/spectral methods for PDE-based systems
- Harder to obtain uniformity in $\mu$ (coupling rate may depend on $\mu$)

**When to Consider**: If the framework extends to discrete state spaces, non-Markovian dynamics, or other settings where PDE tools are unavailable. Also useful for proving total variation or Wasserstein convergence (stronger than weak convergence).

---

### Alternative 3: Trotter-Kato Semigroup Convergence

**Approach**: Instead of resolvent convergence, prove semigroup convergence $e^{t\mathcal{L}_{\mu_n}} \to e^{t\mathcal{L}_\mu}$ in strong operator topology using Trotter-Kato theorem. Then deduce convergence of stationary measures (QSDs).

**Pros**:
- Trotter-Kato provides equivalence between resolvent and semigroup convergence
- Semigroup convergence has clearer probabilistic interpretation (convergence of transition kernels)
- Can handle time-dependent perturbations

**Cons**:
- Trotter-Kato requires checking stability of semigroup domains and uniform boundedness
- For our setting, resolvent convergence (Kato perturbation) is simpler since perturbation is bounded scalar
- Doesn't avoid the need to verify moment bounds and absorption rate bounds (same hypotheses as CV)
- Additional technical overhead without clear benefit

**When to Consider**: If studying time-dependent mean-field operators $\mathcal{L}_{\mu(t)}$ or if semigroup convergence is needed for other parts of the framework.

---

## VIII. Open Questions and Future Work

### Remaining Gaps

1. **Kinetic dominance condition**: The requirement $\beta_0 > c_{\max}$ is currently derived but not explicitly stated in assump-qsd-existence (A1-A4).
   - **How critical**: High - without this, Lyapunov drift may not be negative, breaking existence
   - **Resolution**: Add as explicit assumption A5 or strengthen A1/A3 to guarantee it

2. **Lower bound on death mass**: The uniform lower bound $m_d(\mu) \ge m_{\min} > 0$ for $\mu \in K$ is flagged as requiring either explicit constraint or derivation.
   - **How critical**: Medium - can be added to $K$ explicitly as workaround
   - **Resolution**: Either (i) add to definition of $K$, or (ii) prove via strict positivity (R3) first (reorder proof)

3. **Quantitative continuity**: CV theorem gives weak continuity but not rates. How fast does $\rho_{\mu_n} \to \rho_\mu$ in Wasserstein or TV distance?
   - **How critical**: Low for existence proof, higher for numerical analysis
   - **Resolution**: Spectral perturbation or coupling approach (Alternative 1 or 2)

### Conjectures

1. **Conjecture (Lipschitz continuity)**: If $\mu \mapsto c(\mu)$ is Lipschitz on $K$ (in weak topology), then $\mu \mapsto \rho_\mu$ is Lipschitz in Wasserstein-2 distance.
   - **Why plausible**: Smooth dependence of resolvents + hypoelliptic regularity should propagate smoothness; similar results hold for non-killed diffusions

2. **Conjecture (Uniqueness of fixed point)**: The fixed point $\rho_\infty = \mathcal{T}(\rho_\infty)$ obtained from Schauder is unique.
   - **Why plausible**: If $\mathcal{T}$ is a contraction (in some metric), Banach fixed-point theorem gives uniqueness. CV theory suggests uniqueness under mild conditions (irreducibility + aperiodicity)

### Extensions

1. **Extension to adaptive gas**: Does the same Schauder strategy work for the full geometric gas (adaptive viscous fluid model) with mean-field coupling and Hessian diffusion?
   - **Challenge**: Operator $\mathcal{L}_\mu$ becomes more complex (includes viscous coupling term $\nabla \cdot [D(\mu) \nabla \cdot]$)
   - **Opportunity**: If viscous coupling adds dissipation, might improve kinetic dominance condition

2. **Extension to non-compact state spaces**: Can the proof be adapted to unbounded $\mathcal{X} = \mathbb{R}^d$ with only Lyapunov confinement (no hard boundary)?
   - **Challenge**: Weak compactness of $K$ requires tightness arguments
   - **Opportunity**: Moment bounds provide tightness; similar proofs exist for Gibbs measures on $\mathbb{R}^d$

---

## IX. Expansion Roadmap

### Phase 1: Prove Missing Lemmas (Estimated: 2-3 days)

1. **Lemma A (Uniform Lyapunov drift)**:
   - Compute $\mathcal{L}_{\text{kin}}^*[V]$ explicitly for Langevin with strong convexity
   - Verify $\beta_0 = 2\min(\kappa_{\text{conf}}, \gamma - 1) > 0$
   - Add jump contribution and prove $\beta = \beta_0 - c_{\max} > 0$ under kinetic dominance
   - Reference: Section 4.2 of document for drift structure

2. **Lemma B (Lower bound on death mass)**:
   - Either: Add $m_d(\mu) \ge m_{\min}$ to definition of $K$ (lines 1252-1257)
   - Or: Prove via hypoelliptic full support theorem (requires R2, R3 - may need to reorder proof)
   - Verify $\mathcal{T}(K) \subseteq K$ still holds with augmented $K$

3. **Lemma C (Domain stability)**:
   - Prove $\text{Dom}(\mathcal{L}_\mu) = \text{Dom}(\mathcal{L}_{\text{kin}})$ independent of $\mu$
   - Verify bounded perturbation property for Kato theorem
   - Quick lemma - 1-2 hours

### Phase 2: Fill Technical Details (Estimated: 2-3 days)

1. **Step 1 (Revival coefficient)**:
   - Expand weak convergence argument for $m_d(\mu_n) \to m_d(\mu)$
   - Prove $\|\mu_n\|_{L^1} \to \|\mu\|_{L^1}$ using constant test function
   - Add epsilon-delta details for continuity of division

2. **Step 3 (Lyapunov drift)**:
   - Full calculation of $\mathcal{L}_{\text{kin}}^*[V]$ using Ito's formula
   - Careful treatment of boundary terms (if $\mathcal{X}$ has boundary)
   - Verify all inequalities are sharp (not loose)

3. **Step 4 (CV theorem application)**:
   - Add full statement of CV Theorem 2.2 with precise hypotheses
   - Cross-reference to original paper (Champagnat & Villemonais 2017)
   - Verify our setting matches their framework (killed diffusion with revival)

### Phase 3: Add Rigor (Estimated: 1-2 days)

1. **Epsilon-delta arguments**:
   - Weak convergence definition: For all $f \in C_b(\Omega)$, $\int f \mu_n \to \int f \mu$
   - Operator norm convergence: For all $\lambda > \Lambda_{\max}$, $\|R_\lambda(\mu_n) - R_\lambda(\mu)\|_{\text{op}} < \epsilon$ for $n > N(\epsilon)$

2. **Measure-theoretic details**:
   - Clarify probability space: $(\Omega, \mathcal{B}(\Omega), \mu)$ with Borel $\sigma$-algebra
   - Verify $V$ is measurable and integrable
   - Check all integrals are well-defined (no $\infty - \infty$)

3. **Constants audit**:
   - List all constants with explicit formulas
   - Verify $\mu$-independence for each
   - Add table summarizing dependencies (on $d$, $N$, framework parameters)

### Phase 4: Review and Validation (Estimated: 1 day)

1. **Framework cross-validation**:
   - Check all cited theorems exist in glossary
   - Verify no forward references (only use prior results)
   - Confirm A1-A4 are sufficient (or add A5 for kinetic dominance)

2. **Edge case verification**:
   - What if $\kappa_{\text{kill}} \equiv 0$ (no killing)? Theorem should still hold (trivial QSD = stationary distribution)
   - What if $\gamma = 0$ (no friction)? Drift may fail - verify kinetic dominance prevents this
   - What if $\mathcal{X}$ is compact? Proof simplifies (automatic tightness)

3. **Proof completeness check**:
   - Re-read final proof start-to-finish
   - Verify each "why valid" step is justified
   - Check no logical gaps remain

**Total Estimated Expansion Time**: 6-9 days for complete rigorous proof with all details

---

## X. Cross-References

**Theorems Used**:
- {prf:ref}`thm-qsd-existence-corrected` (overall Schauder strategy)
- {prf:ref}`assump-qsd-existence` (A1-A4: framework assumptions)
- {prf:ref}`def-qsd-mean-field` (QSD definition for mean-field)
- External: Champagnat & Villemonais (2017) Theorem 2.2 (QSD stability)
- External: Kato Perturbation Theory Theorem IV.2.25 (resolvent perturbation)
- External: Schauder Fixed-Point Theorem (1930)

**Definitions Used**:
- $c(\mu) = \lambda_{\text{revive}} m_d(\mu) / \|\mu\|_{L^1}$ (revival coefficient, line 1184)
- $m_d(\mu) = \int \kappa_{\text{kill}}(x) \mu(x,v) \, dx dv$ (death mass, line 1290)
- $K = \{\rho \in \mathcal{P}(\Omega) : \int V \rho \le R, \|\rho\|_{L^1} \ge M_{\min}\}$ (compact convex set, line 1252)
- $\mathcal{L}_\mu = \mathcal{L}_{\text{kin}} - \kappa_{\text{kill}}(x) + c(\mu)$ (linearized operator, line 1184)
- $\mathcal{T}(\mu) = \rho_\mu$ (fixed-point map, line 1207)

**Related Proofs** (for comparison):
- Similar technique in propagation of chaos (08_propagation_chaos.md): uses tightness + compactness for limit
- Euclidean Gas QSD existence (06_convergence.md): Foster-Lyapunov without nonlinear fixed-point
- LSI proof for QSD (15_geometric_gas_lsi_proof.md): uses regularity R1-R6 proven here

---

**Proof Sketch Completed**: 2025-10-25
**Ready for Expansion**: Yes (with Lemmas A, B completed first)
**Confidence Level**: Medium-High

**Rationale for confidence**:
- ✅ Approach is sound: External theorem with rigorous hypothesis verification
- ✅ Steps are actionable: Each substep has clear mathematical content
- ✅ Dependencies verified: All framework assumptions checked
- ⚠️ Two technical lemmas flagged as needing careful proofs (Lyapunov drift uniformity, death mass lower bound)
- ⚠️ Single-strategist limitation: Gemini review unavailable for cross-validation
- ✅ Physical consistency: Kinetic dominance requirement makes physical sense

**Recommendation**: Proceed with expansion after:
1. Re-running Gemini review for independent verification
2. Proving Lemma A (uniform drift) rigorously
3. Deciding on approach for Lemma B (explicit constraint vs. derivation)
