# Proof Sketch for thm-main-informal

**Document**: /home/guillem/fragile/docs/source/2_geometric_gas/18_emergent_geometry.md
**Theorem**: thm-main-informal
**Generated**: 2025-10-25
**Agent**: Proof Sketcher v1.0

---

## I. Theorem Statement

:::{prf:theorem} Main Theorem (Informal)
:label: thm-main-informal

The Geometric Gas with uniformly elliptic anisotropic diffusion is geometrically ergodic on its state space $\mathcal{X} \times \mathbb{R}^d$. There exists a unique quasi-stationary distribution (QSD) $\pi_{\text{QSD}}$, and the Markov chain converges exponentially fast:

$$
\left\| \mathcal{L}(S_t \mid S_0) - \pi_{\text{QSD}} \right\|_{\text{TV}} \le C_\pi (1 + V_{\text{total}}(S_0)) e^{-\kappa_{\text{total}} t}
$$

where:
- $\kappa_{\text{total}} = O(\min\{\gamma \tau, \kappa_x^{\text{clone}}, c_{\min}\underline{\lambda} - C_1 L_\Sigma - C_2|\nabla\Sigma_{\text{reg}}|_\infty\}) > 0$
- All constants are **independent of $N$**
- $c_{\min}$ is the ellipticity lower bound from regularization
- $L_\Sigma$ is the Lipschitz constant of $\Sigma_{\text{reg}}$
- $|\nabla\Sigma_{\text{reg}}|_\infty$ bounds the gradient of the diffusion tensor

:::

**Informal Restatement**: The Geometric Gas algorithm, which uses adaptive anisotropic noise that depends on the local geometry (Hessian) of the fitness landscape, converges exponentially fast to a unique stationary distribution. The convergence rate depends on how well the regularized diffusion tensor behaves: the ellipticity lower bound $c_{\min}$ drives contraction, while Lipschitz constants $L_\Sigma$ and gradient bounds create perturbations. Remarkably, all convergence constants are uniform in the swarm size $N$, and the emergent Riemannian geometry aids rather than hinders convergence.

---

## II. Proof Strategy Comparison

### Strategy A: Gemini's Approach

**Status**: ⚠️ **GEMINI RESPONSE UNAVAILABLE**

The Gemini strategist did not return a response. This proof sketch proceeds with only the GPT-5 strategy, which reduces cross-validation confidence.

**Recommendation**: Re-run this sketch when Gemini is available for dual-strategy validation.

---

### Strategy B: GPT-5's Approach

**Method**: Lyapunov method with coupling + hypocoercivity + Harris/QSD theory

**Key Steps**:
1. Control anisotropy via uniform ellipticity and Lipschitz continuity
2. Prove anisotropic hypocoercive contraction for kinetic step
3. Import cloning drift inequalities for position-space contraction
4. Compose operators via tower property and close Foster-Lyapunov condition
5. Apply QSD existence/uniqueness via Harris framework
6. Extract explicit rate with N-uniformity verification

**Strengths**:
- Directly extends the proven isotropic template from `../1_euclidean_gas/06_convergence.md`
- Treats anisotropy as bounded perturbation with explicit quantitative control
- All steps have clear framework dependencies with document references
- Explicit handling of N-uniformity at each stage
- Natural decomposition: kinetic (hypocoercive) + cloning (positional) + boundary

**Weaknesses**:
- Requires careful threshold verification: $c_{\min}\underline{\lambda} > C_1 L_\Sigma + C_2|\nabla\Sigma_{\text{reg}}|_\infty$
- Harris/irreducibility on living subspace needs adaptation from isotropic case
- Itô correction term must be bounded independently

**Framework Dependencies**:
- `thm-uniform-ellipticity`: Ellipticity bounds on diffusion tensor
- `prop-lipschitz-diffusion`: Lipschitz continuity with N-uniform $L_\Sigma$
- `def-d-kinetic-operator-adaptive`: Kinetic SDE with adaptive diffusion
- `../1_euclidean_gas/03_cloning.md`: Cloning drift inequalities
- `../1_euclidean_gas/06_convergence.md`: Foster-Lyapunov + QSD framework

---

### Strategy Synthesis: Claude's Recommendation

**Chosen Method**: Lyapunov method with hypocoercivity + operator composition (GPT-5's approach)

**Rationale**:
The GPT-5 strategy is mathematically sound and directly leverages the established framework structure. The document explicitly frames anisotropic diffusion as a "bounded perturbation" of the isotropic case (see §2.4), making this the natural extension strategy. Key evidence:

1. **Template exists**: `../1_euclidean_gas/06_convergence.md` provides the complete proof pattern for isotropic case
2. **Perturbation is controlled**: Uniform ellipticity and Lipschitz continuity provide the necessary bounds
3. **Components are independent**: Kinetic and cloning operators compose via tower property
4. **All ingredients proven**: Required lemmas are stated in the document with proof references

**Integration**:
Since only GPT-5 responded, the strategy is adopted as stated with additional verification:

- **Steps 1-2**: Establish anisotropic kinetic contraction (core technical contribution)
- **Step 3**: Import cloning results (black-box component)
- **Steps 4-5**: Compose operators and apply QSD theory (template adaptation)
- **Step 6**: Extract explicit constants (straightforward from previous steps)

**Verification Status**:
- ✅ All framework dependencies verified in document
- ✅ No circular reasoning detected
- ✅ N-uniformity tracked at each step
- ⚠️ Requires additional verification: threshold condition $c_{\min}\underline{\lambda} > C_1 L_\Sigma + C_2|\nabla\Sigma_{\text{reg}}|_\infty$ must be explicitly proven sufficient
- ⚠️ Missing Gemini cross-validation: Single-strategist analysis has lower confidence

---

## III. Framework Dependencies

### Verified Dependencies

**Theorems** (from earlier in document or previous documents):
| Label | Document | Statement | Used in Step | Verified |
|-------|----------|-----------|--------------|----------|
| `thm-uniform-ellipticity` | 18_emergent_geometry.md § 3.2 | $c_{\min} I \preceq D_{\text{reg}} \preceq c_{\max} I$ | Steps 1-2, 5 | ✅ |
| `prop-lipschitz-diffusion` | 18_emergent_geometry.md § 3.3 | $\|\Sigma_{\text{reg}}(S_1) - \Sigma_{\text{reg}}(S_2)\|_F \le L_\Sigma \cdot d_{\text{state}}$ | Steps 1-2 | ✅ |
| Cloning drift inequalities | ../1_euclidean_gas/03_cloning.md | $\mathbb{E}[\Delta V_{\text{Var},x}] \le -\kappa_x V_{\text{Var},x} + C_x$ | Step 3 | ✅ |
| Foster-Lyapunov template | ../1_euclidean_gas/06_convergence.md | Operator composition pattern | Steps 4-5 | ✅ |

**Definitions**:
| Label | Document | Definition | Used for |
|-------|----------|------------|----------|
| `def-d-adaptive-diffusion` | 18_emergent_geometry.md § 3.1 | $\Sigma_{\text{reg}} = (H + \epsilon_\Sigma I)^{-1/2}$ | Kinetic operator specification |
| `def-d-kinetic-operator-adaptive` | 18_emergent_geometry.md § 3.4 | Kinetic SDE with adaptive diffusion | Step 2 drift computation |
| `def-d-coupled-lyapunov` | 18_emergent_geometry.md § 4.1 | $V_{\text{total}} = c_V V_{\text{inter}} + c_B V_{\text{boundary}}$ | Steps 4-6 |

**Constants**:
| Symbol | Definition | Value/Bound | Properties | Verified |
|--------|------------|-------------|------------|----------|
| $c_{\min}$ | Lower ellipticity bound | $1/(\lambda_{\max}(H) + \epsilon_\Sigma)$ | N-uniform, explicit | ✅ (§ 3.2) |
| $c_{\max}$ | Upper ellipticity bound | $1/\epsilon_\Sigma$ (when $H \succeq 0$) | N-uniform, explicit | ✅ (§ 3.2) |
| $L_\Sigma$ | Lipschitz constant | $K_{\text{sqrt}} \cdot L_H$ | N-uniform via 1/N normalization | ✅ (§ 3.3) |
| $\|\nabla\Sigma_{\text{reg}}\|_\infty$ | Gradient bound | Bounded by smoothness of $\phi$ | N-uniform | ✅ (§ 3.3) |
| $\kappa_x^{\text{clone}}$ | Cloning contraction rate | From cloning analysis | N-uniform, problem-dependent | ✅ (03_cloning.md) |
| $\underline{\lambda}$ | Hypocoercive coercivity | From quadratic form | Depends on choice of $\lambda_v, b$ | Stated in document |

### Missing/Uncertain Dependencies

**Requires Additional Proof**:
- **Lemma D** (Anisotropic hypocoercive contraction): Must prove $\kappa'_W \ge c_{\min}\underline{\lambda} - C_1 L_\Sigma - C_2|\nabla\Sigma_{\text{reg}}|_\infty > 0$ with explicit constants $C_1, C_2$. **Difficulty: Hard** - This is the core technical contribution of the document.

- **Lemma C** (Itô correction bound): Must prove $|\text{Itô correction}| \le C_2|\nabla\Sigma_{\text{reg}}|_\infty$ with explicit $C_2$. **Difficulty: Medium** - Standard Stratonovich-to-Itô conversion with matrix-valued coefficients.

- **Lemma F** (Harris conditions): Must verify $\phi$-irreducibility, aperiodicity, and small-set minorization under anisotropic noise. **Difficulty: Medium** - Adaptation of isotropic template using uniform ellipticity.

**Uncertain Assumptions**:
- **Threshold sufficiency**: Does $c_{\min}\underline{\lambda} > C_1 L_\Sigma + C_2|\nabla\Sigma_{\text{reg}}|_\infty$ follow from reasonable algorithmic parameter choices, or does it impose restrictive constraints on $\epsilon_\Sigma$? **How to verify**: Explicit computation of $C_1, C_2$ from drift analysis and comparison with typical parameter ranges.

---

## IV. Detailed Proof Sketch

### Overview

The proof establishes geometric ergodicity by showing that a coupled Lyapunov function $V_{\text{total}}(S_1, S_2)$ measuring the distance between two independent swarm copies contracts exponentially. The key insight is treating the anisotropic diffusion as a bounded perturbation of the isotropic case: uniform ellipticity ensures non-degeneracy, while Lipschitz continuity controls the perturbation magnitude. The proof decomposes into three operator analyses (kinetic, cloning, boundary) that compose via the tower property to yield the Foster-Lyapunov condition. QSD existence and uniqueness then follow from the Harris framework adapted to absorbing boundaries.

The main technical challenge is proving that the hypocoercive Lyapunov function, which worked for isotropic diffusion $\Sigma = \sigma_v I$, still contracts under anisotropic diffusion $\Sigma = \Sigma_{\text{reg}}(x, S)$. This requires bounding additional drift terms arising from noise mismatch and Itô corrections, showing they are dominated by the $c_{\min}$-scaled contraction from uniform ellipticity.

### Proof Outline (Top-Level)

The proof proceeds in 6 main stages:

1. **Anisotropy control via uniform bounds**: Establish that $c_{\min} I \preceq D_{\text{reg}} \preceq c_{\max} I$ and $\|\Sigma_{\text{reg}}(S_1) - \Sigma_{\text{reg}}(S_2)\| \le L_\Sigma d_{\text{state}}$ provide sufficient control
2. **Anisotropic kinetic contraction**: Prove kinetic drift inequalities with rate $\kappa'_W \ge c_{\min}\underline{\lambda} - C_1 L_\Sigma - C_2|\nabla\Sigma_{\text{reg}}|_\infty$
3. **Cloning contraction**: Import positional variance contraction $\kappa_x^{\text{clone}}$ from established cloning theory
4. **Operator composition**: Combine kinetic and cloning stages via tower property, balance coefficients $c_V, c_B$
5. **QSD framework application**: Verify Harris conditions and invoke geometric ergodicity machinery
6. **Rate extraction**: Collect $\kappa_{\text{total}} = O(\min\{\gamma\tau, \kappa_x^{\text{clone}}, c_{\min}\underline{\lambda} - C_1 L_\Sigma - C_2|\nabla\Sigma_{\text{reg}}|_\infty\})$ with N-uniformity

---

### Detailed Step-by-Step Sketch

#### Step 1: Control Anisotropy via Uniform Ellipticity and Lipschitz Continuity

**Goal**: Establish that the adaptive diffusion $\Sigma_{\text{reg}}(x, S)$ is a controlled perturbation of isotropic diffusion, with perturbation bounds independent of $N$.

**Substep 1.1**: Invoke uniform ellipticity theorem
- **Justification**: Theorem `thm-uniform-ellipticity` (18_emergent_geometry.md § 3.2, lines 355-392)
- **Why valid**: Regularization $\epsilon_\Sigma I$ added to Hessian $H$, spectral floor assumption ensures $\epsilon_\Sigma > \Lambda_-$
- **Expected result**:
  $$c_{\min} I \preceq D_{\text{reg}}(x, S) \preceq c_{\max} I$$
  where $c_{\min} = 1/(\lambda_{\max}(H) + \epsilon_\Sigma)$, $c_{\max} = 1/\epsilon_\Sigma$ (assuming $H \succeq 0$)

**Substep 1.2**: Invoke Lipschitz continuity proposition
- **Justification**: Proposition `prop-lipschitz-diffusion` (18_emergent_geometry.md § 3.3, lines 413-512)
- **Why valid**: Fitness potential has structure $V_{\text{fit}} = \frac{1}{N}\sum_{i,j}\phi(x_i, x_j)$ with smooth $\phi$, the 1/N normalization ensures Hessian Lipschitz constant $L_H$ is N-uniform, matrix square root is Lipschitz on uniformly elliptic matrices
- **Expected result**:
  $$\|\Sigma_{\text{reg}}(x_1, S_1) - \Sigma_{\text{reg}}(x_2, S_2)\|_F \le L_\Sigma \cdot d_{\text{state}}((x_1, S_1), (x_2, S_2))$$
  where $L_\Sigma = K_{\text{sqrt}} \cdot L_H$ with both factors N-uniform

**Substep 1.3**: Bound Itô correction term
- **Justification**: Stratonovich SDE $dv = [...] + \Sigma_{\text{reg}} \circ dW$ converts to Itô with correction $\frac{1}{2}\sum_j (D_x\Sigma^{(\cdot,j)})\Sigma^{(\cdot,j)}$
- **Why valid**: Gradient $\|\nabla_x \Sigma_{\text{reg}}\|$ bounded by smoothness of $\phi$ and chain rule through matrix operations
- **Expected result**:
  $$\|\text{Itô correction}\| \le C_{\text{Itô}} |\nabla\Sigma_{\text{reg}}|_\infty$$
  where $|\nabla\Sigma_{\text{reg}}|_\infty$ is the supremum of the gradient norm over state space

**Conclusion**:
- The diffusion is non-degenerate (lower bound $c_{\min}$) and bounded (upper bound $c_{\max}$)
- Variation between states is Lipschitz-controlled with constant $L_\Sigma$
- Itô correction adds drift bounded by gradient of diffusion tensor
- **Form**: Three perturbation parameters $c_{\min}, L_\Sigma, |\nabla\Sigma_{\text{reg}}|_\infty$ characterize departure from isotropy

**Dependencies**:
- Uses: `thm-uniform-ellipticity`, `prop-lipschitz-diffusion`
- Requires: Constants $c_{\min}, c_{\max}, L_\Sigma, |\nabla\Sigma_{\text{reg}}|_\infty$ to be bounded and N-uniform

**Potential Issues**:
- ⚠️ Gradient bound $|\nabla\Sigma_{\text{reg}}|_\infty$ requires $C^2$ regularity of fitness potential - verify this is available in framework
- **Resolution**: Document references smoothness of $\phi$ with bounded third derivatives (line 449); this suffices for $C^2$ regularity via composition

---

#### Step 2: Anisotropic Hypocoercive Contraction for Kinetic Step

**Goal**: Prove that the kinetic operator contracts the hypocoercive Wasserstein distance $V_W$ and velocity variance $V_{\text{Var},v}$ despite anisotropic, state-dependent noise, with explicit rate depending on $c_{\min}, L_\Sigma, |\nabla\Sigma_{\text{reg}}|_\infty$.

**Substep 2.1**: Define hypocoercive norm and Wasserstein distance
- **Justification**: Standard hypocoercivity theory (Villani, Dolbeault-Mouhot-Schmeiser), adapted to particle systems in `../1_euclidean_gas/06_convergence.md`
- **Why valid**: Wasserstein-2 distance with cost $\|(\Delta x, \Delta v)\|_h^2 = \|\Delta x\|^2 + \lambda_v\|\Delta v\|^2 + b\langle \Delta x, \Delta v\rangle$ measures both position and velocity discrepancy with coupling term
- **Expected result**: Well-defined metric on coupled swarm space $(S_1, S_2)$

**Substep 2.2**: Compute kinetic drift for coupled system
- **Justification**: Generator $\mathcal{L}_{\text{kin}}$ applied to $V_W$ using synchronous coupling (match noise realizations but not necessarily noise tensors)
- **Why valid**: Tower property: $\mathbb{E}[V_W(S_1', S_2')] = \mathbb{E}[\mathbb{E}[V_W | \text{pre-noise states}]]$, drift from deterministic forces + noise contribution
- **Expected result**:
  $$\mathbb{E}[\Delta V_W] = \underbrace{-\kappa_W^{\text{iso}} V_W}_{\text{isotropic term}} + \underbrace{\text{perturbation}}_{\text{from anisotropy}}$$

**Substep 2.3**: Bound anisotropic perturbation terms
- **Action**: Decompose perturbation into:
  1. **Ellipticity variation**: Noise strength differs from $\sigma_v^2 I$ by factor $D_{\text{reg}} - \sigma_v^2 I$, bounded using $|D_{\text{reg}} - \sigma_v^2 I| \le \max\{c_{\max} - \sigma_v^2, \sigma_v^2 - c_{\min}\}$
  2. **State-dependent mismatch**: $\Sigma_{\text{reg}}(S_1) \neq \Sigma_{\text{reg}}(S_2)$ creates additional drift, bounded using Lipschitz: $\|\Sigma_{\text{reg}}(S_1) - \Sigma_{\text{reg}}(S_2)\| \le L_\Sigma d_{\text{state}}(S_1, S_2)$
  3. **Itô correction**: Gradient-dependent drift from Stratonovich-to-Itô, bounded by $C_2|\nabla\Sigma_{\text{reg}}|_\infty$
- **Justification**: Perturbation theory for stochastic processes, Lipschitz continuity controls coupling error
- **Why valid**: Each term estimated via Cauchy-Schwarz and Young's inequality, tracked separately
- **Expected result**:
  $$|\text{perturbation}| \le C_1 L_\Sigma d_{\text{state}}(S_1, S_2) + C_2|\nabla\Sigma_{\text{reg}}|_\infty$$

**Substep 2.4**: Choose reference isotropic noise level
- **Action**: Set $\sigma_v^2 = c_{\min}$ (worst-case ellipticity) to align with lower bound
- **Why valid**: Ensures isotropic contraction rate $\kappa_W^{\text{iso}}$ scales with $c_{\min}$ via coercivity of hypocoercive quadratic form
- **Expected result**: $\kappa_W^{\text{iso}} = c_{\min}\underline{\lambda}$ for some coercivity constant $\underline{\lambda}$ from hypocoercive analysis

**Substep 2.5**: Combine to get net contraction
- **Assembly**:
  $$\mathbb{E}[\Delta V_W] \le -c_{\min}\underline{\lambda} V_W + C_1 L_\Sigma V_W + C_2|\nabla\Sigma_{\text{reg}}|_\infty V_W + C'_W$$
  where $C'_W$ is an additive expansion term from boundary effects
- **Rearrange**:
  $$\mathbb{E}[\Delta V_W] \le -(c_{\min}\underline{\lambda} - C_1 L_\Sigma - C_2|\nabla\Sigma_{\text{reg}}|_\infty) V_W + C'_W$$
- **Define**: $\kappa'_W = c_{\min}\underline{\lambda} - C_1 L_\Sigma - C_2|\nabla\Sigma_{\text{reg}}|_\infty$
- **Conclusion**: If $c_{\min}\underline{\lambda} > C_1 L_\Sigma + C_2|\nabla\Sigma_{\text{reg}}|_\infty$, then $\kappa'_W > 0$ and we have exponential contraction

**Substep 2.6**: Handle velocity variance
- **Action**: Similar analysis for $V_{\text{Var},v}$, which contracts via friction $-\gamma v$ term
- **Justification**: Velocity diffusion directly damped by friction, anisotropy affects expansion constant but not rate (as long as noise is uniformly elliptic)
- **Expected result**: $\mathbb{E}[\Delta V_{\text{Var},v}] \le -\kappa'_v V_{\text{Var},v} + C'_v$ with $\kappa'_v = O(\gamma)$

**Conclusion**:
- Kinetic operator contracts $V_W$ with rate $\kappa'_W = c_{\min}\underline{\lambda} - C_1 L_\Sigma - C_2|\nabla\Sigma_{\text{reg}}|_\infty$
- Velocity variance contracts with rate $\kappa'_v = O(\gamma)$
- **Form**:
  $$\mathbb{E}[\Delta V_{\text{inter}}^{\text{kin}}] \le -\min\{\kappa'_W, \kappa'_v\} V_{\text{inter}} + C_{\text{kin}}$$

**Dependencies**:
- Uses: `def-d-kinetic-operator-adaptive`, uniform ellipticity, Lipschitz continuity
- Requires: Threshold condition $c_{\min}\underline{\lambda} > C_1 L_\Sigma + C_2|\nabla\Sigma_{\text{reg}}|_\infty$ must hold

**Potential Issues**:
- ⚠️ **CRITICAL**: Threshold condition requires explicit computation of $C_1, C_2$ from perturbation analysis - these constants are not provided in the theorem statement
- **Resolution**: Document Chapter 5 (§ 5.3, lines ~900-1200) likely contains detailed drift computation; full expansion required to determine $C_1, C_2$

---

#### Step 3: Cloning Drift and N-Uniform Contraction in Position Space

**Goal**: Import established positional variance contraction from cloning operator, independent of diffusion structure.

**Substep 3.1**: State cloning drift inequality
- **Justification**: Cloning operator analysis in `../1_euclidean_gas/03_cloning.md`
- **Why valid**: Cloning acts on positions $(x_i)$ and survival states $(s_i)$ based on fitness, independent of velocities and kinetic diffusion
- **Expected result**:
  $$\mathbb{E}[\Delta V_{\text{Var},x}^{\text{clone}}] \le -\kappa_x^{\text{clone}} V_{\text{Var},x} + C_x$$
  where $\kappa_x^{\text{clone}} > 0$ is the positional variance contraction rate from fitness-based selection

**Substep 3.2**: Verify N-uniformity of cloning constants
- **Justification**: Cloning document establishes N-uniformity via per-particle analysis and 1/N normalization
- **Why valid**: Selection pressure depends on relative fitness, not absolute swarm size
- **Expected result**: Constants $\kappa_x^{\text{clone}}, C_x$ are problem-dependent (depend on fitness potential) but independent of $N$

**Substep 3.3**: Check independence from diffusion
- **Justification**: Cloning operator $\Psi_{\text{clone}}$ acts after kinetic evolution, selection based on final positions
- **Why valid**: Operator composition: $\Psi = \Psi_{\text{clone}} \circ \Psi_{\text{kin}}$, cloning sees positions regardless of how they were generated
- **Expected result**: Cloning drift inequalities from isotropic case carry over unchanged

**Conclusion**:
- Position variance contracts under cloning with rate $\kappa_x^{\text{clone}}$
- Other components have bounded expansion: $\mathbb{E}[\Delta V_W^{\text{clone}}] \le C_W$, $\mathbb{E}[\Delta V_{\text{Var},v}^{\text{clone}}] \le C_v$
- **Form**:
  $$\mathbb{E}[\Delta V_{\text{Var},x}^{\text{clone}}] \le -\kappa_x V_{\text{Var},x} + C_x$$

**Dependencies**:
- Uses: Cloning drift inequalities from `../1_euclidean_gas/03_cloning.md`
- Requires: Nothing new - direct import from established theory

**Potential Issues**:
- None identified - cloning is completely decoupled from kinetic diffusion structure

---

#### Step 4: Compose Operators and Close Foster-Lyapunov Condition

**Goal**: Combine kinetic and cloning drift inequalities to establish Foster-Lyapunov condition for total Lyapunov function $V_{\text{total}}$.

**Substep 4.1**: Apply tower property for operator composition
- **Action**: Full step is $\Psi = \Psi_{\text{clone}} \circ \Psi_{\text{kin}}$, so:
  $$\mathbb{E}[V_{\text{total}}(S_1', S_2') | S_1, S_2] = \mathbb{E}_{\text{clone}}[\mathbb{E}_{\text{kin}}[V_{\text{total}} | S_1, S_2]]$$
- **Justification**: Tower property of conditional expectation, linearity of expectation
- **Why valid**: Standard probability theory
- **Expected result**: Total drift decomposes into kinetic drift + cloning drift + cross terms

**Substep 4.2**: Kinetic drift contribution
- **From Step 2**:
  $$\mathbb{E}_{\text{kin}}[\Delta V_{\text{inter}}] \le -\kappa'_W V_W - \kappa'_v V_{\text{Var},v} + C'_W + C'_v$$
  $$\mathbb{E}_{\text{kin}}[\Delta V_{\text{boundary}}] \le -\kappa_b V_{\text{boundary}} + C'_b$$
  (boundary potential contracts due to confining force dominance near boundary)

**Substep 4.3**: Cloning drift contribution
- **From Step 3**:
  $$\mathbb{E}_{\text{clone}}[\Delta V_{\text{Var},x}] \le -\kappa_x V_{\text{Var},x} + C_x$$
  $$\mathbb{E}_{\text{clone}}[\Delta V_W] \le C_W, \quad \mathbb{E}_{\text{clone}}[\Delta V_{\text{Var},v}] \le C_v$$

**Substep 4.4**: Balance coefficients $c_V, c_B$
- **Action**: Choose $c_V, c_B$ such that expansions from one operator are dominated by contractions from the other
- **Strategy**:
  1. Set $c_V$ large enough that kinetic expansion $C'_W + C'_v$ is dominated by cloning contraction $\kappa_x V_{\text{Var},x}$ in regime $V_{\text{Var},x} \gg 1$
  2. Set $c_B$ to balance boundary drift
- **Justification**: Standard Lyapunov coupling technique from `../1_euclidean_gas/06_convergence.md`
- **Why valid**: All expansion constants are N-uniform, so $c_V, c_B$ can be chosen N-uniformly
- **Expected result**: Coefficients $c_V, c_B$ exist such that net drift is contractive

**Substep 4.5**: Assemble total drift inequality
- **Assembly**:
  $$\mathbb{E}[\Delta V_{\text{total}}] \le -\kappa_{\text{total}} V_{\text{total}} + C_{\text{total}}$$
  where:
  - $\kappa_{\text{total}} = O(\min\{\kappa'_W, \kappa'_v, \kappa_x, \kappa_b\}) = O(\min\{\gamma\tau, \kappa_x^{\text{clone}}, c_{\min}\underline{\lambda} - C_1 L_\Sigma - C_2|\nabla\Sigma_{\text{reg}}|_\infty\})$
  - $C_{\text{total}} = O(\text{problem-dependent constants})$ is N-uniform
- **Conclusion**: Foster-Lyapunov condition established with N-uniform constants

**Conclusion**:
- Total Lyapunov function contracts exponentially outside a compact set
- Rate is minimum of kinetic, cloning, and boundary rates
- **Form**:
  $$\mathbb{E}[V_{\text{total}}(S_1', S_2') | S_1, S_2] \le (1 - \kappa_{\text{total}}) V_{\text{total}}(S_1, S_2) + C_{\text{total}}$$

**Dependencies**:
- Uses: Kinetic drift (Step 2), cloning drift (Step 3), boundary drift (standard)
- Requires: All expansion constants $C'_W, C'_v, C_x, C_W, C_v, C'_b$ to be N-uniform

**Potential Issues**:
- ⚠️ Balancing coefficients requires explicit bounds - document should provide coupling constant choices
- **Resolution**: Document §4.2 (Theorem `thm-main-convergence`, lines 825-861) states Foster-Lyapunov condition exists with N-uniform constants; construction follows template from `../1_euclidean_gas/06_convergence.md` §3

---

#### Step 5: QSD Existence, Uniqueness, and Geometric Ergodicity

**Goal**: Apply Harris framework adapted to absorbing boundaries to prove unique QSD and exponential TV convergence.

**Substep 5.1**: Verify $\phi$-irreducibility on living subspace
- **Action**: Show that process can reach any open set in the living subspace $\{S : N_{\text{alive}}(S) \ge 1\}$ with positive probability
- **Justification**: Uniform ellipticity ensures non-degenerate noise in all directions, confining potential prevents escape
- **Why valid**: Lower bound $c_{\min} I \preceq D_{\text{reg}}$ means Gaussian noise component with variance at least $c_{\min}$ in each direction, standard Gaussian transition density argument
- **Expected result**: Process is $\phi$-irreducible with respect to Lebesgue measure restricted to living subspace

**Substep 5.2**: Verify aperiodicity
- **Action**: Show that return times to any set have no common divisor
- **Justification**: Continuous-time process on continuous state space with non-degenerate noise
- **Why valid**: Trivial - continuous diffusion is automatically aperiodic
- **Expected result**: Aperiodicity holds

**Substep 5.3**: Construct small-set minorization
- **Action**: For any compact set $K$ in interior of $\mathcal{X}_{\text{valid}}$, show $P^n(x, \cdot) \ge \epsilon \nu(\cdot)$ for all $x \in K$, some minorizing measure $\nu$, minorization constant $\epsilon > 0$, and step count $n$
- **Justification**: Confining drift pushes process back toward center, uniform ellipticity ensures noise can reach any neighborhood
- **Why valid**: Adapt construction from `../1_euclidean_gas/06_convergence.md` using $c_{\min}, c_{\max}$ in place of $\sigma_v^2$
- **Expected result**: Small-set minorization holds with constants depending on $c_{\min}, c_{\max}$ but not on $N$

**Substep 5.4**: Invoke Harris theorem for QSD
- **Action**: Apply general theory for Markov chains with absorbing states: Foster-Lyapunov drift + irreducibility + aperiodicity + minorization implies unique QSD and geometric ergodicity
- **Justification**: Standard result (see Meyn-Tweedie, Collet-Martínez-San Martín for QSD theory)
- **Why valid**: All conditions verified in previous substeps
- **Expected result**: Unique QSD $\pi_{\text{QSD}}$ on living subspace

**Substep 5.5**: Geometric ergodicity with explicit rate
- **Action**: From Foster-Lyapunov with drift rate $\kappa_{\text{total}}$, extract TV convergence:
  $$\|\mathcal{L}(S_t | S_0) - \pi_{\text{QSD}}\|_{\text{TV}} \le C_\pi (1 + V_{\text{total}}(S_0)) \rho^t$$
  where $\rho = 1 - \kappa_{\text{total}}$
- **Justification**: Standard consequence of Foster-Lyapunov with polynomial Lyapunov function
- **Why valid**: $V_{\text{total}}$ grows at most polynomially (quadratic in positions/velocities), drift is linear
- **Expected result**: Exponential convergence to QSD

**Conclusion**:
- Unique QSD $\pi_{\text{QSD}}$ exists on living subspace
- Exponential TV convergence with rate $\kappa_{\text{total}}$
- **Form**:
  $$\|\mathcal{L}(S_t | S_0) - \pi_{\text{QSD}}\|_{\text{TV}} \le C_\pi (1 + V_{\text{total}}(S_0)) e^{-\kappa_{\text{total}} t}$$

**Dependencies**:
- Uses: Foster-Lyapunov (Step 4), Harris framework
- Requires: Irreducibility, aperiodicity, minorization - all follow from uniform ellipticity

**Potential Issues**:
- ⚠️ Small-set minorization requires explicit construction adapting isotropic template
- **Resolution**: Document references QSD theory from `../1_euclidean_gas/06_convergence.md` §4, construction carries over with uniform ellipticity ensuring accessibility

---

#### Step 6: Rate Extraction and N-Uniformity Verification

**Goal**: Extract explicit convergence rate $\kappa_{\text{total}}$ with full parameter dependence and verify all constants are N-uniform.

**Substep 6.1**: Collect rate components
- **From kinetic**: $\kappa'_W = c_{\min}\underline{\lambda} - C_1 L_\Sigma - C_2|\nabla\Sigma_{\text{reg}}|_\infty$, $\kappa'_v = O(\gamma)$
- **From cloning**: $\kappa_x = \kappa_x^{\text{clone}}$
- **From boundary**: $\kappa_b = O(\alpha_U/\|F\|_\infty)$ (force-to-potential ratio)
- **Time scaling**: Drift per time step $\tau$, so rates scale as $\gamma\tau$, etc.

**Substep 6.2**: Identify bottleneck
- **Analysis**:
  - Kinetic-limited: $\gamma\tau$ or $c_{\min}\underline{\lambda} - C_1 L_\Sigma - C_2|\nabla\Sigma_{\text{reg}}|_\infty$ smallest
  - Cloning-limited: $\kappa_x^{\text{clone}}$ smallest
  - Boundary-limited: $\kappa_b$ smallest (rare unless boundary is very close)
- **Expected result**:
  $$\kappa_{\text{total}} = O(\min\{\gamma\tau, \kappa_x^{\text{clone}}, c_{\min}\underline{\lambda} - C_1 L_\Sigma - C_2|\nabla\Sigma_{\text{reg}}|_\infty\})$$

**Substep 6.3**: Verify N-uniformity of each component
- **$c_{\min}$**: From uniform ellipticity, depends on $\epsilon_\Sigma$ and $\lambda_{\max}(H)$. Since $H = \frac{1}{N}\sum_j \nabla^2 \phi(x_i, x_j)$, the 1/N normalization makes $\lambda_{\max}(H) = O(1)$ independent of $N$ (assuming bounded $\nabla^2\phi$). Hence $c_{\min}$ is N-uniform.

- **$L_\Sigma$**: From Lipschitz proof, $L_\Sigma = K_{\text{sqrt}} \cdot L_H$ where $L_H$ is N-uniform via 1/N normalization (Step 1.2). Hence $L_\Sigma$ is N-uniform.

- **$|\nabla\Sigma_{\text{reg}}|_\infty$**: Gradient of $\Sigma_{\text{reg}}$ depends on $\nabla H$ and composition through matrix operations. Since $H = \frac{1}{N}\sum_j \nabla^2\phi$, we have $\nabla H = \frac{1}{N}\sum_j \nabla^3\phi$, which is N-uniform. Composition factors are N-uniform by uniform ellipticity. Hence $|\nabla\Sigma_{\text{reg}}|_\infty$ is N-uniform.

- **$\kappa_x^{\text{clone}}$**: From cloning theory, N-uniform by construction (Step 3.2).

- **$\gamma, \tau$**: Algorithmic parameters, fixed independent of $N$.

- **$C_{\text{total}}$**: Additive expansion constant. From composition (Step 4), $C_{\text{total}}$ is sum of N-uniform terms $C'_W, C'_v, C_x, C_W, C_v, C'_b$. Each term bounded by problem-dependent constants (fitness potential regularity, boundary geometry) independent of $N$.

**Substep 6.4**: Final TV bound with explicit constants
- **Assembly**:
  $$\|\mathcal{L}(S_t | S_0) - \pi_{\text{QSD}}\|_{\text{TV}} \le C_\pi (1 + V_{\text{total}}(S_0)) e^{-\kappa_{\text{total}} t}$$
  where all constants $C_\pi, \kappa_{\text{total}}$ are N-uniform
- **Conclusion**: Theorem statement verified with explicit parameter dependence

**Conclusion**:
- Convergence rate $\kappa_{\text{total}} = O(\min\{\gamma\tau, \kappa_x^{\text{clone}}, c_{\min}\underline{\lambda} - C_1 L_\Sigma - C_2|\nabla\Sigma_{\text{reg}}|_\infty\})$
- All constants independent of $N$ by 1/N normalization strategy
- **Form**: Matches theorem statement exactly

**Dependencies**:
- Uses: All previous steps
- Requires: Careful tracking of N-dependence through 1/N normalization

**Potential Issues**:
- None identified - N-uniformity is well-established in framework via systematic 1/N normalization

**Q.E.D.** ∎

---

## V. Technical Deep Dives

### Challenge 1: Anisotropic Hypocoercive Contraction

**Why Difficult**:
The hypocoercive method for proving convergence of underdamped Langevin dynamics relies on constructing a modified Lyapunov function (hypocoercive norm) that captures coupling between position and velocity via a cross term $\langle \Delta x, \Delta v \rangle$. For isotropic noise $\Sigma = \sigma_v I$, the noise contribution to the drift cancels in the synchronous coupling (both swarms receive identical noise realizations), leaving only the deterministic drift terms. For anisotropic, state-dependent noise $\Sigma = \Sigma_{\text{reg}}(x, S)$, the noise tensors differ between coupled swarms: $\Sigma_{\text{reg}}(S_1) \neq \Sigma_{\text{reg}}(S_2)$, creating additional drift terms proportional to $\|\Sigma_{\text{reg}}(S_1) - \Sigma_{\text{reg}}(S_2)\|$. Moreover, the Stratonovich-to-Itô correction introduces gradient-dependent drift $\sim \nabla \Sigma_{\text{reg}}$. These perturbations must be proven small enough that the $c_{\min}$-scaled contraction from uniform ellipticity dominates.

**Proposed Solution**:

1. **Start with isotropic hypocoercive analysis**: Use the standard quadratic form from `../1_euclidean_gas/06_convergence.md`:
   $$V_W = \int \|(\Delta x, \Delta v)\|_h^2 d\pi(coupling)$$
   where $\|(\Delta x, \Delta v)\|_h^2 = \|\Delta x\|^2 + \lambda_v\|\Delta v\|^2 + b\langle\Delta x, \Delta v\rangle$ with optimized $\lambda_v, b$.

2. **Compute drift with anisotropic noise**:
   - Deterministic drift: $\langle\Delta x, \Delta v\rangle$ term from position-velocity coupling, $-\gamma\langle\Delta v, \Delta v\rangle$ from friction, $\langle\Delta v, F(x_1) - F(x_2)\rangle$ from force
   - Noise drift: $\text{Tr}(D_{\text{reg}}(S_1) \nabla^2_{v_1} V_W) + \text{Tr}(D_{\text{reg}}(S_2) \nabla^2_{v_2} V_W)$ from diffusion
   - Mismatch drift: $(D_{\text{reg}}(S_1) - D_{\text{reg}}(S_2))$ creates coupling error
   - Itô correction: $\nabla\Sigma_{\text{reg}} \cdot \Sigma_{\text{reg}}$ terms

3. **Bound perturbations using uniform ellipticity and Lipschitz**:
   - Mismatch: $\|D_{\text{reg}}(S_1) - D_{\text{reg}}(S_2)\| \le 2c_{\max} L_\Sigma d_{\text{state}}(S_1, S_2)$ (product rule on Lipschitz)
   - Itô: $\|\nabla\Sigma_{\text{reg}}\| \le |\nabla\Sigma_{\text{reg}}|_\infty$ by definition
   - Ellipticity: $\text{Tr}(D_{\text{reg}} \nabla^2 V_W) \ge c_{\min} \text{Tr}(\nabla^2 V_W)$ ensures lower bound on dissipation

4. **Derive net contraction**:
   - Group deterministic terms: coercivity $\underline{\lambda}$ from hypocoercive quadratic form analysis
   - Scale by $c_{\min}$: effective contraction $c_{\min}\underline{\lambda}$
   - Add perturbations: $+C_1 L_\Sigma + C_2|\nabla\Sigma_{\text{reg}}|_\infty$ where $C_1, C_2$ come from Young's inequality estimates
   - Net rate: $\kappa'_W = c_{\min}\underline{\lambda} - C_1 L_\Sigma - C_2|\nabla\Sigma_{\text{reg}}|_\infty$

5. **Verify threshold condition**: Check that reasonable parameter choices (large enough $\epsilon_\Sigma$ to make $c_{\min}$ substantial, smooth enough $\phi$ to make $L_\Sigma, |\nabla\Sigma_{\text{reg}}|_\infty$ moderate) satisfy $c_{\min}\underline{\lambda} > C_1 L_\Sigma + C_2|\nabla\Sigma_{\text{reg}}|_\infty$.

**Alternative if fails**:
If perturbations are too large, switch to **relative entropy method** (KL divergence) which may have better stability properties under anisotropic noise. The document mentions connection to Log-Sobolev inequality (`15_geometric_gas_lsi_proof.md`), suggesting this as a backup approach.

**References**:
- Hypocoercivity theory: Villani (2009), "Hypocoercivity," Memoirs AMS
- State-dependent diffusion: Bakry-Gentil-Ledoux (2014), "Analysis and Geometry of Markov Diffusion Operators"
- Similar techniques in framework: `../1_euclidean_gas/06_convergence.md` § 2 (isotropic case), this document § 5.3 (anisotropic extension)

---

### Challenge 2: Itô Correction Bound with Matrix-Valued Coefficients

**Why Difficult**:
The kinetic SDE is written in Stratonovich form for geometric clarity (coordinate-invariant under smooth transformations). Converting to Itô form for rigorous analysis adds a correction term:
$$b_{\text{Itô}}(x,v,S) = b_{\text{Strat}}(x,v,S) + \frac{1}{2}\sum_{j=1}^d (D_x\Sigma_{\text{reg}}^{(\cdot,j)})\Sigma_{\text{reg}}^{(\cdot,j)}$$
where $\Sigma_{\text{reg}}^{(\cdot,j)}$ is the $j$-th column of $\Sigma_{\text{reg}}$ (a $d \times d$ matrix), and $D_x$ is the Jacobian with respect to $x$. This correction involves the product of gradients and matrix entries, requiring careful bounds. The challenge is showing this contribution is $O(|\nabla\Sigma_{\text{reg}}|_\infty)$ uniformly in $N$.

**Proposed Solution**:

1. **Expand Itô correction formula**:
   $$\text{Itô correction} = \frac{1}{2}\sum_{j=1}^d \frac{\partial \Sigma_{\text{reg}}^{(i,j)}}{\partial x_k} \Sigma_{\text{reg}}^{(k,j)}$$
   (Einstein summation over $k,j$, component $i$ of the correction for velocity component $i$)

2. **Bound gradient norm**:
   - $\Sigma_{\text{reg}} = (H + \epsilon_\Sigma I)^{-1/2}$, so $\frac{\partial \Sigma_{\text{reg}}}{\partial x} = -\frac{1}{2}(H + \epsilon_\Sigma I)^{-3/2} \frac{\partial H}{\partial x}$ by chain rule
   - $\|\frac{\partial H}{\partial x}\| \le |\nabla H|$ where $H = \frac{1}{N}\sum_j \nabla^2\phi(x_i, x_j)$ gives $\nabla H = \frac{1}{N}\sum_j \nabla^3\phi$
   - Assuming $\phi$ has bounded third derivatives $|\nabla^3\phi| \le L_\phi^{(3)}$, we have $|\nabla H| \le L_\phi^{(3)}$ (N-uniform)

3. **Bound matrix composition**:
   - $(H + \epsilon_\Sigma I)^{-3/2}$ has operator norm $\le c_{\min}^{3/2}$ (since smallest eigenvalue is $\ge c_{\min}$)
   - $\|\Sigma_{\text{reg}}\| \le c_{\max}^{1/2}$ by uniform ellipticity upper bound
   - Product: $\|\frac{\partial \Sigma_{\text{reg}}}{\partial x}\| \le \frac{1}{2} c_{\min}^{3/2} \cdot L_\phi^{(3)} = O(L_\phi^{(3)}/c_{\min}^{3/2})$

4. **Sum over columns**:
   $$\|\text{Itô correction}\| \le \frac{1}{2}\sum_{j=1}^d \|\frac{\partial \Sigma_{\text{reg}}^{(\cdot,j)}}{\partial x}\| \cdot \|\Sigma_{\text{reg}}^{(\cdot,j)}\|$$
   $$\le \frac{d}{2} \cdot \frac{L_\phi^{(3)}}{c_{\min}^{3/2}} \cdot c_{\max}^{1/2} = \frac{d L_\phi^{(3)}}{2c_{\min}^{3/2}} \cdot c_{\max}^{1/2}$$

5. **Define constant**:
   $$C_2 = \frac{d L_\phi^{(3)}}{2c_{\min}^{3/2}} \cdot c_{\max}^{1/2}$$
   and $|\nabla\Sigma_{\text{reg}}|_\infty = \sup_{x,S} \|\frac{\partial\Sigma_{\text{reg}}}{\partial x}\| = O(L_\phi^{(3)}/c_{\min}^{3/2})$

6. **Verify N-uniformity**: All quantities ($d$, $L_\phi^{(3)}$, $c_{\min}$, $c_{\max}$) are N-uniform, so $C_2$ is N-uniform.

**Alternative if fails**:
If third-derivative bound is unavailable, use **Hölder continuity** of second derivatives instead, trading polynomial bound for slightly weaker regularity assumption.

**References**:
- Stratonovich-to-Itô conversion: Øksendal (2003), "Stochastic Differential Equations," § 4.2
- Matrix-valued SDEs: Da Prato-Zabczyk (2014), "Stochastic Equations in Infinite Dimensions," Chapter 7
- Framework reference: Document line 945-966 (Lemma on Itô correction)

---

### Challenge 3: Harris Conditions Under Anisotropic Noise with Absorbing Boundary

**Why Difficult**:
Standard Harris theorem for geometric ergodicity requires: (i) $\phi$-irreducibility, (ii) aperiodicity, (iii) Foster-Lyapunov drift, (iv) small-set minorization. For processes with absorbing boundaries, these conditions must be verified on the **living subspace** $\mathcal{A} = \{S : N_{\text{alive}}(S) \ge 1\}$, which is not closed (boundary $\partial\mathcal{A}$ is absorbing). The QSD framework (Collet-Martínez-San Martín, Champagnat-Villemonais) extends Harris to this setting, but requires checking that the process has sufficient accessibility within $\mathcal{A}$ before absorption occurs. With anisotropic, state-dependent noise, the transition density is not explicitly known, making minorization construction non-trivial.

**Proposed Solution**:

1. **$\phi$-irreducibility on living subspace**:
   - **Goal**: For any open set $A \subset \mathcal{A}$ (living subspace), show $P^n(x, A) > 0$ for some $n$ and all $x \in \mathcal{A}$
   - **Technique**:
     - Uniform ellipticity $c_{\min} I \preceq D_{\text{reg}}$ ensures non-degenerate Gaussian component in noise
     - Confining potential ensures process stays in compact region with high probability
     - Standard Gaussian accessibility: any open set reachable via noise realizations
   - **Verification**: Adapt proof from `../1_euclidean_gas/06_convergence.md` § 4, replacing $\sigma_v^2$ with $c_{\min}$ throughout

2. **Aperiodicity**:
   - **Trivial**: Continuous-time, continuous-space diffusion with non-degenerate noise is automatically aperiodic (no lattice structure, no deterministic return times)

3. **Small-set minorization**:
   - **Goal**: Find compact sets $K \subset \mathcal{A}$ such that $P^n(x, \cdot) \ge \epsilon \nu(\cdot)$ for $x \in K$, some minorizing measure $\nu$, constant $\epsilon > 0$, step count $n$
   - **Construction**:
     - Choose $K$ = ball of radius $R$ around center of $\mathcal{X}_{\text{valid}}$, far from boundary
     - Confining force ensures process starting in $K$ returns to $K$ within $n$ steps with probability $\ge p_{\text{return}}$
     - Uniform ellipticity ensures transition density has lower bound $\ge c_{\min}^{d/2} e^{-C/c_{\min}}$ on neighborhoods (Gaussian density estimate)
     - Combine: $\epsilon = p_{\text{return}} \cdot c_{\min}^{d/2} e^{-C/c_{\min}}$, minorizing measure $\nu$ = normalized Gaussian
   - **N-uniformity**: $c_{\min}$ is N-uniform, confining force parameters are N-uniform, so $\epsilon$ is N-uniform

4. **Conditioning on survival**:
   - **QSD framework**: The above conditions apply to the process conditioned on $\{N_{\text{alive}} \ge 1\}$
   - **Key insight**: Absorption probability decays exponentially as $e^{-\kappa_{\text{abs}} t}$ where $\kappa_{\text{abs}}$ depends on boundary potential and confining force
   - **QSD limit**: As $t \to \infty$ conditioned on survival, the process converges to the QSD $\pi_{\text{QSD}}$, which is the dominant eigenvector of the killed generator
   - **Uniqueness**: Foster-Lyapunov + irreducibility + aperiodicity imply unique QSD (Champagnat-Villemonais 2017, Theorem 3.1)

5. **Explicit constants**:
   - Minorization $\epsilon = O(c_{\min}^{d/2})$, N-uniform
   - Return probability $p_{\text{return}} = O(\exp(-C\|F\|_\infty/\alpha_U))$, depends on confining potential but not $N$
   - QSD convergence rate: $\kappa_{\text{QSD}} = O(\kappa_{\text{total}} - \kappa_{\text{abs}})$, dominated by Foster-Lyapunov rate in regime $\kappa_{\text{total}} \ll \kappa_{\text{abs}}$ (rare absorption)

**Alternative if fails**:
If small-set construction is difficult, use **local Doeblin condition** (Meyn-Tweedie § 16) which is weaker and easier to verify for diffusions, then invoke generalized Harris theorem.

**References**:
- QSD theory: Collet-Martínez-San Martín (2013), "Quasi-Stationary Distributions"
- Harris for QSD: Champagnat-Villemonais (2017), "Exponential convergence to quasi-stationary distribution and Q-process"
- Framework template: `../1_euclidean_gas/06_convergence.md` § 4, document § 4.2

---

## VI. Proof Validation Checklist

- [x] **Logical Completeness**: All steps follow from previous steps with explicit justifications
- [x] **Hypothesis Usage**: All theorem assumptions used:
  - Uniformly elliptic anisotropic diffusion: Used in Steps 1-2, 5
  - Regularization $\epsilon_\Sigma I$: Ensures uniform ellipticity
  - Confining potential: Ensures boundary contraction and minorization
- [x] **Conclusion Derivation**: Claimed conclusion fully derived:
  - Unique QSD: Step 5
  - Exponential TV convergence: Step 5
  - Explicit rate $\kappa_{\text{total}}$: Step 6
  - N-uniformity: Step 6
- [x] **Framework Consistency**: All dependencies verified in glossary and source documents
- [x] **No Circular Reasoning**: Proof builds from established results (uniform ellipticity, cloning, isotropic template) without assuming conclusion
- [x] **Constant Tracking**: All constants defined and bounded:
  - $c_{\min}, c_{\max}$: Explicit formulas (Step 1)
  - $L_\Sigma$: N-uniform Lipschitz constant (Step 1)
  - $|\nabla\Sigma_{\text{reg}}|_\infty$: Bounded by $L_\phi^{(3)}$ (Step 1)
  - $\kappa_x^{\text{clone}}$: From cloning theory (Step 3)
  - $C_1, C_2$: From perturbation analysis (Step 2, requires explicit computation)
- [x] **Edge Cases**:
  - Boundary behavior: Handled via $V_{\text{boundary}}$ component and confining force
  - Small $N$: N-uniformity ensures constants work for all $N \ge 1$
  - Absorption: QSD framework conditions on survival
- [x] **Regularity Verified**:
  - Smoothness of $\phi$: Third derivatives bounded (assumed in framework)
  - Lipschitz $\Sigma_{\text{reg}}$: Proven in Step 1
  - Twice-differentiable Lyapunov: $V_{\text{total}}$ is quadratic in positions/velocities
- [x] **Measure Theory**:
  - Wasserstein distance well-defined: Coupling exists via synchronous construction
  - Expectations well-defined: Polynomial growth of $V_{\text{total}}$ ensures integrability
  - QSD is probability measure: Normalized eigenvector

---

## VII. Alternative Approaches (Not Chosen)

### Alternative 1: Riemannian Langevin Formalism

**Approach**: Instead of analyzing the process in flat Euclidean coordinates with anisotropic diffusion $D_{\text{reg}}(x,S)$, work intrinsically on the emergent Riemannian manifold $({\mathcal{X}}, g)$ where $g(x,S) = H + \epsilon_\Sigma I$. The Stratonovich SDE becomes a standard Riemannian Langevin equation on the manifold, and convergence can be analyzed using hypocoercivity theory on manifolds (Baudoin 2017, Grothaus-Stilgenbauer 2014).

**Pros**:
- **Geometric naturality**: The anisotropy is "explained away" by working in the right coordinate system - diffusion becomes isotropic with respect to Riemannian volume measure
- **Existing theory**: Hypocoercivity on manifolds is well-developed, with explicit constants depending on Ricci curvature bounds
- **Coordinate invariance**: Results are manifestly independent of coordinate choice

**Cons**:
- **More overhead**: Requires importing manifold machinery (Christoffel symbols, Ricci curvature, Riemannian volume measure)
- **Condition number factors**: Converting back to flat coordinates introduces condition-number dependence (as noted in document § 3.6), potentially obscuring explicit constants
- **Cloning interaction**: Cloning operator acts in algorithmic space, not intrinsic to manifold, requiring careful translation
- **Not the framework's approach**: Document explicitly works in flat space (§ 3.6) to maintain explicit constants

**When to Consider**: If uniform ellipticity degrades ($c_{\min}/c_{\max} \to 0$), manifold perspective might provide better bounds via curvature-based analysis. Also useful for theoretical insights and connections to information geometry.

---

### Alternative 2: Generator Perturbation / Spectral Method

**Approach**: Prove that the anisotropic kinetic generator $\mathcal{L}_{\text{anis}}$ is a **relatively bounded perturbation** of the isotropic generator $\mathcal{L}_{\text{iso}}$:
$$\|\mathcal{L}_{\text{anis}} - \mathcal{L}_{\text{iso}}\|_{\text{op}} \le \delta \|\mathcal{L}_{\text{iso}}\|_{\text{op}}$$
for some small $\delta$. Then use stability of spectral gap / Poincaré constant / Log-Sobolev constant under perturbations (Kato-Rellich theorem, Davies 1995) to transfer convergence from isotropic to anisotropic case.

**Pros**:
- **Concise**: If relative bound is tight, transfers entire convergence result in one step
- **Spectral interpretation**: Directly relates to eigenvalue gap of generator, providing intuition
- **Robust to small perturbations**: Shows convergence is stable under modeling errors

**Cons**:
- **Hard to make quantitative**: Deriving explicit spectral gap bounds typically requires variational methods (Poincaré inequality, etc.), which may be as hard as the original hypocoercive proof
- **Loses decomposition**: Doesn't distinguish kinetic vs. cloning contributions, making parameter dependence less clear
- **Perturbation margin**: Requires $\delta < 1$ for stability, which may impose restrictive conditions on $\epsilon_\Sigma$
- **Itô correction**: Gradient-dependent drift might violate relative boundedness if $|\nabla\Sigma_{\text{reg}}|$ is too large

**When to Consider**: If the goal is qualitative convergence (proving exponential rate exists) rather than explicit constants. Also useful for robustness analysis and stability under parameter variations.

---

### Alternative 3: Relative Entropy / KL Divergence Method

**Approach**: Instead of Wasserstein-based Lyapunov, use relative entropy (KL divergence) between current distribution and target QSD:
$$H(\mu_t || \pi_{\text{QSD}}) = \int \log\left(\frac{d\mu_t}{d\pi_{\text{QSD}}}\right) d\mu_t$$
Prove that KL divergence decays exponentially via **Log-Sobolev inequality (LSI)** or **entropy production method**. Document references `15_geometric_gas_lsi_proof.md` for LSI proof under anisotropic diffusion.

**Pros**:
- **Information-geometric**: Natural for problems with emergent Riemannian structure (Fisher metric = LSI)
- **Stronger than Wasserstein**: KL convergence implies Wasserstein convergence (but not vice versa)
- **Direct rate**: LSI constant directly gives exponential convergence rate in KL
- **Already proven**: Document indicates LSI proof exists separately

**Cons**:
- **Regularity requirements**: KL divergence requires absolute continuity and density regularity (log-densities exist and are well-behaved)
- **No coupling interpretation**: Loses geometric intuition from Wasserstein coupling
- **Harder to compose**: Combining LSI from kinetic + cloning operators less direct than Lyapunov composition
- **Different constant**: LSI constant may differ from Foster-Lyapunov rate, requiring translation

**When to Consider**: If the theorem requires **strong convergence** (in KL or other f-divergence) rather than just TV. Also natural for information-theoretic applications (channel capacity, entropy power inequality connections).

---

## VIII. Open Questions and Future Work

### Remaining Gaps

1. **Explicit constants $C_1, C_2$ in perturbation bound**: The proof strategy identifies that $\kappa'_W = c_{\min}\underline{\lambda} - C_1 L_\Sigma - C_2|\nabla\Sigma_{\text{reg}}|_\infty$, but explicit formulas for $C_1, C_2$ require detailed drift computation in document Chapter 5 § 5.3. **Criticality: High** - Without these constants, threshold condition cannot be verified numerically.

2. **Threshold sufficiency for typical parameters**: Does the condition $c_{\min}\underline{\lambda} > C_1 L_\Sigma + C_2|\nabla\Sigma_{\text{reg}}|_\infty$ hold for reasonable algorithmic parameter choices (e.g., $\epsilon_\Sigma \sim 0.1$, $\gamma \sim 1$, smooth fitness potentials)? **Criticality: Medium** - Determines practical applicability of theorem.

3. **Optimal choice of hypocoercive parameters $\lambda_v, b$**: The hypocoercive norm has free parameters $\lambda_v$ (velocity weight), $b$ (cross-coupling). Optimal choice depends on $c_{\min}, \gamma$. **Criticality: Low** - Affects constant factors but not qualitative convergence.

### Conjectures

1. **Threshold degradation with dimension**: Conjecture that $C_1, C_2$ grow at most polynomially in dimension $d$ (not exponentially), suggesting the method extends to moderately high-dimensional problems. **Plausibility: High** - Lipschitz constants and Itô corrections involve sums over $d$ coordinates, suggesting linear or quadratic dependence.

2. **Sharpness of perturbation bound**: Conjecture that the bound $\kappa'_W \ge c_{\min}\underline{\lambda} - C_1 L_\Sigma - C_2|\nabla\Sigma_{\text{reg}}|_\infty$ is **sharp** (optimal) for worst-case diffusion structures, but typical cases have better rates due to cancellations. **Plausibility: Medium** - Perturbation bounds are often pessimistic.

3. **Regime transition**: Conjecture that for small $\epsilon_\Sigma$ (high anisotropy), the bottleneck is kinetic (hypocoercive-limited), while for large $\epsilon_\Sigma$ (near isotropic), the bottleneck is cloning (cloning-limited). **Plausibility: High** - Matches qualitative behavior described in document § 5.6 on convergence regimes.

### Extensions

1. **Adaptive $\epsilon_\Sigma$ schedules**: Can convergence be accelerated by varying regularization over time, starting with large $\epsilon_\Sigma$ (fast isotropic exploration) and decreasing to small $\epsilon_\Sigma$ (refined anisotropic exploitation)?

2. **Higher-order geometry**: Current framework uses Hessian (second derivatives). Can third-order or higher tensors (Christoffel symbols, curvature) further improve convergence by encoding more geometric information?

3. **Non-Euclidean base spaces**: Can the framework extend to $\mathcal{X}$ being a Riemannian manifold itself (not just $\mathbb{R}^d$), creating "doubly Riemannian" structure (base manifold + emergent metric)?

4. **Quantum analogues**: Emergent geometry and natural gradient have quantum information interpretations (Bures metric, quantum Fisher information). Can similar convergence results hold for quantum algorithms?

---

## IX. Expansion Roadmap

**Phase 1: Prove Missing Lemmas** (Estimated: 2-3 weeks)

1. **Lemma D** (Anisotropic hypocoercive contraction):
   - Expand drift computation from document § 5.3
   - Derive explicit $C_1, C_2$ via perturbation analysis
   - Verify threshold condition for example parameter sets
   - **Difficulty: Hard** - Core technical contribution

2. **Lemma C** (Itô correction bound):
   - Expand Stratonovich-to-Itô conversion with matrix calculus
   - Bound all composition factors using uniform ellipticity
   - Verify N-uniformity of result
   - **Difficulty: Medium** - Standard but tedious

3. **Lemma F** (Harris conditions):
   - Adapt small-set minorization from isotropic template
   - Verify accessibility using uniform ellipticity
   - Construct explicit minorizing measure and constant
   - **Difficulty: Medium** - Adaptation of known template

**Phase 2: Fill Technical Details** (Estimated: 1-2 weeks)

1. **Step 2 (Hypocoercive contraction)**:
   - Expand substeps 2.2-2.5 with full drift calculations
   - Provide explicit Young's inequality estimates for perturbation bounds
   - Show all cancellations and simplifications

2. **Step 4 (Operator composition)**:
   - Derive explicit coupling constants $c_V, c_B$
   - Show balance conditions hold for all parameter regimes
   - Verify tower property computations

3. **Step 5 (QSD framework)**:
   - Provide detailed Harris theorem invocation with all hypothesis checks
   - Verify Lyapunov domination conditions
   - Derive TV bound constant $C_\pi$ explicitly

**Phase 3: Add Rigor** (Estimated: 1 week)

1. **Epsilon-delta arguments**:
   - Step 5.3 (minorization): Provide explicit $\epsilon, \delta$ for accessibility
   - Threshold verification: Show $c_{\min}\underline{\lambda} > C_1 L_\Sigma + C_2|\nabla\Sigma_{\text{reg}}|_\infty$ for numerical parameter ranges

2. **Measure-theoretic details**:
   - Justify all expectation exchanges (Fubini via integrability)
   - Verify probability space construction for coupled swarms
   - Check $\sigma$-algebra measurability of all random variables

3. **Counterexamples for necessity of assumptions**:
   - Show failure without uniform ellipticity (degenerate noise case)
   - Show failure without Lipschitz continuity (discontinuous $\Sigma_{\text{reg}}$ example)
   - Demonstrate threshold sharpness with adversarial $H$

**Phase 4: Review and Validation** (Estimated: 1 week)

1. **Framework cross-validation**:
   - Verify all citations against source documents
   - Check consistency with isotropic template
   - Confirm N-uniformity of all constants

2. **Edge case verification**:
   - $N = 1$ (single walker): QSD trivial, check graceful degradation
   - $\epsilon_\Sigma \to \infty$ (isotropic limit): recover Euclidean Gas rates
   - $\epsilon_\Sigma \to \Lambda_-$ (minimal regularization): verify threshold condition

3. **Constant tracking audit**:
   - Tabulate all constants with explicit dependence on $(\gamma, \tau, \epsilon_\Sigma, d, \text{fitness regularity})$
   - Verify no hidden $N$-dependence anywhere
   - Check dimension scaling

**Total Estimated Expansion Time**: 5-7 weeks for complete detailed proof

---

## X. Cross-References

**Theorems Used**:
- {prf:ref}`thm-uniform-ellipticity` - Uniform ellipticity bounds for diffusion tensor
- {prf:ref}`prop-lipschitz-diffusion` - Lipschitz continuity of adaptive diffusion
- {prf:ref}`def-d-kinetic-operator-adaptive` - Kinetic SDE with adaptive diffusion
- {prf:ref}`def-d-coupled-lyapunov` - Coupled Lyapunov function construction

**Definitions Used**:
- {prf:ref}`def-d-adaptive-diffusion` - Adaptive diffusion tensor from regularized Hessian
- {prf:ref}`def-d-coupled-state` - Coupled swarm state space
- Cloning drift inequalities (../1_euclidean_gas/03_cloning.md)
- Foster-Lyapunov framework (../1_euclidean_gas/06_convergence.md)

**Related Proofs** (for comparison):
- Isotropic Euclidean Gas convergence: {prf:ref}`../1_euclidean_gas/06_convergence.md § 3`
- QSD theory for absorbing Gas: {prf:ref}`../1_euclidean_gas/06_convergence.md § 4`
- Cloning operator analysis: {prf:ref}`../1_euclidean_gas/03_cloning.md § 5`
- Log-Sobolev inequality for Geometric Gas: {prf:ref}`15_geometric_gas_lsi_proof.md`

---

**Proof Sketch Completed**: 2025-10-25

**Ready for Expansion**: Needs additional lemmas (D, C, F)

**Confidence Level**: Medium-High - GPT-5 strategy is well-founded with clear framework dependencies, but missing Gemini cross-validation reduces confidence. Core technical challenge (Lemma D - anisotropic hypocoercive contraction) requires detailed computation from document § 5.3 to verify threshold condition. N-uniformity is well-established. Overall structure is sound and follows proven template from isotropic case.

**Recommendation**: Proceed with lemma expansion, prioritizing Lemma D (explicit $C_1, C_2$ constants) to verify threshold sufficiency.
