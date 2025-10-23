# Viability Analysis: Symmetry Redefinition in the Fractal Set Framework

**Document Status:** Research Report
**Version:** 1.0
**Date:** 2025-10-23
**Authors:** Claude Code (analysis), with Gemini 2.5 Pro (peer review)

---

## Abstract

This document provides a rigorous mathematical and physical analysis of a proposed redefinition of the gauge symmetries in the Fractal Set framework—specifically the U(1)_fitness and SU(2)_weak symmetries that emerge from the Adaptive Gas algorithm dynamics. The proposal suggests using **processed collective field values** (the rescaled outputs d'_i and r'_i from the measurement pipeline) as the basis for gauge phases, rather than the raw algorithmic distances currently used. This aims to create a more direct mapping between the algorithm's intrinsic "processed perception" of state and the gauge structure of the Standard Model.

**Key findings:**

1. **Algorithmic Viability**: The proposal is mathematically well-defined and computationally sound. The apparent circular dependency identified in initial review was based on a misunderstanding—the algorithm employs two independent companion selection mechanisms (diversity pairing vs cloning companion), which resolves the recursion concern.

2. **Collective Field Structure**: The quantities d'_i and r'_i are not single-walker singlets but **collective local field values** that depend on the entire swarm state through z-score standardization. This is a novel feature distinct from both single-particle and pairwise quantities.

3. **Gauge Theory Interpretation**: Critical analysis reveals that the proposed phases, constructed from physical observables and their statistics, appear to be **gauge-invariant** rather than gauge-covariant. This raises fundamental questions about whether the structure can support a local gauge theory in the Standard Model sense.

4. **Three Interpretations**: We identify three possible theoretical frameworks for understanding the proposed structure: (1) local gauge theory (requires proof of non-trivial transformation), (2) mean-field/collective field theory (most likely), and (3) global symmetry with state-dependent parameters.

**Recommendation:** The proposed redefinition is algorithmically viable and physically meaningful, but should be framed as an **emergent collective field theory** rather than a direct Standard Model analog until the transformation properties of the collective fields can be rigorously established.

---

## Executive Summary

### Research Question

Can the gauge symmetries in the Fractal Set framework be redefined to use the **processed outputs** of the measurement pipeline (d'_i, r'_i) rather than raw algorithmic distances, creating a tighter connection between the algorithm's intrinsic operations and Standard Model physics?

### Proposed Changes

**U(1)_fitness Symmetry:**
- Current: Phase θ_ik^(div) = -d_alg^2(i,k) / (2ε_d^2 ℏ_eff) (raw pairwise distance)
- Proposed: Phase θ_i = f(d'_i) where d'_i = [g_A((d_i - μ_d)/σ'_d) + η]^β (collective field)

**SU(2)_weak Symmetry:**
- Current: Phase θ_ij^(SU(2)) = -d_alg^2(i,j) / (2ε_c^2 ℏ_eff) (raw pairwise distance)
- Proposed: Phase θ_ij = g(S_i(j)) where S_i(j) is cloning score (fitness comparison)

**Higgs Mechanism:**
- Current: Fitness potential V_fit as abstract Higgs-like role
- Proposed: Reward channel output r'_i = [g_A((r_i - μ_r)/σ'_r) + η]^α as explicit "Higgs field"

### Key Findings Summary

| Aspect | Status | Details |
|--------|--------|---------|
| **Circular Dependency** | ✅ Resolved | Two independent companion selections |
| **Collective Field Structure** | ✅ Well-defined | d'_i, r'_i depend on entire swarm via zscore |
| **Algorithmic Soundness** | ✅ Viable | Computable, preserves convergence |
| **Gauge Covariance** | ⚠️ Unclear | Phases may be gauge-invariant, not covariant |
| **SM Mapping** | ⚠️ Indirect | May be mean-field theory, not local gauge theory |
| **Novelty** | ✅ High | Collective field structure is novel |

### Verdict

**Split Verdict:**
- **Algorithmic level**: ✅ **VIABLE** — mathematically sound, computationally efficient
- **Physics interpretation**: ⚠️ **UNCLEAR** — may not be local gauge theory; likely mean-field/collective theory

**Recommendation:** Adopt the structure for algorithmic purposes, develop as **emergent collective field theory** rather than claiming direct Standard Model correspondence.

---

## 1. Introduction

### 1.1. Motivation

The Fractal Set framework, developed in {doc}`01_fractal_set.md` and {doc}`03_yang_mills_noether.md`, provides a discrete spacetime representation of the Adaptive Gas algorithm with an emergent gauge theory structure. This framework identifies:

- **S_N permutation symmetry** (walker label indistinguishability)
- **SU(2)_weak local gauge symmetry** (cloning isospin doublet structure)
- **U(1)_fitness global symmetry** (fitness scale invariance)

These symmetries suggest a remarkable connection to the Standard Model's electroweak sector. However, the current framework bases its gauge phases on **raw algorithmic distances** d_alg^2(i,j), which are primitive geometric quantities. The measurement pipeline then processes these distances through standardization (z-scores) and rescaling to produce the **collective field values** d'_i and r'_i that actually drive the algorithm's fitness-based decisions.

**Central question:** If the algorithm's "processed perception" of state is encoded in d'_i and r'_i, should the gauge structure also be defined in terms of these quantities rather than the raw inputs?

### 1.2. Conceptual Appeal of the Proposal

The proposed redefinition has several conceptually attractive features:

**1. Algorithmic Naturalness:**
The algorithm makes decisions based on V_fit = (d'_i)^β · (r'_i)^α, not on raw distances. Using d'_i and r'_i as phase sources would align the gauge structure with the algorithm's actual operating principles.

**2. Relative vs Absolute Information:**
The z-score operation extracts *relative* information (how does walker i compare to the swarm?), removing absolute scales. This mirrors the gauge principle that absolute phases are unphysical—only relative phases matter.

**3. Fitness-Driven Interaction:**
Using the cloning score S_i(j) = (V_fit_j - V_fit_i) / (V_fit_i + ε) as the SU(2) phase directly connects interaction strength to fitness comparison, which is physically intuitive (better walkers should attract cloning more strongly).

**4. Higgs Mechanism Analogy:**
Identifying r'_i as the "Higgs field" gives the reward channel an explicit symmetry-breaking role, analogous to how the SM Higgs gives mass to particles.

### 1.3. Initial Concerns and Their Resolution

An initial dual review (Gemini 2.5 Pro and Claude Code) identified three critical concerns:

**Concern #1 (Initial): "U(1) phase is not pairwise"**
- Initial assessment: θ_i = f(d'_i) depends only on walker i, not pair (i,k)
- **Resolution**: While d'_i is formally a function of i, it depends on the entire swarm through (μ_d, σ'_d). It is a **collective local field value**, not a single-particle property. This is novel but not automatically invalid.

**Concern #2 (Initial): "Circular dependency in SU(2) phase"**
- Initial assessment: Phase uses S_i(j) which depends on V_fit, creating Phase → V_fit → Phase
- **Resolution**: INCORRECT. The algorithm uses **two independent companion selections**:
  - Diversity companion c_div(i) for measuring d_i (feeds into V_fit)
  - Cloning companion c_clone(i) for cloning decision (uses V_fit but doesn't feed back into it)
  - Causal flow is: c_div → d_i → V_fit, then separately c_clone + V_fit → S_i(c_clone)
  - No circle exists.

**Concern #3 (Initial): "r'_i is a singlet, not doublet"**
- Initial assessment: r'_i doesn't transform under SU(2) like SM Higgs
- **Resolution**: PARTIALLY CORRECT. r'_i is not an SU(2) doublet, but it is a **collective field** (depends on entire swarm through z-score). The question is whether collective fields can play the Higgs role, not whether they're singlets vs doublets.

**Concern #4 (New): "Phases are gauge-invariant"**
- New critical issue identified after clarifications: If d'_i and r'_i are constructed from physical observables, they are gauge-invariant. A local gauge theory requires gauge-*covariant* fields that transform non-trivially.
- This is the **central unresolved question** of this analysis.

### 1.4. Document Structure

The remainder of this document is organized as follows:

- **Section 2**: Mathematical framework—detailed exposition of the two-companion system and collective field structure
- **Section 3**: Proposed symmetry redefinitions—precise mathematical definitions
- **Section 4**: Viability analysis—algorithmic soundness and gauge theory interpretation
- **Section 5**: Three interpretations—local gauge, mean-field, and global symmetry frameworks
- **Section 6**: Standard Model mapping—what physics can and cannot be described
- **Section 7**: Conclusions and recommendations
- **Appendices**: Technical details, proof attempts, and comparisons

---

## 2. Mathematical Framework: The Fragile Gas Measurement Pipeline

To rigorously evaluate the proposed symmetry redefinition, we must first establish the precise mathematical structure of the Fragile Gas algorithm, with particular attention to:
1. The **two independent companion selection mechanisms**
2. The **collective field structure** arising from z-score standardization
3. The **causal flow** through the measurement pipeline

All definitions in this section are drawn from {doc}`../1_euclidean_gas/03_cloning.md` §5.

### 2.1. Algorithmic Distance Metric

:::{prf:definition} Algorithmic Distance for Intra-Swarm Proximity
:label: def-algorithmic-distance-recap

For two walkers i and j with states w_i = (x_i, v_i, s_i) and w_j = (x_j, v_j, s_j), the **algorithmic distance** is:

$$
d_{\text{alg}}(i,j) := \sqrt{\|x_i - x_j\|^2 + \lambda_{\text{alg}} \|v_i - v_j\|^2}
$$

where λ_alg ≥ 0 is a hyperparameter determining the velocity contribution.

**Key property:** d_alg is a **physical observable**—it depends only on the walker states, not on any abstract coordinates or gauge choices. It is therefore **gauge-invariant** under any reasonable definition of gauge transformation.
:::

### 2.2. Two-Companion Selection System

The algorithm employs **two distinct, independent companion selection mechanisms** with different purposes and interaction scales.

#### 2.2.1. Diversity Companion Selection (for Measurement)

:::{prf:definition} Diversity Pairing Operator
:label: def-diversity-pairing-recap

The **diversity pairing operator** creates a companion map c_div : A_t → A_t for the alive set A_t at timestep t.

**Purpose:** Measure geometric diversity for the fitness potential calculation.

**Interaction scale:** ε_d > 0 (diversity interaction range).

**Procedure** (Sequential Stochastic Greedy Pairing, {doc}`../1_euclidean_gas/03_cloning.md` §5.1.2):
1. Initialize unpaired set U ← A_t
2. While |U| > 1:
   - Select walker i from U
   - For each j ∈ U \\ {i}, compute weight: w_ij = exp(-d_alg^2(i,j)/(2ε_d^2))
   - Sample c_div(i) ~ Categorical(w_ij / Σ_k w_ik)
   - Remove both i and c_div(i) from U
3. Return companion map c_div

**Output:** For each walker i ∈ A_t, a diversity companion c_div(i) ∈ A_t \\ {i}.

**Usage:** Compute raw distance d_i := d_alg(i, c_div(i)) for walker i.
:::

#### 2.2.2. Cloning Companion Selection (for Action)

:::{prf:definition} Cloning Companion Operator
:label: def-cloning-companion-recap

The **cloning companion operator** independently samples a companion c_clone(i) for each walker i when determining cloning probabilities.

**Purpose:** Select a target for fitness comparison in the cloning decision.

**Interaction scale:** ε_c > 0 (cloning interaction range, typically different from ε_d).

**Procedure** (Independent Softmax, {doc}`../1_euclidean_gas/03_cloning.md` §5.7.1):

For alive walker i ∈ A_t:

$$
P(c_{\text{clone}}(i) = j \mid i \in A_t) = \frac{\exp\left(-\frac{d_{\text{alg}}^2(i,j)}{2\epsilon_c^2}\right)}{\sum_{k \in A_t \setminus \{i\}} \exp\left(-\frac{d_{\text{alg}}^2(i,k)}{2\epsilon_c^2}\right)}
$$

**Key independence property:**
- c_clone(i) is sampled **after** the fitness potential V_fit has been computed
- c_clone(i) does **not** affect the computation of V_fit
- c_clone(i) is used only in the cloning score S_i(c_clone(i))
:::

:::{prf:remark} Independence of Companion Selections
:class: important

The two companion selections are **independent and non-circular**:

**Diversity pairing:** Used for distance measurement → feeds into fitness V_fit
**Cloning companion:** Uses fitness V_fit → determines cloning probability

**Causal flow:**

$$
\text{c}_{\text{div}}(i) \to d_i \to z_{d,i} \to d'_i \to V_{\text{fit},i}
$$

$$
r_i \to z_{r,i} \to r'_i \to V_{\text{fit},i}
$$

$$
V_{\text{fit},i} + \text{c}_{\text{clone}}(i) \to S_i(\text{c}_{\text{clone}}(i)) \to \text{cloning decision}
$$

There is **no feedback loop**. The cloning companion selection occurs in a later algorithmic stage and does not affect fitness computation.
:::

### 2.3. Measurement Pipeline: From Raw Values to Collective Fields

The fitness potential V_fit is not computed directly from raw distances and rewards. Instead, the raw values pass through a six-stage pipeline that performs **collective standardization** and **nonlinear rescaling**.

#### Stage 1-2: Raw Measurement

:::{prf:definition} Raw Measurements
:label: def-raw-measurements

For walker i ∈ A_t at timestep t:

**Raw distance:**

$$
d_i := d_{\text{alg}}(i, c_{\text{div}}(i))
$$

**Raw reward:**

$$
r_i := R(x_i, v_i) - c_{v_{\text{reg}}} \|v_i\|^2
$$

where R : X × ℝ^d → ℝ is the environment reward function and c_v_reg > 0 is the velocity regularization coefficient.
:::

#### Stage 3: Swarm Aggregation

:::{prf:definition} Swarm Statistics
:label: def-swarm-statistics

Given the raw distance vector d = (d_1, ..., d_k) and raw reward vector r = (r_1, ..., r_k) for k alive walkers:

**Distance statistics:**

$$
\mu_d := \frac{1}{k} \sum_{i \in A_t} d_i, \quad \sigma_d^2 := \frac{1}{k} \sum_{i \in A_t} (d_i - \mu_d)^2
$$

**Regularized distance standard deviation:**

$$
\sigma'_d := \sigma'_{\text{patch}}(\sigma_d^2) := \max\left(\sigma_d, \sigma_{\min,\text{patch}}\right)
$$

where σ_min,patch > 0 is a regularization constant.

**Reward statistics:** μ_r and σ'_r computed analogously.

**Key property:** These statistics depend on the **entire swarm state**, not just individual walkers.
:::

#### Stage 4: Standardization (Z-scores)

:::{prf:definition} Z-Score Standardization
:label: def-zscore-standardization

For each alive walker i ∈ A_t:

**Distance z-score:**

$$
z_{d,i} := \frac{d_i - \mu_d}{\sigma'_d}
$$

**Reward z-score:**

$$
z_{r,i} := \frac{r_i - \mu_r}{\sigma'_r}
$$

**Properties:**
1. Translation invariance: If all d_i → d_i + c, then z_d,i unchanged (μ_d shifts by c)
2. Scale invariance: If all d_i → λ · d_i, then z_d,i unchanged (both numerator and denominator scale)
3. **Collective dependence:** z_d,i changes when **any** walker's distance changes (affects μ_d or σ'_d)
:::

#### Stage 5: Rescaling

:::{prf:definition} Logistic Rescale Function
:label: def-logistic-rescale-recap

The **canonical logistic rescale function** g_A : ℝ → (0, 2) is:

$$
g_A(z) := \frac{2}{1 + e^{-z}}
$$

**Properties:** C^∞, strictly increasing, bounded, Lipschitz continuous.
:::

#### Stage 6: Collective Field Construction

:::{prf:definition} Collective Local Field Values
:label: def-collective-fields

For walker i ∈ A_t, the **collective local field values** are:

**Diversity field:**

$$
d'_i := g_A(z_{d,i}) + \eta = g_A\left(\frac{d_i - \mu_d}{\sigma'_d}\right) + \eta
$$

**Reward field:**

$$
r'_i := g_A(z_{r,i}) + \eta = g_A\left(\frac{r_i - \mu_r}{\sigma'_r}\right) + \eta
$$

where η > 0 is a floor constant.

**Fitness potential:**

$$
V_{\text{fit},i} := (d'_i)^\beta \cdot (r'_i)^\alpha
$$

where α, β > 0 are dynamical weights.
:::

:::{prf:remark} Nature of Collective Fields
:class: note

The quantities d'_i and r'_i are **neither single-particle nor pairwise** quantities. They are **collective local field values**:

1. **Local:** Associated with a specific walker i
2. **Collective:** Depend on the entire swarm through (μ_d, σ'_d, μ_r, σ'_r)
3. **Field:** Assign a value to each point in the swarm configuration space

This is a novel structure not commonly seen in standard gauge theories, where fields are typically either:
- External background fields (constant or slowly varying)
- Dynamical gauge fields (independent degrees of freedom)
- Matter fields (local particle properties)

Here, d'_i and r'_i are **auxiliary fields** determined self-consistently by the swarm configuration through a collective averaging operation.
:::

### 2.4. Cloning Score and Decision

:::{prf:definition} Cloning Score
:label: def-cloning-score-recap

After fitness values V_fit,i have been computed for all walkers, the **cloning score** for walker i with respect to cloning companion c_clone(i) = j is:

$$
S_i(j) := \frac{V_{\text{fit},j} - V_{\text{fit},i}}{V_{\text{fit},i} + \varepsilon_{\text{clone}}}
$$

where ε_clone > 0 is a regularization constant.

**Interpretation:**
- S_i(j) > 0: Companion j is fitter than i → i likely to clone (copy j's state)
- S_i(j) < 0: Walker i is fitter than j → i unlikely to clone
- S_i(j) ≈ 0: Similar fitness → neutral cloning pressure
:::

### 2.5. Summary: Complete Causal Flow

The complete algorithmic flow is:

```
[Step 1] Diversity pairing: Sample c_div(i) for all i using ε_d
         → Produces companion map for diversity measurement

[Step 2] Raw distance measurement: d_i = d_alg(i, c_div(i))
         Raw reward measurement: r_i = R(x_i, v_i) - c_v_reg ||v_i||^2

[Step 3] Swarm aggregation: Compute (μ_d, σ'_d, μ_r, σ'_r) over all walkers

[Step 4] Standardization: z_d,i = (d_i - μ_d)/σ'_d, z_r,i = (r_i - μ_r)/σ'_r

[Step 5] Rescaling: d'_i = g_A(z_d,i) + η, r'_i = g_A(z_r,i) + η

[Step 6] Fitness: V_fit,i = (d'_i)^β · (r'_i)^α

[Step 7] Cloning companion: Sample c_clone(i) independently using ε_c

[Step 8] Cloning score: S_i(c_clone(i)) = (V_fit,c_clone(i) - V_fit,i) / (V_fit,i + ε)

[Step 9] Cloning decision: Compare S_i(c_clone(i)) to threshold, execute cloning
```

**Critical observation:** Steps 1-6 form a **feed-forward pipeline**. Step 7 uses the output of Step 6 but does not create feedback. This structure is **acyclic** and **computationally sound**.

---

## 3. Proposed Symmetry Redefinitions

We now present the precise mathematical definitions of the proposed symmetry structure using collective fields.

### 3.1. Proposed U(1)_fitness Symmetry

#### 3.1.1. Current Definition (Baseline)

From {doc}`03_yang_mills_noether.md` §1.2, the current U(1)_fitness structure is:

**Dressed walker state:**

$$
|\psi_i\rangle := \sum_{k \in A_t \setminus \{i\}} \psi_{ik}^{(\text{div})} |k\rangle
$$

where

$$
\psi_{ik}^{(\text{div})} = \sqrt{P_{\text{comp}}^{(\text{div})}(k|i)} \cdot e^{i\theta_{ik}^{(\text{div})}}
$$

**Amplitude (current):**

$$
\sqrt{P_{\text{comp}}^{(\text{div})}(k|i)} = \frac{\exp\left(-\frac{d_{\text{alg}}^2(i,k)}{2\epsilon_d^2}\right)}{\sqrt{\sum_{j} \exp\left(-\frac{d_{\text{alg}}^2(i,j)}{2\epsilon_d^2}\right)}}
$$

**Phase (current):**

$$
\theta_{ik}^{(\text{div})} = -\frac{d_{\text{alg}}^2(i,k)}{2\epsilon_d^2 \hbar_{\text{eff}}}
$$

**Properties:**
- Amplitude and phase both pairwise: depend on (i,k)
- Based on raw algorithmic distance
- Clear geometric interpretation

#### 3.1.2. Proposed Definition

**Amplitude (unchanged):**

$$
\sqrt{P_{\text{comp}}^{(\text{div})}(k|i)} \quad \text{(same as current)}
$$

**Phase (new):**

$$
\theta_i^{(\text{new})} := \frac{(d'_i)^\beta}{\hbar_{\text{eff}}}
$$

where d'_i is the collective diversity field from {prf:ref}`def-collective-fields`.

**Alternative formulation:**

$$
\psi_{ik}^{(\text{div, new})} = \sqrt{P_{\text{comp}}^{(\text{div})}(k|i)} \cdot e^{i\theta_i^{(\text{new})}}
$$

Note: The phase is now a function of walker i only (though it depends on entire swarm through d'_i).

**Rationale:**
1. Uses processed information (d'_i) rather than raw distance
2. Phase represents walker's **relative position in fitness landscape**
3. Collective dependence: phase changes when swarm configuration changes
4. Removes arbitrary scale (z-score normalization)

:::{prf:remark} Structure of Proposed U(1) Phase
:class: note

The proposed phase θ_i^(new) has a fundamentally different structure from the current phase:

**Current:** θ_ik is **pairwise** — encodes relative geometry between i and k
**Proposed:** θ_i is a **collective local value** — encodes i's position relative to entire swarm

This is not a single-particle property (depends on μ_d, σ'_d from all walkers) nor a two-particle property (not symmetric/antisymmetric under i ↔ k). It is a **collective field** evaluated at walker i.

Whether this can support a gauge symmetry requires analysis of transformation properties.
:::

### 3.2. Proposed SU(2)_weak Symmetry

#### 3.2.1. Current Definition (Baseline)

From {doc}`03_yang_mills_noether.md` §1.2-1.3, the current SU(2) structure is:

**Interaction state:**

$$
|\Psi_{ij}\rangle = |↑\rangle \otimes |\psi_i\rangle + |↓\rangle \otimes |\psi_j\rangle \in \mathbb{C}^2 \otimes \mathbb{C}^{N-1}
$$

where |↑⟩ and |↓⟩ represent "cloner" and "target" roles (weak isospin).

**Phase (current):**

$$
\theta_{ij}^{(\text{SU}(2))} = -\frac{d_{\text{alg}}^2(i,j)}{2\epsilon_c^2 \hbar_{\text{eff}}}
$$

Based on raw algorithmic distance with cloning interaction scale ε_c.

#### 3.2.2. Proposed Definition

**Amplitude (new):**

$$
A_{ij}^{(\text{pairing})} := P_{\text{pairing}}(i \leftrightarrow j \mid \text{diversity pairing})
$$

This is the probability that walkers i and j are paired together in the diversity pairing operator ({prf:ref}`def-diversity-pairing-recap`).

**Phase (new):**

$$
\theta_{ij}^{(\text{new})} := \frac{S_i(j)}{\hbar_{\text{eff}}} = \frac{1}{\hbar_{\text{eff}}} \cdot \frac{V_{\text{fit},j} - V_{\text{fit},i}}{V_{\text{fit},i} + \varepsilon_{\text{clone}}}
$$

where S_i(j) is the cloning score from {prf:ref}`def-cloning-score-recap`.

**Rationale:**
1. Phase directly encodes **fitness comparison** (physical interaction strength)
2. S_i(j) > 0 → j is fitter → strong attraction for cloning
3. S_i(j) < 0 → i is fitter → repulsion (no cloning)
4. Uses algorithm's actual decision variable
5. Antisymmetry: S_i(j) ≈ -S_j(i) (with regularization)

:::{prf:remark} Non-Recursion in SU(2) Phase
:class: important

The phase θ_ij^(new) = S_i(j) / ℏ_eff uses V_fit,i which was computed from the **diversity companion** c_div(i), NOT the cloning companion c_clone(i).

**Causal independence:**
- V_fit,i computed using c_div(i) (Step 6)
- Cloning score S_i(j) uses V_fit plus independently sampled c_clone(i) = j (Step 8)

If we interpret the phase as applying to the (i, j = c_clone(i)) pair, then:
- Phase uses fitness values that were computed BEFORE c_clone(i) was sampled
- No circular dependency exists

This is algorithmically sound.
:::

### 3.3. Proposed Higgs Mechanism Analog

#### 3.3.1. Current Interpretation

In the current framework ({doc}`03_yang_mills_noether.md` §2.2-2.3), the fitness potential V_fit plays a Higgs-like role:

- It appears in the effective mass term: m_eff = ⟨Ψ_ij | Ŝ_ij | Ψ_ij⟩
- It breaks fitness symmetry (distinguishes walkers by fitness)
- It gives "mass" to interactions (determines cloning probability)

However, this role is implicit—V_fit is not explicitly structured as a doublet field.

#### 3.3.2. Proposed Interpretation

**"Higgs field":** The reward collective field

$$
\Phi_{\text{Higgs}}(i) := r'_i = \left[g_A\left(\frac{r_i - \mu_r}{\sigma'_r}\right) + \eta\right]^\alpha
$$

**Properties:**
1. **Collective field:** Depends on entire swarm through (μ_r, σ'_r)
2. **Symmetry breaking:** Distinguishes walkers by relative reward
3. **Mass generation:** Enters fitness potential V_fit = (d'_i)^β · (r'_i)^α
4. **Scalar under SU(2):** Does not transform under isospin rotations

:::{prf:remark} Difference from Standard Model Higgs
:class: warning

The Standard Model Higgs field φ is an **SU(2)_L doublet**:

$$
\phi = \begin{pmatrix} \phi^+ \\ \phi^0 \end{pmatrix}
$$

with vacuum expectation value ⟨φ^0⟩ = v / √2.

The proposed Φ_Higgs(i) = r'_i is a **scalar singlet**—it does not transform under SU(2). This is a significant structural difference.

**Consequence:** r'_i can break a global symmetry (all walkers get uniform shift in fitness), but it cannot spontaneously break a local SU(2) gauge symmetry via the Higgs mechanism in the SM sense. The analogy is conceptual (field giving mass) but not structural (doublet breaking gauge symmetry).
:::

### 3.4. Complete Proposed Structure

Combining all elements, the proposed gauge structure is:

**U(1)_fitness:**
- Amplitude: √P_comp^(div)(k|i) [pairwise, from raw distances]
- Phase: θ_i = (d'_i)^β / ℏ_eff [collective field]

**SU(2)_weak:**
- Amplitude: A_ij^(pairing) [probability of pairing in diversity matching]
- Phase: θ_ij = S_i(j) / ℏ_eff [fitness comparison via cloning score]

**Higgs:**
- Field: Φ_Higgs(i) = (r'_i)^α [collective reward field]
- Couples to: Enters fitness V_fit = (d'_i)^β · (r'_i)^α

**Key novelty:** Phases and "Higgs field" are all **collective fields**—locally defined but globally influenced.

---

## 4. Viability Analysis

We now perform a critical analysis of the proposed redefinition from two perspectives:
1. **Algorithmic viability** — Is it mathematically well-defined and computationally sound?
2. **Gauge theory interpretation** — Does it represent a valid local gauge theory?

### 4.1. Algorithmic Viability Assessment

#### 4.1.1. Resolution of Initial Concerns

**Concern #1: "U(1) phase depends only on single walker" — RESOLVED**

**Initial objection (Gemini):** The phase θ_i = f(d'_i) depends only on walker i, not on a pair (i,k), violating the requirement that gauge phases be pairwise and antisymmetric.

**Resolution:** This objection was based on a category error. The quantity d'_i is not a "single-walker property" in the usual sense—it is a **collective local field value** that depends on the entire swarm state through the z-score operation:

$$
d'_i = g_A\left(\frac{d_i - \mu_d}{\sigma'_d}\right) + \eta
$$

where μ_d = ⟨d⟩ and σ'_d = σ'(d) are swarm-wide statistics.

**Key insight:** d'_i changes when ANY walker's position or companion changes, because this affects (μ_d, σ'_d). It is a field value at point i that is determined by the global configuration.

**Verdict:** ✅ The phase is well-defined. Whether it can support a gauge symmetry is a separate question about **transformation properties** (addressed in §4.2).

**Concern #2: "Circular dependency in SU(2) phase" — RESOLVED**

**Initial objection (Gemini):** The phase θ_ij = S_i(j) / ℏ_eff depends on V_fit, which itself depends on the phase through the measurement pipeline, creating a circular dependency (Phase → Pipeline → V_fit → S_i → Phase).

**Resolution:** This objection was based on a misunderstanding of the algorithm structure. The algorithm employs **two independent companion selections**:

1. **Diversity companion** c_div(i): Used to compute d_i = d_alg(i, c_div(i)), which feeds into V_fit
2. **Cloning companion** c_clone(i): Sampled independently AFTER V_fit is computed

The cloning score is:

$$
S_i(c_{\text{clone}}(i)) = \frac{V_{\text{fit},c_{\text{clone}}(i)} - V_{\text{fit},i}}{V_{\text{fit},i} + \varepsilon_{\text{clone}}}
$$

**Causal flow (no circle):**

```
Diversity pairing → c_div(i) → d_i → z_d,i → d'_i → V_fit,i
                                                        ↓
                        Cloning companion c_clone(i) + V_fit,i → S_i(c_clone(i))
```

V_fit is computed from c_div (diversity), S_i uses c_clone (independent). No feedback loop exists.

**Verdict:** ✅ No circular dependency. The algorithm is acyclic and computable in a single forward pass.

**Concern #3: "r'_i is a singlet, not doublet" — PARTIALLY RESOLVED**

**Initial objection (Gemini):** The SM Higgs is an SU(2) doublet, but r'_i is a scalar singlet, so the analogy is structurally weak.

**Resolution:** This is **correct as stated**. The field r'_i does not transform under SU(2) like the SM Higgs doublet φ = (φ⁺, φ⁰). However, the objection overstated the problem by calling r'_i a "single-walker singlet."

**Clarification:** r'_i is a **collective field**:

$$
r'_i = g_A\left(\frac{r_i - \mu_r}{\sigma'_r}\right) + \eta
$$

It depends on the entire swarm through (μ_r, σ'_r). It is scalar under SU(2) (doesn't transform), but it is not a local single-particle property.

**Verdict:** ⚠️ The structural mismatch with SM Higgs remains. r'_i can break global fitness symmetry (uniform fitness shift) but cannot spontaneously break local SU(2) gauge symmetry via the Higgs mechanism. The analogy is **conceptual** (field giving mass) but not **structural** (doublet symmetry breaking).

#### 4.1.2. Computational Soundness

**Computability:** ✅
- All quantities (d'_i, r'_i, S_i(j)) are computable in polynomial time
- Pipeline is feed-forward (no iteration required)
- No numerical instabilities identified

**Algorithmic convergence:** ✅ (Expected)
- Changing the phase source (raw d_alg → processed d'_i) does not alter the pipeline structure
- Keystone Principle ({doc}`../1_euclidean_gas/03_cloning.md` Chapter 7-8) should still hold
- Convergence proofs depend on pipeline properties (bounded operators, Lipschitz continuity, N-uniform gaps), not on phase interpretation
- **Caveat:** Formal re-proof required to verify, but no obvious obstruction

**Efficiency:** ✅
- Same computational complexity as current framework (O(N²) per timestep)
- No additional overhead from using processed values

**Summary:** ✅ The proposed structure is **algorithmically viable**—well-defined, computable, and expected to preserve convergence properties.

### 4.2. Gauge Theory Interpretation: The Critical Issue

The central unresolved question is whether the proposed structure can support a **local gauge theory** in the mathematical physics sense. This requires analyzing the **transformation properties** of the collective fields.

#### 4.2.1. Gauge Invariance vs Gauge Covariance

**Fundamental distinction in gauge theory:**

1. **Gauge-invariant quantity:** Does not change under gauge transformation
   - Example: Electromagnetic field strength F_μν = ∂_μ A_ν - ∂_ν A_μ
   - Example: Particle number, probability amplitudes |ψ|²

2. **Gauge-covariant field:** Transforms in a specific way that cancels gauge transformation of matter fields
   - Example: Gauge potential A_μ → A_μ + ∂_μ α (under ψ → e^(iα) ψ)
   - Example: Covariant derivative D_μ = ∂_μ - i e A_μ

**Requirement for local gauge symmetry:**
- Gauge fields (like A_μ) must be **gauge-covariant**, not gauge-invariant
- Physical observables (like |ψ|², F_μν) must be **gauge-invariant**
- The Lagrangian (or action) must be gauge-invariant when constructed from covariant objects

#### 4.2.2. Gemini's Critical Observation

After clarifications, Gemini 2.5 Pro identified a new critical issue:

:::{prf:proposition} Proposed Phases Appear to Be Gauge-Invariant
:label: prop-phase-invariance

The collective fields d'_i and r'_i, and consequently the proposed phases θ_i and θ_ij, are constructed entirely from **physical observables** and appear to be **gauge-invariant**.

**Argument:**

1. The algorithmic distance d_alg(i,k) = √(||x_i - x_k||² + λ_alg ||v_i - v_k||²) is a **physical separation** in phase space. It depends only on observable walker states, not on any gauge choice. Therefore: **d_alg is gauge-invariant**.

2. The raw reward r_i = R(x_i, v_i) - c_v_reg ||v_i||² is a **physical observable**. The environment reward R and kinetic energy are measurable quantities. Therefore: **r_i is gauge-invariant**.

3. The swarm statistics μ_d, σ'_d, μ_r, σ'_r are computed as averages and standard deviations of the gauge-invariant quantities d_i and r_i. Averaging gauge-invariant quantities yields gauge-invariant statistics. Therefore: **μ_d, σ'_d, μ_r, σ'_r are gauge-invariant**.

4. The z-scores z_d,i = (d_i - μ_d) / σ'_d and z_r,i = (r_i - μ_r) / σ'_r are ratios of gauge-invariant quantities. Therefore: **z_d,i and z_r,i are gauge-invariant**.

5. The rescale function g_A is a fixed mathematical function. Applying it to a gauge-invariant quantity yields a gauge-invariant result. Therefore: **d'_i = g_A(z_d,i) + η and r'_i = g_A(z_r,i) + η are gauge-invariant**.

6. The fitness potential V_fit,i = (d'_i)^β · (r'_i)^α and cloning score S_i(j) = (V_fit,j - V_fit,i) / (V_fit,i + ε) are constructed from gauge-invariant fields. Therefore: **V_fit,i and S_i(j) are gauge-invariant**.

**Conclusion:** The proposed phases θ_i = (d'_i)^β / ℏ_eff and θ_ij = S_i(j) / ℏ_eff are **gauge-invariant**, not gauge-covariant.

**Implication:** A theory with gauge-invariant phases cannot support a non-trivial local gauge symmetry. If ψ_i → e^(iα_i(x)) ψ_i (local phase transformation), there is no compensating transformation of the phase θ_i to maintain covariance. The symmetry is broken—or never existed as a local gauge symmetry.
:::

#### 4.2.3. Critical Evaluation of Gemini's Argument

**Strengths of the argument:**
- ✅ Logically rigorous chain of reasoning
- ✅ Based on standard gauge theory principles
- ✅ Correctly identifies that gauge fields must transform non-trivially
- ✅ Highlights fundamental structural difference from SM gauge theories

**Potential subtleties and objections:**

**Objection 1: "What is being gauged?"**

The argument assumes we are gauging the U(1) phase of walker wavefunctions: ψ_i → e^(iα_i) ψ_i. But in the Fragile Gas framework, the "wavefunctions" |ψ_i⟩ are defined by the amplitude and phase structure itself:

$$
|\psi_i\rangle = \sum_k \sqrt{P_{\text{comp}}(k|i)} \cdot e^{i\theta_i} |k\rangle
$$

If we redefine θ_i → (d'_i)^β / ℏ_eff, have we changed the **definition** of the wavefunction, or have we proposed a **gauge transformation**? These are distinct operations.

**Resolution:** If the proposal is to **redefine** what we mean by the phase (changing the dictionary between algorithm and physics), then gauge covariance is not required—we're proposing a different mapping. However, if we claim this represents the **same physics** under a gauge transformation, then Gemini's objection holds.

**Objection 2: "Gauge symmetry might be emergent, not fundamental"**

Perhaps the symmetry emerges from the algorithm dynamics rather than being imposed externally. In emergent gauge theories (e.g., in condensed matter), the "gauge fields" are often composite objects constructed from matter fields. These can have different transformation properties than fundamental gauge fields.

**Resolution:** This is a valid point and suggests the **mean-field interpretation** (Section 5.2) is more appropriate. The collective fields d'_i and r'_i would be **auxiliary emergent fields** rather than fundamental gauge bosons.

**Objection 3: "The z-score operation is a gauge-fixing procedure"**

The z-score operation z_i = (v_i - μ_v) / σ'_v removes scale and origin ambiguity. Perhaps this is analogous to choosing a gauge (fixing gauge freedom)?

**Analysis:** Gauge-fixing in QFT selects a representative from each gauge orbit to make path integrals well-defined (e.g., Lorenz gauge, Coulomb gauge). The z-score operation is similar in spirit—it removes degeneracy in how we describe the system. However, standard gauge-fixing is:
- Applied to the gauge field A_μ (not to observables)
- A choice made by the analyst (not part of the dynamics)
- Removes redundancy in description (doesn't change physics)

The z-score operation is:
- Applied to observable quantities (d_i, r_i)
- Built into the algorithm dynamics
- Changes which information the algorithm uses

**Resolution:** The z-score operation has some gauge-fixing-like properties (removes scale/origin freedom), but it operates on observables and is dynamical. It might be better understood as a **background field subtraction** or **statistical gauge** rather than traditional gauge-fixing.

**Objection 4: "Statistical gauge symmetry?"**

Perhaps the relevant symmetry is not U(1) phase rotation but **affine transformation** of the raw values:

$$
d_i \to a \cdot d_i + b
$$

The z-score is invariant under global affine transformations (a and b affect μ_d and σ_d equally). This is a **global Aff(1,ℝ) symmetry**, not a local U(1) symmetry.

**Resolution:** This is a valid observation. The z-score operation creates invariance under a different symmetry group than originally assumed. This weakens the claim of U(1) gauge structure.

#### 4.2.4. Verdict on Gauge Theory Interpretation

**Gemini's conclusion: ⚠️ The proposed structure appears to be gauge-invariant, not gauge-covariant. Therefore, it does not straightforwardly represent a local gauge theory in the Standard Model sense.**

**My assessment: ✅ Gemini's logic is sound given standard gauge theory assumptions. However, alternative interpretations exist:**

1. **Reinterpretation (not transformation):** We're proposing a different mapping from algorithm to physics, not a gauge transformation of the same theory.

2. **Emergent gauge structure:** The fields are auxiliary/composite, not fundamental gauge bosons. This is common in condensed matter (holographic duality, gauge/gravity correspondence).

3. **Statistical gauge symmetry:** The relevant symmetry is affine transformation invariance (global Aff(1,ℝ)), not local U(1) phase rotation.

4. **Mean-field theory:** The collective fields arise from self-consistent mean-field equations, not from local gauge principle.

**Recommendation:** Do not claim the proposed structure is a **local gauge theory** without rigorous proof of non-trivial transformation properties. Instead, frame it as one of the alternative interpretations above.

---

## 5. Three Theoretical Interpretations

Given the ambiguity around gauge covariance, we present three possible theoretical frameworks for understanding the proposed symmetry structure. Each has different implications for physics interpretation and SM mapping.

### 5.1. Interpretation 1: Local Gauge Theory (Requires Proof)

**Claim:** The collective fields d'_i and r'_i are **gauge-covariant fields** that transform non-trivially under local gauge transformations, supporting a genuine local gauge symmetry.

#### 5.1.1. Requirements for Validity

To establish this interpretation, the following must be proven:

:::{prf:axiom} Local Gauge Transformation (To Be Defined)
:label: axiom-local-gauge-transformation

Define a local U(1) gauge transformation on walker states:

$$
|\psi_i(t)\rangle \to |\psi'_i(t)\rangle = e^{i\alpha_i(t)} |\psi_i(t)\rangle
$$

where α_i(t) is an arbitrary smooth function of walker index i and time t.

**Required property:** This transformation must act on the phase component of the wavefunction while preserving probability amplitudes.
:::

:::{prf:proposition} Non-Trivial Transformation of Collective Fields (To Be Proven)
:label: prop-collective-field-transformation

Under the local gauge transformation {prf:ref}`axiom-local-gauge-transformation`, prove that the collective fields transform as:

$$
d'_i \to d'_i + \Delta_i[\alpha] + O(\alpha^2)
$$

$$
r'_i \to r'_i + \Gamma_i[\alpha] + O(\alpha^2)
$$

where Δ_i[α] and Γ_i[α] are non-zero functionals depending on the gauge parameter α and its derivatives.

**Key requirement:** Δ_i and Γ_i must be **non-trivial** (not identically zero). If d'_i and r'_i are gauge-invariant (Δ_i = Γ_i = 0), the interpretation fails.
:::

:::{prf:theorem} Gauge Invariance of Physical Observables (To Be Proven)
:label: thm-observable-gauge-invariance

Assuming {prf:ref}`prop-collective-field-transformation` holds, prove that physical observables are gauge-invariant:

1. **Total cloning probability:**

$$
P_{\text{total}}(i,j) := P_{\text{clone}}(i \to j) + P_{\text{clone}}(j \to i)
$$

is gauge-invariant (as claimed in {doc}`03_yang_mills_noether.md` Proposition 1.3).

2. **Fitness potential expectation:**

$$
\langle V_{\text{fit}} \rangle := \frac{1}{k} \sum_{i \in A_t} V_{\text{fit},i}
$$

is gauge-invariant.

3. **Cloning operator action:** The post-cloning swarm distribution is gauge-invariant.
:::

:::{prf:theorem} Yang-Mills Action Construction (To Be Proven)
:label: thm-yang-mills-action

Assuming gauge covariance of collective fields, construct a gauge-invariant Yang-Mills action:

$$
S[\psi, d', r'] = \int dt \sum_i \mathcal{L}_{\text{matter}}(\psi_i, D_\mu \psi_i) + \mathcal{L}_{\text{gauge}}(F_{\mu\nu})
$$

where:
- $D_\mu$ is the gauge-covariant derivative constructed from d'_i and r'_i
- $F_{\mu\nu}$ is the field strength tensor
- The action is invariant under combined gauge transformation of ψ_i and the collective fields

Derive the equations of motion and verify consistency with Fractal Set dynamics.
:::

#### 5.1.2. Expected Challenges

**Challenge 1: Proving non-trivial transformation**

Gemini's argument suggests this will fail. Since d'_i is constructed from gauge-invariant primitives (d_alg, raw averages), deriving a non-trivial transformation requires finding hidden degrees of freedom or redefinition of "gauge."

**Possible approach:**
- Consider gauge transformation acting on the **companion map** c_div, not just phases
- Define gauge as transformation of the **reference frame** for statistics (μ_d, σ'_d)
- Reinterpret z-score operation as gauge-fixing procedure

**Challenge 2: Maintaining algorithmically dynamics**

Even if gauge covariance can be established, the gauge-transformed system must still implement the correct algorithmic dynamics (Keystone Principle, convergence).

#### 5.1.3. If Successful

**Implications:**
- ✅ Strong mathematical foundation for SM mapping
- ✅ Rigorous gauge theory structure
- ✅ Can derive conserved Noether currents
- ✅ Publishable in mathematical physics journals

**Physics accessible:**
- Yang-Mills gauge bosons (from parallel transport of d'_i, r'_i)
- Fermion interactions via covariant derivatives
- Spontaneous symmetry breaking (if doublet structure can be added)
- Standard Model observables (scattering amplitudes, etc.)

**Recommendation:** Attempt the proofs. If they succeed, this is the strongest interpretation. If they fail (as Gemini predicts), move to Interpretation 2 or 3.

### 5.2. Interpretation 2: Mean-Field / Collective Field Theory (Most Likely)

**Claim:** The collective fields d'_i and r'_i are **auxiliary emergent fields** determined self-consistently by the swarm configuration through a collective averaging mechanism. The theory is a **mean-field approximation** to a more fundamental many-body interaction, not a fundamental local gauge theory.

#### 5.2.1. Theoretical Framework

**Structure:**

1. **Fundamental degrees of freedom:** Walker states w_i = (x_i, v_i, s_i)

2. **Interactions:** Walkers interact through:
   - Algorithmic distance-weighted coupling (diversity pairing)
   - Fitness comparison (cloning)

3. **Collective fields:** d'_i and r'_i are **mean-field variables**:

$$
d'_i = d'_i[w_1, \ldots, w_N] = \text{functional of all walker states}
$$

These are not independent dynamical fields but are **determined by the walker configuration** through the z-score operation.

4. **Effective single-walker description:** Each walker i experiences an effective potential created by the collective field background:

$$
V_{\text{eff},i} = V_{\text{eff}}(x_i, v_i; [d'], [r'])
$$

where [d'] = {d'_1, ..., d'_N} denotes the collective field configuration.

#### 5.2.2. Analogy to Statistical Mechanics

**This structure is analogous to:**

**Mean-field theory in magnetism:**
- Spins interact: H = -J Σ_{ij} S_i · S_j
- Mean-field approximation: Each spin sees average field M = ⟨S⟩
- Self-consistency: M = tanh(β J M) (must be solved)
- Collective field M is auxiliary, determined by configuration

**Hartree-Fock in quantum chemistry:**
- Electrons interact via Coulomb repulsion
- Hartree-Fock: Each electron moves in average field of others
- Self-consistent field: φ_i(r) determined by {φ_j(r)}_{j≠i}
- Effective potential V_eff(r) = Σ_j ∫ |φ_j(r')|² / |r-r'| dr'

**BCS theory of superconductivity:**
- Fermions interact via phonon exchange
- Mean-field: Introduce gap parameter Δ = ⟨c_↑ c_↓⟩
- Self-consistency: Δ must satisfy gap equation
- Δ is not fundamental field, emerges from pairing

**Fragile Gas collective fields:**
- Walkers interact via fitness comparison and pairing
- Mean-field: d'_i = g_A((d_i - μ_d)/σ'_d) + η [average geometry]
- Mean-field: r'_i = g_A((r_i - μ_r)/σ'_r) + η [average reward]
- Self-consistency: (μ_d, σ'_d, μ_r, σ'_r) determined by {d'_i, r'_i}

#### 5.2.3. Symmetry Structure in Mean-Field Interpretation

**Global vs local symmetry:**

In mean-field theories, gauge structure can emerge differently:

**Global symmetry preserved:**
- If the mean-field equations preserve a global symmetry, the effective theory has that symmetry
- Example: BCS preserves U(1) charge conservation (Δ → e^(iθ) Δ for global θ)

**Local gauge symmetry emergent:**
- In some condensed matter systems, effective gauge fields emerge from mean-field treatment
- Example: Slave-boson/slave-fermion theories in strongly correlated systems
- Example: Gauge fields in quantum spin liquids
- These are **not fundamental gauge theories** but emergent low-energy descriptions

**For Fragile Gas:**
- The collective fields d'_i, r'_i preserve certain **global symmetries**:
  - Affine transformation invariance: d_i → a·d_i + b leaves z_d,i unchanged
  - Permutation symmetry: S_N acts on walker labels
- Local U(1) or SU(2) gauge symmetry is **not manifest** in the mean-field equations
- However, effective gauge-like interactions can emerge from the coupling structure

#### 5.2.4. Advantages of This Interpretation

✅ **Resolves gauge covariance issue:**
- Collective fields are not required to be gauge-covariant
- They are auxiliary fields determined by self-consistency
- Physical observables (cloning probabilities) are automatically well-defined

✅ **Matches algorithm structure:**
- The pipeline is literally computing a mean-field approximation
- z-score: walker sees average geometric environment (μ_d, σ'_d)
- Fitness V_fit: single-walker effective potential in collective background

✅ **Has strong physics precedent:**
- Mean-field theories are ubiquitous in physics
- Well-understood mathematical framework
- Standard path from mean-field to fluctuations and correlations

✅ **Explains "collective field" nature:**
- d'_i depends on all walkers because it's a mean-field quantity
- Not a gauge artifact, but the essence of mean-field approximation

#### 5.2.5. Physics Accessible Under This Interpretation

**Can describe:**
- ✅ Emergent collective modes (like phonons in crystals)
- ✅ Phase transitions (when mean-field solution bifurcates)
- ✅ Effective interactions mediated by collective fields
- ✅ Symmetry breaking (when mean-field breaks symmetry of Hamiltonian)
- ✅ Quasi-particle excitations

**Cannot describe (or describes differently):**
- ❌ Fundamental gauge bosons (collective fields are not gauge bosons)
- ❌ Local gauge invariance (not a gauge theory)
- ⚠️ Fluctuations beyond mean-field (need systematic expansion)

**Analogies:**
- Phonons in solids (collective lattice vibrations) ↔ collective fitness modes
- Plasmons in metals (collective charge oscillations) ↔ collective fitness oscillations
- Cooper pairs in superconductors ↔ cloning pairs
- Spin waves in magnets ↔ fitness waves in walker swarm

#### 5.2.6. Connection to Standard Model

**Indirect connection:**

The SM also has effective field theory descriptions at different scales. The Fragile Gas collective field theory could be:

1. **Effective low-energy theory:** Like chiral perturbation theory (effective theory of QCD at low energy)
2. **Composite gauge theory:** Like technicolor models (gauge bosons are composite)
3. **Emergent gauge theory:** Like gauge fields in string theory (emerge from gravitational dynamics)

**Key insight:** Many phenomena in nature exhibit gauge-like structure without fundamental gauge symmetry. Examples:
- Topological phases of matter
- Quantum Hall effect
- Gauge/gravity duality in holography

**Recommendation:** Frame Fragile Gas as **emergent collective field theory** with gauge-like structure, not as fundamental gauge theory.

#### 5.2.7. Research Directions Under This Interpretation

**Short-term:**
- [ ] Formalize mean-field equations explicitly
- [ ] Identify fixed points and stability
- [ ] Compute collective mode spectrum

**Medium-term:**
- [ ] Develop systematic expansion beyond mean-field (fluctuations)
- [ ] Classify phase transitions in parameter space
- [ ] Compute transport coefficients (diffusion, friction)

**Long-term:**
- [ ] Connect to emergent gauge theories in condensed matter
- [ ] Explore topological phases
- [ ] Study quantum analogs (if system is quantized)

### 5.3. Interpretation 3: Global Symmetry with State-Dependent Parameters

**Claim:** The symmetries are **global** (same transformation for all walkers), but the coupling parameters are **state-dependent** (change with swarm configuration). This is not a gauge theory but a dynamical system with feedback.

#### 5.3.1. Structure

**Global symmetry:**

Instead of local transformations ψ_i → e^(iα_i) ψ_i (different α for each i), consider **global** transformations:

$$
|\psi_i\rangle \to e^{i\alpha} |\psi_i\rangle \quad \text{(same $\alpha$ for all $i$)}
$$

This is a **global U(1) symmetry**, like electric charge conservation in QED (not a gauge symmetry like local phase in QED).

**State-dependent background:**

The "phases" θ_i = (d'_i)^β / ℏ_eff are not gauge connection terms but **external background fields** that vary with swarm configuration:

$$
\theta_i[S_t] = \theta_i(x_1, \ldots, x_N, v_1, \ldots, v_N)
$$

These are **moduli** or **parameters** that characterize the swarm state, not dynamical gauge fields.

#### 5.3.2. Analogy to Physical Systems

**Yukawa couplings in SM:**

In the Standard Model, Yukawa couplings y_f depend on which fermion f interacts with the Higgs:

$$
\mathcal{L}_{\text{Yukawa}} = -y_e \bar{e}_L \phi e_R - y_\mu \bar{\mu}_L \phi \mu_R - \ldots
$$

When the Higgs has VEV ⟨φ⟩ = v/√2, fermions get masses m_f = y_f v.

**Analog in Fragile Gas:**

The "coupling" for walker i to the collective fields is (d'_i, r'_i), which depends on swarm state:

$$
\mathcal{L}_{\text{walker},i} \sim d'_i[S_t]^\beta \cdot r'_i[S_t]^\alpha \cdot (\text{walker field})
$$

This is state-dependent coupling, like position-dependent mass in condensed matter.

**Running coupling constants:**

In QFT, coupling constants "run" with energy scale μ via renormalization group:

$$
\alpha(\mu) = \frac{\alpha(\mu_0)}{1 - \beta_0 \alpha(\mu_0) \log(\mu/\mu_0)}
$$

where β_0 is the beta function.

**Analog in Fragile Gas:**

The "couplings" (d'_i, r'_i) run with **swarm configuration** rather than energy scale. The "beta function" is the z-score transformation:

$$
d'_i(S_t) = g_A\left(\frac{d_i - \mu_d[S_t]}{\sigma'_d[S_t]}\right) + \eta
$$

#### 5.3.3. Symmetry Structure

**Conserved charges (Noether):**

Global symmetries imply conserved charges by Noether's theorem. For global U(1):

$$
Q = \sum_i \frac{\partial \mathcal{L}}{\partial \dot{\phi}_i} \quad (\text{conserved})
$$

**For Fragile Gas:**

If U(1)_fitness is a global symmetry (not local gauge), there should be a conserved fitness charge:

$$
Q_{\text{fitness}} = \sum_i (\text{some function of fitness})
$$

**Question:** Is there a conserved quantity in the algorithm?

**Answer:** Not obviously. The total fitness Σ_i V_fit,i is not conserved—it changes due to cloning. However, there might be subtle conserved quantities related to information or entropy.

#### 5.3.4. Advantages of This Interpretation

✅ **Simpler mathematical structure:**
- Global symmetries are easier to analyze than local gauge symmetries
- No need to prove gauge covariance

✅ **State-dependent parameters are common in physics:**
- Position-dependent effective mass in semiconductors
- Dielectric constant varying with material
- Running couplings in QFT

✅ **Matches algorithmic feedback structure:**
- Algorithm adapts parameters based on current state
- Feedback is natural feature, not bug

#### 5.3.5. Limitations

❌ **Weaker connection to Standard Model:**
- SM is a local gauge theory, not global symmetry
- Less structural correspondence

⚠️ **Less novel:**
- State-dependent parameters are not new physics
- Doesn't explain emergent gauge structure

#### 5.3.6. When This Interpretation Is Most Appropriate

**Best if:**
- Gauge covariance cannot be established
- Mean-field interpretation is too complex
- Want simple effective description

**Not best if:**
- Goal is direct SM correspondence
- Want to understand emergent gauge structure

### 5.4. Comparison of Three Interpretations

| Feature | Local Gauge | Mean-Field | Global Symmetry |
|---------|-------------|------------|-----------------|
| **Math rigor** | Highest (if proven) | High | Moderate |
| **SM connection** | Strong | Indirect | Weak |
| **Novelty** | High | Very High | Low |
| **Difficulty** | Very High | Moderate | Low |
| **Likelihood** | Low (per Gemini) | High | Moderate |
| **Phase covariance** | Required | Not required | Not required |
| **Collective fields** | Gauge bosons | Auxiliary fields | Background fields |
| **Testability** | Requires proofs | Computable | Observable |

**Recommended path:**
1. **Attempt Interpretation 1** (local gauge): Try to prove gauge covariance
2. **If fails:** Adopt **Interpretation 2** (mean-field) as primary framework
3. **Use Interpretation 3** (global symmetry) for simpler effective descriptions

---

## 6. Standard Model Mapping Assessment

### 6.1. Current Framework → Standard Model

**Current gauge structure** ({doc}`03_yang_mills_noether.md`):

| Fragile Gas | Standard Model | Correspondence |
|-------------|---------------|----------------|
| S_N permutation | - | Unique to Fragile (discrete gauge) |
| U(1)_fitness | U(1)_Y hypercharge | Both global/abelian, phase from geometry |
| SU(2)_weak | SU(2)_L weak isospin | Doublet structure (cloner/target ↔ up/down) |
| V_fit | Higgs φ | Implicit symmetry-breaking role |

**Physics accessible:**
- ✅ Yang-Mills action construction
- ✅ Gauge boson analogs (W^±, Z^0-like from SU(2))
- ✅ Conserved Noether currents
- ⚠️ Fermion masses (not explicit Yukawa mechanism)
- ❌ Electroweak symmetry breaking (no doublet VEV)

### 6.2. Proposed Framework → Standard Model

**Proposed structure** (this document):

| Fragile Gas | Standard Model | Correspondence |
|-------------|---------------|----------------|
| d'_i collective field | ? | Collective geometric field (no clear SM analog) |
| r'_i collective field | Higgs-like? | Scalar singlet (not doublet!) |
| S_i(j) cloning score | Interaction vertex | Direct fitness-to-force mapping |

**Under mean-field interpretation:**

| Fragile Gas | Physics | Correspondence |
|-------------|---------|----------------|
| d'_i | Phonon-like mode | Collective geometric excitation |
| r'_i | Order parameter | Symmetry-breaking field (like magnetization) |
| V_fit = (d'_i)^β · (r'_i)^α | Effective potential | Like Landau free energy |

### 6.3. What Can Be Described

#### 6.3.1. Under Gauge Theory Interpretation (if proven)

**Accessible physics:**
- ✅ Gauge bosons from parallel transport
- ✅ Fermion-gauge coupling via covariant derivatives
- ✅ Conserved currents (Noether)
- ✅ Wilson loops, holonomy
- ⚠️ Mass generation (but not via Higgs doublet mechanism)

**Example calculation: W boson mass analog**

In SM: $M_W = \frac{1}{2} g v$ where g is SU(2) coupling, v is Higgs VEV.

In Fragile Gas: Analog would be:
- g → coupling strength from S_i(j) statistics
- v → average fitness ⟨V_fit⟩

$$
M_W^{\text{analog}} \sim \frac{1}{\hbar_{\text{eff}}} \sqrt{\langle S_i(j)^2 \rangle} \cdot \langle V_{\text{fit}} \rangle
$$

This is computable but not derived from doublet mechanism.

#### 6.3.2. Under Mean-Field Interpretation

**Accessible physics:**
- ✅ Collective excitations (phonon-like modes in fitness landscape)
- ✅ Phase transitions (bifurcations in mean-field solution)
- ✅ Effective interactions (mediated by d'_i, r'_i fields)
- ✅ Symmetry breaking (spontaneous or explicit)
- ✅ Transport properties (diffusion, friction coefficients)

**Example calculation: Collective mode frequency**

Linearize around mean-field solution:

$$
d'_i(t) = \bar{d}' + \delta d'_i(t)
$$

Collective modes: oscillations δd'_i(t) ~ e^(iωt) with frequency ω determined by:

$$
\omega^2 = \frac{\partial^2 F}{\partial (d')^2} \bigg|_{\bar{d}'}
$$

where F is effective free energy (Lyapunov function).

### 6.4. What Cannot Be Described

#### Common limitations (both interpretations):

❌ **True Higgs mechanism:** r'_i is scalar singlet, not SU(2) doublet
- Cannot replicate VEV structure φ = (0, v/√2)^T
- Cannot generate W/Z masses via spontaneous breaking

❌ **Fermion mass hierarchy:** No Yukawa coupling structure
- SM: m_f = y_f v (different y_f for each fermion)
- Fragile: All walkers use same V_fit formula

❌ **Electroweak unification:** No SU(2) × U(1) → U(1)_em pattern
- Would need doublet structure and mixing angle

❌ **Gauge boson self-interactions:** No F^a_μν F^{μν}_a kinetic term
- Would need field strength from collective fields

#### Mean-field specific limitations:

❌ **Fundamental particles:** d'_i, r'_i are composite, not elementary
❌ **High-energy behavior:** Mean-field breaks down at small scales
❌ **Quantum coherence:** Classical mean-field approximation

### 6.5. Concrete Example: Computing an Analog Observable

**Question:** Can we compute a "scattering amplitude" analog?

**SM process:** e^- μ^+ → e^- μ^+ (elastic scattering via photon exchange)

**Fragile Gas analog:** Walker i interacting with walker j via fitness comparison

**Setup:**
- Initial state: Walkers i and j at positions (x_i, x_j) with fitnesses (V_i, V_j)
- Interaction: Cloning probability P(i clones from j)
- Final state: Post-cloning configuration

**Amplitude:**

$$
\mathcal{M}_{ij} \sim \sqrt{P_{\text{comp}}(j|i)} \cdot e^{i\theta_{ij}} \cdot S_i(j)
$$

where:
- √P_comp: Propagator (distance-dependent coupling)
- e^(iθ_ij): Phase factor (from proposed structure)
- S_i(j): Vertex (fitness comparison)

**Cross-section analog:**

$$
\sigma_{ij} \sim |\mathcal{M}_{ij}|^2 = P_{\text{comp}}(j|i) \cdot S_i(j)^2
$$

**Physical interpretation:** Probability that walker i clones from j, weighted by fitness difference squared.

**Comparison to SM:**
- SM: σ ~ α^2 / s (where α is fine structure constant, s is center-of-mass energy)
- Fragile: σ ~ exp(-d^2/ε^2) · (ΔV/V)^2 (distance and fitness suppression)

**Similarity:** Both have propagator × coupling^2 structure
**Difference:** Fragile has distance suppression, not energy dependence

### 6.6. Assessment Summary

**Best SM correspondence:**
- **Current framework** has cleaner gauge structure (phases from raw geometry)
- **Proposed framework** has more algorithmic naturalness (phases from processed info)

**For simulating SM:**
- If goal is structural fidelity → current framework better
- If goal is emergent dynamics → proposed framework more interesting

**For novel physics:**
- Proposed framework opens new directions (collective field theory, emergent gauge)
- More suitable for condensed matter analogs than particle physics

---

## 7. Conclusions and Recommendations

### 7.1. Summary of Findings

#### Algorithmic Viability

✅ **VIABLE:** The proposed symmetry redefinition is mathematically well-defined and computationally sound.

**Key results:**
1. **No circular dependency:** Two independent companion selections (diversity vs cloning) prevent recursion
2. **Collective fields well-defined:** d'_i and r'_i are computable functions of entire swarm state
3. **Expected convergence preservation:** Pipeline structure unchanged, Keystone Principle likely still holds
4. **Computational efficiency:** Same O(N²) complexity as current framework

**Resolution of initial concerns:**
- "U(1) phase not pairwise" → Resolved: It's a collective field (novel but valid)
- "Circular dependency" → Resolved: Two companion selections are independent
- "r'_i is singlet" → Partially: It's collective but still scalar (not doublet)

#### Gauge Theory Interpretation

⚠️ **UNCLEAR:** Whether the structure represents a local gauge theory requires proof of gauge covariance.

**Critical issue (Gemini's observation):**
- Proposed phases θ_i = (d'_i)^β / ℏ_eff and θ_ij = S_i(j) / ℏ_eff are constructed from physical observables
- Physical observables are gauge-invariant
- Therefore phases appear to be gauge-invariant, not gauge-covariant
- **Implication:** May not support local gauge symmetry in SM sense

**Three possible interpretations:**
1. **Local gauge theory:** Requires proving d'_i transforms non-trivially (challenging)
2. **Mean-field theory:** Collective fields are auxiliary, not fundamental gauge bosons (most likely)
3. **Global symmetry:** State-dependent parameters, weaker SM connection (simple alternative)

#### Standard Model Mapping

⚠️ **INDIRECT:** Connection to SM electroweak theory is conceptual, not structural.

**Strengths:**
- ✅ Uses processed algorithmic information (d'_i, r'_i)
- ✅ Direct fitness-to-interaction mapping (S_i(j) as phase)
- ✅ Explicit symmetry-breaking role for reward channel

**Limitations:**
- ❌ r'_i is scalar singlet (not SU(2) doublet like SM Higgs)
- ❌ Cannot replicate electroweak spontaneous symmetry breaking
- ⚠️ Gauge covariance unclear (phases may be invariant)

**Best analog:** Mean-field collective theory (like phonons in solids) rather than fundamental gauge theory (like QED/QCD)

### 7.2. Final Verdict

**Split verdict reflecting algorithmic vs physics interpretation:**

**Algorithmic Level:** ✅ **VIABLE**
- Well-defined, computable, preserves convergence
- Represents algorithm's processed perception better than current framework
- Recommended for implementation and testing

**Physics Interpretation:** ⚠️ **REFRAME NEEDED**
- Do not claim "local gauge theory" without proof of gauge covariance
- Frame as "emergent collective field theory" instead
- SM analogy is conceptual (symmetry breaking, mass generation) not structural (doublet mechanism)

### 7.3. Recommendations

#### Short-Term (1-3 months)

**1. Test gauge covariance:**
- Define explicit local gauge transformation on walker states
- Compute how d'_i, r'_i transform
- If invariant → abandon Interpretation 1 (local gauge)
- If covariant → pursue rigorously

**2. Formalize mean-field interpretation:**
- Write effective free energy/Lyapunov function in terms of collective fields
- Derive mean-field equations explicitly
- Identify fixed points and stability

**3. Numerical experiments:**
- Implement proposed structure in codebase
- Compare with current framework on benchmarks
- Measure collective mode spectrum

#### Medium-Term (3-6 months)

**4. Develop one interpretation fully:**
- If gauge covariance proven → pursue Interpretation 1 (high-risk, high-reward)
- If not → develop Interpretation 2 (mean-field) as primary framework
- Use Interpretation 3 (global symmetry) for simple effective descriptions

**5. Re-prove convergence:**
- Verify Keystone Principle holds with new phase structure
- Re-establish Wasserstein contraction (if possible)
- Prove QSD convergence

**6. Connect to established physics:**
- Survey emergent gauge theories in condensed matter
- Find closest analogs (phonons, plasmons, spin waves)
- Develop dictionary between Fragile Gas and condensed matter

#### Long-Term (6-12 months)

**7. Physics applications:**
- Compute collective mode spectrum
- Study phase transitions
- Calculate transport coefficients
- Explore topological phases

**8. Publication strategy:**
- **If Interpretation 1 succeeds:** Submit to mathematical physics journal (Annals of Mathematical Physics, Communications in Mathematical Physics)
- **If Interpretation 2:** Submit to interdisciplinary journal (Physical Review X, PNAS, Nature Physics)
- **Frame as:** "Emergent collective field theory in algorithmic optimization" not "Standard Model from algorithm"

### 7.4. Open Questions

**Fundamental:**
1. Are d'_i and r'_i gauge-invariant or gauge-covariant?
2. What is the correct mathematical framework (gauge theory, mean-field, or global symmetry)?
3. Is there a conserved Noether charge for U(1)_fitness?

**Algorithmic:**
4. Does convergence rate change with new phase structure?
5. Are there parameter regimes where proposed structure outperforms current?
6. Can we observe collective modes experimentally in algorithm behavior?

**Physics:**
7. What condensed matter systems are closest analogs?
8. Can emergent gauge structure arise from collective fields?
9. Is there a path from mean-field to fundamental gauge theory?

### 7.5. Final Thoughts

The proposed symmetry redefinition reveals a deep and subtle structure in the Fragile Gas algorithm. While it may not be a "local gauge theory" in the strict Standard Model sense, it represents something potentially more interesting: an **emergent collective field theory** where the algorithm's statistical processing (z-scores) creates auxiliary fields that mediate interactions.

**Key insight:** The measurement pipeline is not just data processing—it's **dynamically generating an effective field theory description** of the swarm. The collective fields d'_i and r'_i are the algorithm's intrinsic "effective fields" that encode how each walker perceives and interacts with the collective environment.

This is reminiscent of:
- **Emergent gauge theories** in quantum spin liquids
- **Holographic duality** (bulk fields emerge from boundary dynamics)
- **Gauge/gravity correspondence** in string theory
- **Effective field theories** in condensed matter

**The proposal is valuable not because it's identical to the Standard Model, but because it reveals the Fragile Gas as a natural laboratory for studying emergent collective field theories.**

**Recommended framing for future work:**
> "We propose that the Fragile Gas algorithm generates an emergent collective field theory through its measurement pipeline. The rescaled outputs d'_i and r'_i function as auxiliary mean-field variables that encode each walker's position in the fitness landscape relative to the swarm. While this structure exhibits gauge-like features (phase factors, interaction vertices), it is most appropriately understood as a mean-field approximation to the algorithm's many-body dynamics, analogous to phonons in solids or Cooper pairs in superconductors."

---

## Appendices

### Appendix A: Z-Score Transformation Properties

:::{prf:proposition} Affine Invariance of Z-Score
:label: prop-affine-invariance

The z-score transformation is invariant under global affine transformations of the raw values.

**Statement:** Let z_i = (v_i - μ_v) / σ'_v be the z-score of value v_i. Under the affine transformation:

$$
v_i \to \tilde{v}_i = a \cdot v_i + b \quad (\text{all } i)
$$

the z-score is invariant: $\tilde{z}_i = z_i$.

**Proof:**

New mean:

$$
\tilde{\mu}_v = \frac{1}{N} \sum_i \tilde{v}_i = \frac{1}{N} \sum_i (a v_i + b) = a \mu_v + b
$$

New standard deviation:

$$
\tilde{\sigma}_v^2 = \frac{1}{N} \sum_i (\tilde{v}_i - \tilde{\mu}_v)^2 = \frac{1}{N} \sum_i (a v_i + b - a\mu_v - b)^2 = a^2 \sigma_v^2
$$

Therefore $\tilde{\sigma}'_v = |a| \sigma'_v$ (assuming regularization scales linearly).

New z-score:

$$
\tilde{z}_i = \frac{\tilde{v}_i - \tilde{\mu}_v}{\tilde{\sigma}'_v} = \frac{(a v_i + b) - (a\mu_v + b)}{|a| \sigma'_v} = \frac{a(v_i - \mu_v)}{|a| \sigma'_v} = \frac{v_i - \mu_v}{\sigma'_v} = z_i
$$

(assuming a > 0; if a < 0, sign flips but magnitude preserved).

**Physical interpretation:** Z-score removes arbitrary additive constant (b) and multiplicative scale (a). It extracts **relative** information independent of units or offset.
:::

**Symmetry group:** The affine transformations v → av + b form the group Aff(1,ℝ). The z-score operation is Aff(1,ℝ)-invariant.

**Connection to gauge theory:** This is a **global symmetry** (same a, b for all walkers), not a local gauge symmetry. It's analogous to choosing units (scale) and origin (offset) in physics.

### Appendix B: Attempted Proof of Gauge Covariance

Here we attempt to prove that d'_i transforms non-trivially under a local gauge transformation and document where the proof fails.

**Setup:** Define local U(1) gauge transformation:

$$
|\psi_i(t)\rangle \to |\psi'_i(t)\rangle = e^{i\alpha_i(t)} |\psi_i(t)\rangle
$$

**Question:** How does d'_i transform?

**Attempt 1: Phase affects amplitude**

Suppose the transformation affects the companion selection probability:

$$
P_{\text{comp}}(k|i) \to P'_{\text{comp}}(k|i) = f[\alpha_i, \alpha_k] \cdot P_{\text{comp}}(k|i)
$$

Then the raw distance might transform as:

$$
d_i = d_{\text{alg}}(i, c_{\text{div}}(i)) \to d'_i = d_{\text{alg}}(i, c'_{\text{div}}(i))
$$

where $c'_{\text{div}}(i)$ is a different companion selected under transformed probabilities.

**Problem:** d_alg is a physical distance. It doesn't depend on any phase. The transformation α_i acts on abstract quantum amplitudes, not on positions (x_i, v_i). Therefore, d_alg is gauge-invariant:

$$
d_{\text{alg}}(i,k) \to d_{\text{alg}}(i,k) \quad \text{(unchanged)}
$$

**Attempt 2: Transformation affects statistics**

Suppose somehow the statistics (μ_d, σ'_d) transform:

$$
\mu_d \to \mu_d + \Delta\mu[\alpha], \quad \sigma'_d \to \sigma'_d + \Delta\sigma[\alpha]
$$

Then:

$$
z_{d,i} = \frac{d_i - \mu_d}{\sigma'_d} \to z'_{d,i} = \frac{d_i - (\mu_d + \Delta\mu)}{\sigma'_d + \Delta\sigma} \approx z_{d,i} - \frac{\Delta\mu}{\sigma'_d} + O(\Delta^2)
$$

This would give:

$$
d'_i \to d'_i + \text{(function of } \Delta\mu, \Delta\sigma)
$$

**Problem:** Where do Δμ and Δσ come from? The statistics are computed from d_i values:

$$
\mu_d = \frac{1}{N} \sum_{j=1}^N d_j
$$

If each d_j is gauge-invariant, then μ_d is also gauge-invariant:

$$
\mu_d \to \mu_d \quad \text{(unchanged)}
$$

Similarly for σ'_d. Therefore, Δμ = Δσ = 0.

**Conclusion:** d'_i = g_A((d_i - μ_d)/σ'_d) + η is a function of gauge-invariant quantities only. Under standard assumptions, **d'_i is gauge-invariant**.

**Where could this fail?**
1. If gauge transformation affects **companion map** c_div(i), not just phases
2. If we redefine "gauge" to act on statistical reference frame (μ_d, σ'_d)
3. If there are hidden degrees of freedom coupling to phases

These are non-standard and would require new theoretical framework.

### Appendix C: Comparison Table (Current vs Proposed)

| Aspect | Current Framework | Proposed Framework |
|--------|------------------|-------------------|
| **U(1) phase source** | Raw d_alg^2(i,k) | Collective field (d'_i)^β |
| **U(1) amplitude** | √P_comp(k\|i) | √P_comp(k\|i) [same] |
| **SU(2) phase source** | Raw d_alg^2(i,j) | Cloning score S_i(j) |
| **SU(2) amplitude** | (implicit) | Pairing probability A_ij |
| **Higgs** | V_fit (implicit) | r'_i (explicit) |
| **Phase structure** | Pairwise | Collective local field |
| **Geometric interpretation** | Direct (raw distance) | Processed (relative position) |
| **Circular dependency** | No | No (two companions) |
| **Gauge covariance** | (assumed) | Unclear (may be invariant) |
| **SM structural match** | Better (clean phases) | Weaker (collective fields) |
| **Algorithmic naturalness** | Uses raw inputs | Uses processed outputs |
| **Convergence proofs** | Complete | Require re-derivation |
| **Novelty** | Moderate | High (collective fields) |
| **Risk** | Low (proven) | Medium (new framework) |

### Appendix D: Glossary of Terms

**Algorithmic distance:** $d_{\text{alg}}(i,j) = \sqrt{\|x_i - x_j\|^2 + \lambda_{\text{alg}} \|v_i - v_j\|^2}$ — phase-space proximity measure

**Cloning companion:** Independently sampled walker c_clone(i) used for cloning decision

**Cloning score:** $S_i(j) = (V_{\text{fit},j} - V_{\text{fit},i}) / (V_{\text{fit},i} + \varepsilon)$ — fitness comparison

**Collective field:** Quantity depending on entire swarm (e.g., d'_i via statistics)

**Diversity companion:** Walker c_div(i) paired for diversity measurement

**Gauge-covariant:** Transforms to compensate matter field transformation

**Gauge-invariant:** Unchanged under gauge transformation

**Local gauge symmetry:** Symmetry under position-dependent transformations

**Global symmetry:** Symmetry under uniform transformations

**Mean-field theory:** Approximation where particle sees average environment

**Rescaling:** Nonlinear map g_A : ℝ → (0,2) applied to z-scores

**Z-score:** Standardized value $z_i = (v_i - \mu_v) / \sigma'_v$

---

**End of Document**

*For questions or feedback on this analysis, consult the framework documents in `docs/source/` or review the collaborative analysis protocol in `GEMINI.md`.*
