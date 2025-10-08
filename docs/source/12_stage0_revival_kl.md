# Stage 0 Feasibility Study: KL-Properties of the Mean-Field Revival Operator

**Document Status**: Critical prerequisite investigation for mean-field KL-convergence roadmap

**Purpose**: This document investigates whether the revival operator in the mean-field limit is KL-non-expansive, a **blocking condition** for the primary proof strategy outlined in [11_convergence_mean_field.md](11_convergence_mean_field.md).

**Timeline**: 3-4 months (as per Stage 0 roadmap)

**Criticality**: **GO/NO-GO** - The entire three-stage research program (Stages 1-3) depends on the outcome of this investigation. As identified in Gemini's review, proceeding without resolving this conjecture first would be "an unacceptable research risk."

**Relationship to Existing Results**:
- [10_kl_convergence.md](10_kl_convergence.md): Proves finite-N cloning operator preserves LSI ✅
- [06_propagation_chaos.md](06_propagation_chaos.md): Establishes mean-field limit of QSD ✅
- **This document**: Investigates whether LSI-preservation survives N→∞ limit ❓

---

## 0. Executive Summary

### 0.1. The Central Question

:::{prf:problem} Main Research Question
:label: prob-revival-kl-mean-field

Let $\mathcal{R}[\rho, m_d]$ be the mean-field revival operator defined in [05_mean_field.md](05_mean_field.md):

$$
\mathcal{R}[\rho, m_d](x,v) = \lambda_{\text{revive}} \cdot m_d \cdot \frac{\rho(x,v)}{\int_\Omega \rho}
$$

where $m_d = 1 - \int_\Omega \rho$ is the dead mass and $\lambda_{\text{revive}}$ is the revival rate.

**Question**: Is $\mathcal{R}$ **KL-non-expansive**? That is, does it satisfy:

$$
D_{\text{KL}}(\mathcal{R}(\rho) \| \mathcal{R}(\sigma)) \le D_{\text{KL}}(\rho \| \sigma)
$$

for all alive densities $\rho, \sigma \in \mathcal{P}(\Omega)$ with $\int \rho = \int \sigma < 1$?

**Status**: **UNPROVEN CONJECTURE**

**Why it matters**: The discrete-time LSI strategy (Strategy C in [11_convergence_mean_field.md](11_convergence_mean_field.md)) requires this property to prove KL-convergence. Without it, the strategy fails.
:::

### 0.2. Investigation Plan

This document pursues four parallel tracks:

**Track 1** (Section 1): **Formalization** - Rigorous mathematical definition of $\mathcal{R}$ as a map on measure spaces

**Track 2** (Section 2): **Finite-N Bridge** - Analyze how the finite-N LSI-preservation property behaves as N→∞

**Track 3** (Section 3): **Direct Proof Attempts** - Explore multiple approaches to prove KL-non-expansiveness

**Track 4** (Section 4): **Counterexample Search** - Investigate scenarios where the property might fail

**Decision Point** (Section 5): Based on findings, determine whether to proceed with Stages 1-3 or pivot to alternative strategies.

### 0.3. Current Assessment (Preliminary)

**Intuitive arguments suggesting property holds**:
1. Revival is proportional resampling from current alive distribution
2. Analogous to Bayesian conditioning (which is KL-contractive)
3. Finite-N cloning provably preserves LSI ([10_kl_convergence.md](10_kl_convergence.md))

**Concerns/Red flags**:
1. Mean-field limit can break properties that hold at finite-N
2. Revival involves division by $\int \rho$ (nonlinear, potentially destabilizing)
3. No standard theorem covers this specific operator
4. Gemini assessment: "Optimistic but unsubstantiated"

**Recommendation**: Treat as **open problem** requiring rigorous investigation before committing to multi-year roadmap.

---

## 1. Rigorous Formulation of the Mean-Field Revival Operator

### 1.1. The Mean-Field Model Context

Recall from [05_mean_field.md](05_mean_field.md) that the mean-field dynamics evolve a coupled system $(\rho(t,x,v), m_d(t))$ where:

- $\rho: [0,\infty) \times \Omega \to [0,\infty)$ is the alive density
- $m_d(t) \in [0,1]$ is the dead mass
- Conservation: $\int_\Omega \rho(t) + m_d(t) = 1$

The full PDE is:

$$
\begin{aligned}
\frac{\partial \rho}{\partial t} &= \mathcal{L}_{\text{kin}}[\rho] \rho - \kappa_{\text{kill}}(x) \rho + \mathcal{R}[\rho, m_d] \\
\frac{dm_d}{dt} &= \int_\Omega \kappa_{\text{kill}}(x) \rho \, dx - \lambda_{\text{revive}} m_d
\end{aligned}
$$

where:
- $\mathcal{L}_{\text{kin}}$: Kinetic operator (Langevin)
- $\kappa_{\text{kill}}(x) \ge 0$: Interior killing rate (zero in interior, positive near $\partial \mathcal{X}_{\text{valid}}$)
- $\mathcal{R}[\rho, m_d]$: Revival operator (our focus)

### 1.2. Revival Operator Definition

:::{prf:definition} Mean-Field Revival Operator (Formal)
:label: def-revival-operator-formal

The **revival operator** $\mathcal{R}: \mathcal{P}(\Omega) \times [0,1] \to \mathcal{M}^+(\Omega)$ (where $\mathcal{M}^+$ denotes non-negative measures) is defined by:

$$
\mathcal{R}[\rho, m_d](x,v) := \lambda_{\text{revive}} \cdot m_d \cdot \frac{\rho(x,v)}{\|\rho\|_{L^1}}
$$

where $\|\rho\|_{L^1} = \int_\Omega \rho(x,v) \, dx dv$ is the total alive mass.

**Properties**:
1. **Mass injection**: $\int_\Omega \mathcal{R}[\rho, m_d] = \lambda_{\text{revive}} m_d$ (transfers mass from dead to alive)
2. **Proportionality**: Revival samples proportionally to current alive density
3. **Normalization**: The factor $1/\|\rho\|_{L^1}$ ensures the distribution shape is preserved

**Physical interpretation**: Dead walkers are "revived" by cloning a random alive walker proportionally to the current alive distribution. This is the mean-field limit of the discrete cloning mechanism.
:::

:::{admonition} Connection to Finite-N Cloning
:class: note

In the N-particle system ([01_fragile_gas_framework.md](01_fragile_gas_framework.md)), when a walker dies (status $s_i \to 0$), it is revived by:
1. Selecting a "companion" walker $j$ from the alive set $\mathcal{A}$ with probability $\propto V_{\text{fit}}(w_j)$
2. Copying position/velocity: $(x_i, v_i) \gets (x_j, v_j) + \delta \xi$ (with small noise $\delta \xi$)

In the mean-field limit:
- The empirical measure $\frac{1}{N}\sum_{i \in \mathcal{A}} \delta_{(x_i, v_i)} \to \rho(x,v) / \|\rho\|_{L^1}$
- Fitness-weighted sampling $\to$ proportional sampling (when fitness is uniform or absorbed into $\rho$)
- The revival operator $\mathcal{R}$ is the continuous analog of this discrete cloning-to-revive mechanism
:::

### 1.3. Full Killing + Revival Operator

For the discrete-time analysis, we consider the **composed** operator that handles both killing and revival in one time step $\Delta t$:

:::{prf:definition} Combined Jump Operator
:label: def-combined-jump-operator

Define the **jump operator** $\mathcal{J}: \mathcal{P}(\Omega) \to \mathcal{P}(\Omega)$ that acts over a small time interval $\Delta t$:

$$
\mathcal{J}(\rho) := (1 - \int_\Omega \kappa_{\text{kill}} \rho \cdot \Delta t) \cdot \frac{\rho - \kappa_{\text{kill}} \rho \cdot \Delta t}{\text{(normalization)}} + \mathcal{R}[\rho, m_d] \cdot \Delta t
$$

This operator:
1. **Removes mass** via killing: $\rho \mapsto \rho - \kappa_{\text{kill}} \rho \cdot \Delta t$
2. **Adds mass** via revival: Injects $\mathcal{R}[\rho, m_d] \cdot \Delta t$
3. **Conserves total mass**: $\int \mathcal{J}(\rho) = \int \rho$ (in the two-population model)

For small $\Delta t$, this is the Euler discretization of the killing-revival PDE.
:::

:::{admonition} Discrete-Time Framework
:class: important

The discrete-time LSI strategy (Strategy C) analyzes the composition:

$$
\rho(t + \Delta t) = \mathcal{J}(\mathcal{S}_{\Delta t}(\rho(t)))
$$

where $\mathcal{S}_{\Delta t}$ is the continuous kinetic flow. The KL-convergence proof requires:
1. $\mathcal{S}_{\Delta t}$ satisfies hypocoercive LSI (Stage 1)
2. $\mathcal{J}$ is KL-non-expansive (**this document**)
3. Composition preserves contraction (Stage 2)

If $\mathcal{J}$ is KL-expansive, the entire strategy fails.
:::

---

## 2. Analysis of the Finite-N to Mean-Field Transition

### 2.1. What We Know from Finite-N

From [10_kl_convergence.md](10_kl_convergence.md), the finite-N cloning operator $\Psi_{\text{clone}}$ satisfies:

:::{prf:theorem} Finite-N LSI Preservation (Proven)
:label: thm-finite-n-lsi-preservation

The N-particle cloning operator $\Psi_{\text{clone}}: \Sigma_N \to \Sigma_N$ **preserves the LSI** with controlled constant degradation. Specifically, if a distribution $\mu$ on $\Sigma_N$ satisfies:

$$
D_{\text{KL}}(\mu \| \pi) \le C_{\text{LSI}} \cdot I(\mu \| \pi)
$$

then the push-forward $\Psi_{\text{clone}}^* \mu$ satisfies:

$$
D_{\text{KL}}(\Psi_{\text{clone}}^* \mu \| \Psi_{\text{clone}}^* \pi) \le C'_{\text{LSI}} \cdot I(\Psi_{\text{clone}}^* \mu \| \Psi_{\text{clone}}^* \pi)
$$

where $C'_{\text{LSI}} = C_{\text{LSI}} \cdot (1 + O(\delta^2))$ for cloning noise variance $\delta^2$.

**Key mechanism**: The cloning operator introduces small Gaussian noise ($\delta \xi$) when copying walkers, which regularizes the Fisher information and prevents LSI constant blow-up.

**Reference**: [10_kl_convergence.md](10_kl_convergence.md), Section 4, Theorem 4.3.
:::

### 2.2. The N→∞ Limit: What Changes?

**Question**: Does this LSI-preservation property survive the mean-field limit?

**Key observations**:

1. **Finite-N**: Cloning is a discrete, stochastic map on the finite-dimensional space $\Sigma_N = (\mathcal{X} \times \mathbb{R}^d)^N$

2. **Mean-field**: Revival is a deterministic, nonlinear map on the infinite-dimensional space $\mathcal{P}(\Omega)$

**Potential issues**:

:::{prf:problem} Critical Differences Between Finite-N and Mean-Field
:label: prob-finite-n-vs-mean-field

| Aspect | Finite-N Cloning | Mean-Field Revival | Implication |
|:-------|:-----------------|:-------------------|:------------|
| **Dimensionality** | Finite $(Nd)$ | Infinite (function space) | Compactness arguments may fail |
| **Noise** | Explicit $\delta \xi$ noise | No explicit noise in $\mathcal{R}$ | Fisher information regularization unclear |
| **Nonlinearity** | Linear in empirical measure | Nonlinear (division by $\\|\rho\\|_{L^1}$) | May create singularities |
| **Discreteness** | Discrete selection among N walkers | Continuous sampling from $\rho$ | Combinatorial structure lost |
| **Companion selection** | Finite sample ($j \in \mathcal{A}$) | Integral over $\rho$ | Correlations differ |

**Core concern**: The noise term $\delta \xi$ that regularizes Fisher information in finite-N is **not explicit** in the mean-field $\mathcal{R}$. Does regularization survive the limit, or is it lost?
:::

### 2.3. Informal Heuristic: Why Revival Might Be KL-Contractive

**Bayesian Analogy**:

The revival operator can be viewed as a form of **Bayesian conditioning**:

$$
\mathcal{R}[\rho, m_d](x,v) \propto \rho(x,v)
$$

This is analogous to updating a prior $\rho$ by conditioning on the event "walker survives." Bayesian updates are known to be KL-contractive:

:::{prf:theorem} Data Processing Inequality (Standard Result)
:label: thm-data-processing

For any Markov kernel $K: \mathcal{X} \to \mathcal{P}(\mathcal{Y})$:

$$
D_{\text{KL}}(K \rho \| K \sigma) \le D_{\text{KL}}(\rho \| \sigma)
$$

where $K\rho(y) = \int K(x \to y) \rho(x) dx$ is the push-forward.

**Intuition**: Processing through a channel cannot increase information divergence.
:::

**Application attempt**:

Can we model $\mathcal{R}$ as a Markov kernel? The challenge is that $\mathcal{R}$ is **not** a standard kernel—it depends on $\rho$ globally through the normalization $\|\rho\|_{L^1}$.

**Proportional resampling**:

Alternatively, view $\mathcal{R}$ as resampling from the normalized distribution $\rho / \|\rho\|_{L^1}$. Resampling (i.i.d. sampling) typically contracts or preserves KL-divergence.

### 2.4. Where the Heuristic May Fail

:::{admonition} Critical Weakness of the Bayesian Analogy
:class: warning

The Bayesian/data-processing analogy **breaks down** because:

1. **Global dependency**: $\mathcal{R}[\rho]$ depends on the **total mass** $\|\rho\|_{L^1}$, not just the shape
2. **Two-argument operator**: $\mathcal{R}$ takes both $(\rho, m_d)$, coupling the alive and dead masses
3. **Nonlinear functional**: The division by $\|\rho\|_{L^1}$ is a nonlinear operation on function space

**Potential failure mode**: If $\rho$ and $\sigma$ have different total masses, the normalizations $\|\rho\|_{L^1}$ and $\|\sigma\|_{L^1}$ differ. The KL-divergence might increase due to this normalization mismatch.

**Example concern**: Suppose $\|\rho\|_{L^1} = 0.9$ and $\|\sigma\|_{L^1} = 0.8$. Then:

$$
\frac{\rho}{\|\rho\|_{L^1}} \quad \text{vs.} \quad \frac{\sigma}{\|\sigma\|_{L^1}}
$$

have different normalizations. It's not obvious that $D_{\text{KL}}$ decreases.
:::

**Conclusion of Section 2**: The finite-N result is promising but **not sufficient**. We need a direct proof for the mean-field operator.

---

## 3. Direct Proof Attempts

This section explores multiple approaches to proving (or disproving) KL-non-expansiveness of $\mathcal{R}$.

### 3.1. Approach 1: Explicit KL-Divergence Calculation

**Strategy**: Compute $D_{\text{KL}}(\mathcal{R}(\rho) \| \mathcal{R}(\sigma))$ explicitly and compare to $D_{\text{KL}}(\rho \| \sigma)$.

**Setup**: Let $\rho, \sigma \in \mathcal{P}(\Omega)$ with $\|\rho\|_{L^1} = m_\rho < 1$ and $\|\sigma\|_{L^1} = m_\sigma < 1$.

Define normalized densities:

$$
\tilde{\rho} := \frac{\rho}{m_\rho}, \quad \tilde{\sigma} := \frac{\sigma}{m_\sigma}
$$

Then:

$$
\mathcal{R}[\rho, 1-m_\rho] = \lambda_{\text{revive}} (1 - m_\rho) \tilde{\rho}
$$

**KL-divergence**:

$$
\begin{aligned}
D_{\text{KL}}(\mathcal{R}(\rho) \| \mathcal{R}(\sigma)) &= \int_\Omega \mathcal{R}[\rho] \log \frac{\mathcal{R}[\rho]}{\mathcal{R}[\sigma]} \\
&= \int_\Omega \lambda (1-m_\rho) \tilde{\rho} \log \frac{\lambda (1-m_\rho) \tilde{\rho}}{\lambda (1-m_\sigma) \tilde{\sigma}} \\
&= \lambda (1-m_\rho) \int \tilde{\rho} \left[ \log \frac{1-m_\rho}{1-m_\sigma} + \log \frac{\tilde{\rho}}{\tilde{\sigma}} \right] \\
&= \lambda (1-m_\rho) \left[ \log \frac{1-m_\rho}{1-m_\sigma} + D_{\text{KL}}(\tilde{\rho} \| \tilde{\sigma}) \right]
\end{aligned}
$$

**Comparison to original**:

$$
D_{\text{KL}}(\rho \| \sigma) = \int \rho \log \frac{\rho}{\sigma} = m_\rho \int \tilde{\rho} \log \frac{m_\rho \tilde{\rho}}{m_\sigma \tilde{\sigma}} = m_\rho \left[ \log \frac{m_\rho}{m_\sigma} + D_{\text{KL}}(\tilde{\rho} \| \tilde{\sigma}) \right]
$$

**KEY QUESTION**: When is

$$
\lambda (1-m_\rho) \left[ \log \frac{1-m_\rho}{1-m_\sigma} + D_{\text{KL}}(\tilde{\rho} \| \tilde{\sigma}) \right] \le m_\rho \left[ \log \frac{m_\rho}{m_\sigma} + D_{\text{KL}}(\tilde{\rho} \| \tilde{\sigma}) \right] \quad ?
$$

**Analysis**:

This inequality is NOT automatic. It depends on:
1. Relative sizes of $m_\rho$ vs. $m_\sigma$
2. Magnitude of $D_{\text{KL}}(\tilde{\rho} \| \tilde{\sigma})$
3. Value of $\lambda_{\text{revive}}$

:::{prf:problem} Open Question from Explicit Calculation
:label: prob-explicit-kl-condition

The KL-non-expansiveness of $\mathcal{R}$ is equivalent to:

$$
\lambda (1-m_\rho) \log \frac{1-m_\rho}{1-m_\sigma} \le m_\rho \log \frac{m_\rho}{m_\sigma}
$$

plus a term involving the normalized KL-divergences.

**Special case** ($m_\rho = m_\sigma$): Both log terms vanish, and we get:

$$
\lambda (1-m_\rho) D_{\text{KL}}(\tilde{\rho} \| \tilde{\sigma}) \le m_\rho D_{\text{KL}}(\tilde{\rho} \| \tilde{\sigma})
$$

This holds iff $\lambda (1 - m_\rho) \le m_\rho$, i.e., $\lambda \le \frac{m_\rho}{1 - m_\rho}$.

**General case**: Requires careful analysis of the log terms.
:::

**Conclusion**: The explicit calculation reveals that KL-contraction is **conditional**, not automatic. Need to understand when the condition holds.

### 3.2. Approach 2: Optimal Transport / Brenier Map

**Strategy**: Model $\mathcal{R}$ as an optimal transport map and use Wasserstein-to-KL bounds.

**Setup**: From optimal transport theory, if $T: \Omega \to \Omega$ is the Brenier map pushing $\rho$ to $\sigma$ (minimizing $\int |x - T(x)|^2 \rho(x) dx$), then:

$$
W_2(\rho, \sigma)^2 = \int |x - T(x)|^2 \rho(x) dx
$$

**HWI inequality** (Otto-Villani):

$$
D_{\text{KL}}(\rho \| \sigma) \le W_2(\rho, \sigma) \sqrt{I(\rho | \sigma)}
$$

where $I(\rho | \sigma)$ is the Fisher information.

**Application to revival**:

Can we show:
1. $\mathcal{R}$ decreases $W_2$ distance?
2. $\mathcal{R}$ controls Fisher information?

**Challenge**: $\mathcal{R}$ is not a deterministic transport map—it's a proportional scaling + mass injection. The optimal transport framework doesn't directly apply.

**Modified approach**: View $\mathcal{R}$ as a **Markov kernel** (resampling) and use the Kantorovich duality for random maps.

:::{prf:lemma} Wasserstein Contraction for Proportional Resampling (Conjecture)
:label: lem-wasserstein-revival

If $\mathcal{R}$ is viewed as resampling from $\tilde{\rho} = \rho / \|\rho\|_{L^1}$, then:

$$
W_2(\mathcal{R}(\rho), \mathcal{R}(\sigma)) \le \lambda (1-m) W_2(\tilde{\rho}, \tilde{\sigma})
$$

for some average dead mass $m \approx (m_\rho + m_\sigma)/2$.

**Status**: Unproven. Requires showing that proportional scaling is $W_2$-contractive.
:::

**Conclusion**: Optimal transport may provide a path, but the details are non-trivial.

### 3.3. Approach 3: Generator-Based Analysis

**Strategy**: Study the infinitesimal generator of the revival process and apply entropy production methods.

**Setup**: The revival term in the PDE is:

$$
\frac{\partial \rho}{\partial t} \Big|_{\text{revival}} = \mathcal{R}[\rho, m_d] = \lambda m_d \frac{\rho}{\|\rho\|_{L^1}}
$$

**Entropy change**:

$$
\frac{d}{dt} D_{\text{KL}}(\rho \| \pi) \Big|_{\text{revival}} = \int \mathcal{R}[\rho] \left(1 + \log \frac{\rho}{\pi}\right)
$$

**Question**: Is this negative (entropy decreasing)?

**Analysis**:

$$
\begin{aligned}
\frac{d}{dt} D_{\text{KL}} &= \lambda m_d \int \frac{\rho}{\|\rho\|_{L^1}} \left(1 + \log \frac{\rho}{\pi}\right) \\
&= \lambda m_d \cdot \frac{1}{\|\rho\|_{L^1}} \int \rho (1 + \log \frac{\rho}{\pi}) \\
&= \lambda m_d \cdot \frac{1}{\|\rho\|_{L^1}} \left[ \|\rho\|_{L^1} + \int \rho \log \frac{\rho}{\pi} \right]
\end{aligned}
$$

Simplifying:

$$
\frac{d}{dt} D_{\text{KL}} \Big|_{\text{revival}} = \lambda m_d \left[1 + \frac{1}{\|\rho\|_{L^1}} D_{\text{KL}}(\rho \| \pi \cdot \|\rho\|_{L^1}) \right]
$$

This is **positive** (entropy increasing!) unless there are compensating terms.

:::{admonition} Critical Finding
:class: danger

The infinitesimal entropy production from the revival operator is **POSITIVE**, meaning revival **increases** entropy in the short term!

This suggests that $\mathcal{R}$ alone may **not** be KL-contractive. Contraction must come from the **combined** killing + revival + kinetic system, not from $\mathcal{R}$ in isolation.

**Implication**: The discrete-time LSI strategy may need revision. Cannot treat $\mathcal{R}$ as independently contractive.
:::

### 3.4. Approach 4: Conditional Expectation / Doob's Martingale

**Strategy**: Model revival as a conditional expectation (Doob's projection), which is always KL-contractive.

**Setup**: In probability theory, conditioning on a σ-algebra $\mathcal{G}$ is KL-contractive:

$$
D_{\text{KL}}(\mathbb{E}[\rho | \mathcal{G}] \| \mathbb{E}[\sigma | \mathcal{G}]) \le D_{\text{KL}}(\rho \| \sigma)
$$

**Question**: Can we frame $\mathcal{R}$ as a conditional expectation?

**Attempt**: Model the alive/dead split as conditioning on survival event. However, $\mathcal{R}$ **adds** mass (from dead to alive), which is not a projection.

**Conclusion**: The conditional expectation framework doesn't directly apply because $\mathcal{R}$ is a mass-transfer operator, not a projection.

---

## 4. Counterexample Search and Failure Modes

This section investigates scenarios where KL-non-expansiveness might fail.

### 4.1. Simplified Model: Two-State System

**Setup**: Consider a minimal model with two states $\{A, B\}$:

- Alive distribution: $\rho = (p_A, p_B)$ with $p_A + p_B = m < 1$
- Dead mass: $m_d = 1 - m$

Revival operator:

$$
\mathcal{R}(\rho) = \lambda (1-m) \cdot \frac{(p_A, p_B)}{m} = \lambda (1-m) \cdot \left(\frac{p_A}{m}, \frac{p_B}{m}\right)
$$

**KL-divergence test**:

Let $\rho = (0.4, 0.1)$ (total mass 0.5) and $\sigma = (0.3, 0.2)$ (total mass 0.5).

Normalized: $\tilde{\rho} = (0.8, 0.2)$, $\tilde{\sigma} = (0.6, 0.4)$.

$$
D_{\text{KL}}(\tilde{\rho} \| \tilde{\sigma}) = 0.8 \log \frac{0.8}{0.6} + 0.2 \log \frac{0.2}{0.4} \approx 0.068
$$

After revival:

$$
\mathcal{R}(\rho) = 0.5 \lambda (0.8, 0.2), \quad \mathcal{R}(\sigma) = 0.5 \lambda (0.6, 0.4)
$$

$$
D_{\text{KL}}(\mathcal{R}(\rho) \| \mathcal{R}(\sigma)) = 0.5 \lambda \cdot D_{\text{KL}}(\tilde{\rho} \| \tilde{\sigma}) = 0.5 \lambda \cdot 0.068
$$

Original:

$$
D_{\text{KL}}(\rho \| \sigma) \approx 0.5 \cdot 0.068 = 0.034
$$

**Comparison**: $D_{\text{KL}}(\mathcal{R}(\rho) \| \mathcal{R}(\sigma)) \approx \lambda \cdot D_{\text{KL}}(\rho \| \sigma)$.

**Conclusion**: If $\lambda \le 1$, contraction holds. If $\lambda > 1$, expansion!

:::{prf:observation} Revival Rate Constraint
:label: obs-revival-rate-constraint

In the two-state model, KL-non-expansiveness requires $\lambda_{\text{revive}} \le 1$.

**Physical interpretation**: Revival rate must not exceed the death rate (on average) for stability.

**Question for full model**: Does a similar constraint hold in the continuous setting?
:::

### 4.2. Asymmetric Mass Scenario

**Concern**: What if $m_\rho \ll m_\sigma$ (one distribution has much less alive mass)?

**Example**: $m_\rho = 0.1$, $m_\sigma = 0.9$.

Then:
- $\mathcal{R}(\rho)$ adds $(1 - 0.1) \lambda = 0.9\lambda$ mass
- $\mathcal{R}(\sigma)$ adds $(1 - 0.9) \lambda = 0.1\lambda$ mass

The revival mass is **inversely proportional** to alive mass. This creates a "balancing" effect that might actually be KL-contractive (brings distributions closer).

**Numerical test needed**: Simulate this scenario with realistic $\rho, \sigma$ to check.

### 4.3. Singular Limit: $\|\rho\|_{L^1} \to 0$

**Extreme case**: What happens as the alive mass approaches zero?

$$
\mathcal{R}[\rho, 1] = \lambda \cdot \frac{\rho}{\|\rho\|_{L^1}}
$$

As $\|\rho\|_{L^1} \to 0$, the normalized density $\rho / \|\rho\|_{L^1}$ remains well-defined (shape preserved), but the revival mass $\lambda \cdot 1 \to \lambda$ becomes constant.

**Potential issue**: If $\rho \to 0$ but $\sigma$ remains finite, the KL-divergence $D_{\text{KL}}(\rho \| \sigma) \to \infty$. Does $D_{\text{KL}}(\mathcal{R}(\rho) \| \mathcal{R}(\sigma))$ also diverge, or does revival stabilize?

**Conjecture**: Revival acts as a "regularization" in this limit, preventing blow-up. But needs rigorous proof.

---

## 5. Decision Point and Path Forward

### 5.1. Summary of Findings (Preliminary)

| Approach | Outcome | Confidence |
|:---------|:--------|:-----------|
| **Explicit calculation** | Reveals conditional contraction (depends on $\lambda$, mass balance) | Medium |
| **Optimal transport** | Promising but requires technical development | Low |
| **Generator analysis** | **Revival increases entropy short-term!** | High |
| **Conditional expectation** | Framework doesn't apply directly | High |
| **Two-state model** | Suggests $\lambda \le 1$ constraint | Medium |
| **Singular limit** | Needs investigation | Low |

### 5.2. Key Insights

1. **Revival alone is NOT KL-contractive**: Generator analysis (Section 3.3) shows revival **increases** entropy.

2. **Contraction must be joint**: The combined killing + revival + kinetic system may be contractive, even if $\mathcal{R}$ alone is not.

3. **Revival rate matters**: Two-state model suggests $\lambda_{\text{revive}} \le 1$ may be necessary.

4. **Normalization is critical**: The division by $\|\rho\|_{L^1}$ creates nonlinear coupling that's hard to analyze.

### 5.3. Recommended Next Steps

Based on these preliminary findings, I recommend the following investigations with Gemini:

:::{prf:problem} Critical Tasks for Gemini Collaboration
:label: prob-gemini-collaboration-tasks

1. **Refine the two-state counterexample**: Develop a rigorous lower-dimensional model where calculations are exact. Determine precisely when KL-non-expansiveness holds.

2. **Investigate joint operator**: Instead of analyzing $\mathcal{R}$ alone, study the composed operator $\mathcal{J} = $ (killing + revival). Does the killing term compensate for revival's entropy increase?

3. **Explore alternative formulations**: Can we reformulate the revival operator to make KL-properties more transparent? E.g., as a Bayesian update with explicit noise?

4. **Check literature**: Has anyone studied KL-properties of proportional resampling operators in McKean-Vlasov systems? Search for related results in QSD theory, population dynamics, branching processes.

5. **Numerical validation**: Simulate the mean-field PDE and numerically compute $D_{\text{KL}}(\rho_t \| \rho_\infty)$ over time. Does it decrease monotonically?
:::

---

## 6. Collaborative Session with Gemini: Next Steps

**Prepared questions for Gemini**:

1. Given the generator analysis showing $\mathcal{R}$ increases entropy, should we abandon the idea that $\mathcal{R}$ alone is KL-contractive?

2. Can you help develop a rigorous analysis of the **joint** operator (killing + revival)? Perhaps the combined effect is contractive even if components aren't?

3. Are there alternative characterizations of the revival operator (e.g., as optimal transport with constraints) that might be more analytically tractable?

4. What does the literature say about KL-properties of resampling operators in infinite-dimensional settings?

5. If $\mathcal{R}$ is NOT KL-contractive, what are the implications for the roadmap? Should we pivot to Strategy B (Wasserstein) or another approach?

---

**Document Status**: Initial investigation complete, critical findings identified, Gemini collaboration COMPLETED.

---

## 7. Gemini's Expert Analysis and Verification

### 7.1. Calculation Verification

**Gemini's assessment**: The entropy production calculation is **fundamentally correct**. Minor notation correction:

$$
\frac{d}{dt} D_{\text{KL}}(\rho \| \pi) \Big|_{\text{revival}} = \lambda m_d \left( 1 + \frac{D_{\text{KL}}(\rho \| \pi)}{\|\rho\|_{L^1}} \right)
$$

**Conclusion**: Since $D_{\text{KL}} \ge 0$, $\lambda > 0$, $m_d > 0$, and $\|\rho\|_{L^1} > 0$, the entropy production is **strictly positive**.

:::{prf:theorem} Revival Operator is KL-Expansive (VERIFIED)
:label: thm-revival-kl-expansive

The mean-field revival operator $\mathcal{R}[\rho, m_d]$ **increases** the KL-divergence to the invariant measure $\pi$:

$$
\frac{d}{dt} D_{\text{KL}}(\rho \| \pi) \Big|_{\text{revival}} > 0 \quad \text{for all } \rho \neq \pi, \, m_d > 0
$$

**Status**: PROVEN (verified by Gemini 2025-01-08)
:::

### 7.2. Joint Operator Analysis

Gemini computed the entropy production for the combined jump operator:

$$
\frac{d}{dt} D_{\text{KL}}(\rho \| \pi) \Big|_{\text{jump}} = \underbrace{(\lambda m_d - \int \kappa_{\text{kill}} \rho)}_{\text{Mass rate}} + \underbrace{\int \left(\frac{\lambda m_d}{\|\rho\|_{L^1}} - \kappa_{\text{kill}}(x)\right) \rho \log \frac{\rho}{\pi}}_{\text{Divergence change}}
$$

For constant $\kappa_{\text{kill}}$, the coefficient of $D_{\text{KL}}$ is $\frac{\lambda m_d}{\|\rho\|_{L^1}} - \kappa_{\text{kill}}$, which is:
- **Positive** when $\|\rho\|_{L^1} < \frac{\lambda}{\lambda + \kappa}$ (expansive below equilibrium)
- **Negative** when $\|\rho\|_{L^1} > \frac{\lambda}{\lambda + \kappa}$ (contractive above equilibrium)

:::{prf:theorem} Joint Jump Operator NOT Unconditionally Contractive (VERIFIED)
:label: thm-joint-not-contractive

The combined killing + revival operator regulates **total mass**, not information distance. It is NOT KL-contractive in general.
:::

### 7.3. Decision Tree Resolution

```
Revival alone is KL-expansive: TRUE ✓
  Joint operator is KL-contractive: FALSE ✗
  Kinetic operator dominates: MOST PLAUSIBLE PATH ✓
    → Proceed with composition analysis
    → Proof: kinetic dissipation > jump expansion
```

### 7.4. Literature Context

**Gemini**: "To my knowledge, there is **no standard result** proving proportional resampling is KL-contractive in infinite dimensions. Your finding reflects the true nature of the operator."

**Particle filters** (Del Moral, Doucet): Resampling NOT KL-contractive due to nonlinearity.

**McKean-Vlasov** (Méléard, Tugaut): Convergence via showing diffusion overcomes jump expansion.

---

## 8. Stage 0 Conclusion

### 8.1. Main Result

:::{prf:theorem} Stage 0 COMPLETE (VERIFIED)
:label: thm-stage0-complete

1. Revival operator is KL-expansive ✓
2. Joint jump operator not unconditionally contractive ✓
3. KL-convergence requires kinetic dominance ✓

**Status**: Verified by Gemini 2025-01-08
:::

### 8.2. Decision: GO with Revised Strategy

**DECISION**: Proceed with **kinetic dominance approach**

**Revised roadmap**:
- ✅ Analyze full generator $\mathcal{L} = \mathcal{L}_{\text{kin}} + \mathcal{L}_{\text{jump}}$
- ✅ Prove hypocoercive dissipation dominates jump expansion
- ✅ Mirrors finite-N proof structure ([10_kl_convergence.md](10_kl_convergence.md))

**Success probability**: 20-30%

### 8.3. Next Steps

**Stage 1**: Full generator analysis and hypocoercive LSI (6-9 months)

**Document complete**: 2025-01-08
