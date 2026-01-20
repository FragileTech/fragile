# Wasserstein-2 Control via Keystone-Based Variance Proxy

## 0. TLDR

**Centered positional control under cloning**: We work with phase-space $W_2$ on $z=(x,v)$ and the barycenter decomposition
$W_2^2(\mu_1, \mu_2) = \|\bar{z}_1 - \bar{z}_2\|^2 + W_2^2(\tilde{\mu}_1, \tilde{\mu}_2)$. For the centered positional marginals, let
$V_{\text{x,struct}} := W_{2,x}^2(\tilde{\mu}_{x,1}, \tilde{\mu}_{x,2})$ and define the variance proxy
$V_{\text{x,proxy}} := \text{Var}_x(S_1) + \text{Var}_x(S_2)$.
Lemma {prf:ref}`lem-centered-w2-variance-bound` shows $V_{\text{x,struct}} \le V_{\text{x,proxy}}$.
Using the Quantitative Keystone Lemma ({doc}`03_cloning`), cloning yields the N-uniform drift bound
$\mathbb{E}[\Delta V_{\text{x,proxy}}] \le -\kappa_x V_{\text{x,proxy}} + C_x$.
Thus cloning gives N-uniform **control** of the centered positional $W_2$ component via a contractive proxy.

**No alignment axiom**: The analysis avoids cross-swarm alignment assumptions. It relies on the Keystone causal chain (high error → fit/unfit signal → cloning pressure) and the positional variance drift proved in {doc}`03_cloning`.

**Full $W_2$ contraction needs kinetic**: Cloning does not contract the barycenter or velocity components. The kinetic operator $\Psi_{\text{kin}}$ provides this missing contraction, so the combined dynamics yields full phase-space $W_2$ contraction.

**Explicit constants**: $\kappa_x = \frac{\chi(\varepsilon)}{4} c_{\text{struct}}$ with
$\chi(\varepsilon)=p_u(\varepsilon)c_{\text{err}}(\varepsilon)$ and
$g_{\max}(\varepsilon)=\max(p_u(\varepsilon) g_{\text{err}}(\varepsilon), \chi(\varepsilon) R_{\text{spread}}^2)$ (Section 8). All constants are N-uniform.

**Dependencies**: {doc}`03_cloning`, {doc}`02_euclidean_gas`

## 1. Introduction

### 1.1. Goal and Scope

The goal of this document is to prove that the **cloning operator** $\Psi_{\text{clone}}$ of the Fragile Gas framework induces an **N-uniform drift bound on a variance proxy** that controls the centered/structural component of the Wasserstein-2 distance. A closed drift inequality for the centered positional term is obtained under an explicit structural-dominance assumption. These results bridge the finite-particle dynamics to the mean-field limit and support propagation of chaos.

The central mathematical object is the Wasserstein-2 distance $W_2(\mu_1, \mu_2)$ between two empirical swarm distributions $\mu_1, \mu_2$ on **phase space** $z := (x, v)$, supported on $N$ walkers. We decompose it into barycenter and centered components and control the centered **positional** part under cloning. Let $\bar{z}_k := \int z \, d\mu_k = (\bar{x}_k, \bar{v}_k)$ and $\tilde{\mu}_k := (z - \bar{z}_k)_\# \mu_k$, so that:

$$
W_2^2(\mu_1, \mu_2) = \|\bar{z}_1 - \bar{z}_2\|^2 + W_2^2(\tilde{\mu}_1, \tilde{\mu}_2)

$$

Define the centered positional Wasserstein term:

$$
V_{\text{x,struct}} := W_{2,x}^2(\tilde{\mu}_{x,1}, \tilde{\mu}_{x,2})

$$

where $\tilde{\mu}_{x,k}$ is the positional marginal of $\tilde{\mu}_k$. We also define the **variance proxy**:

$$
V_{\text{x,proxy}} := \text{Var}_x(S_1) + \text{Var}_x(S_2)

$$

In the all-alive regime, $\text{Var}_x(S_k) = \frac{1}{N}\sum_{i=1}^N \|\delta_{x,k,i}\|^2$. Lemma {prf:ref}`lem-centered-w2-variance-bound` shows
$V_{\text{x,struct}} \le V_{\text{x,proxy}}$.
We prove that applying the cloning operator to both swarms yields a **variance-proxy drift bound**:

$$
\mathbb{E}[\Delta V_{\text{x,proxy}}] \leq -\kappa_x V_{\text{x,proxy}} + C_x

$$

where $\kappa_x > 0$ is N-uniform and $C_x$ is a state-independent noise constant. This gives N-uniform control of the centered positional $W_2$ component via a contractive proxy. By coercivity in {doc}`03_cloning`, the hypocoercive structural error satisfies
$V_{\text{struct}} \geq \lambda_2 W_2^2(\tilde{\mu}_1, \tilde{\mu}_2) \geq \lambda_2 V_{\text{x,struct}}$. The barycenter term $\|\bar{z}_1 - \bar{z}_2\|^2$ is handled by the kinetic operator.

The critical challenge is establishing **N-uniformity** of the drift coefficient. Previous attempts using single-walker coupling failed because they required a minimum matching probability $q_{\min} > 0$ independent of $N$, which is impossible for $N!$ permutations. This document resolves this obstruction by importing the Keystone Lemma's N-uniform constants and working with variance-level quantities that are invariant under relabeling.

The scope of this document is strictly focused on the cloning operator's centered/structural Wasserstein control via a variance proxy (and the conditional closed drift under structural dominance). The complementary analysis of the kinetic operator $\Psi_{\text{kin}}$, which provides contraction in the velocity and barycenter/location components, and the full convergence analysis combining both operators are addressed in companion documents. We use the framework axioms and proven results from {doc}`03_cloning` (particularly Chapters 6-8 on the Keystone Principle) as foundational building blocks.

### 1.2. Why Wasserstein-2 Contraction Matters

Centered Wasserstein-2 control via the variance proxy under the cloning operator is not merely a technical result—it is the **rigorous justification** for treating the Fragile Gas as a continuum physics model and for deriving its mean-field limit.

**Connection to Mean-Field Theory**: The propagation of chaos framework (documented in {doc}`09_propagation_chaos`) establishes that an N-particle system converges to a mean-field limit if its dynamics contract in Wasserstein distance with **N-uniform constants**. Without this property, the limiting behavior could degenerate as $N \to \infty$, invalidating the mean-field PDE. Our result shows that the cloning operator supplies an N-uniform drift on a **variance proxy** that controls the centered positional $W_2$ component; full phase-space $W_2$ contraction follows once the kinetic operator controls the barycenter and velocity components.

**Role in Convergence Theory**: The Fragile Gas alternates between two operators: the cloning operator $\Psi_{\text{clone}}$ (which we analyze here) and the kinetic operator $\Psi_{\text{kin}}$ (analyzed in {doc}`02_euclidean_gas` and {doc}`05_kinetic_contraction`). Together, they form a **hypocoercive** dynamics where each operator contracts different error components:
- **Cloning operator**: Contracts the positional variance proxy $V_{\text{x,proxy}}$ that bounds $V_{\text{x,struct}}$
- **Kinetic operator**: Contracts barycenter and velocity components

The Foster-Lyapunov drift analysis (Chapter 12 of {doc}`03_cloning`) combines these partial contractions to prove exponential convergence to a unique quasi-stationary distribution (QSD). Our centered Wasserstein-2 **proxy control** provides the geometric foundation for this convergence.

**Complementary to KL-Convergence**: An alternative convergence analysis using Kullback-Leibler (KL) divergence and log-Sobolev inequalities (LSI) is developed in {doc}`15_kl_convergence`. The KL approach may yield faster convergence rates via entropy methods, while the Wasserstein-2 approach provides geometric intuition and explicit N-uniform constants. Both frameworks are valid and mutually reinforcing—the existence of multiple independent proofs strengthens confidence in the Fragile Gas's stability.

:::{important}
**Why N-Uniformity is Non-Negotiable**

A drift coefficient $\kappa_x(N)$ that vanishes as $N \to \infty$ (e.g., $\kappa_x(N) \sim 1/N$) would imply that large swarms contract the positional variance arbitrarily slowly. This would invalidate:
1. The mean-field limit (no well-defined continuum behavior)
2. The propagation of chaos (N-particle correlations could persist)
3. The interpretation of the Fragile Gas as a physical system with thermodynamic properties

Our Keystone-based proof establishes that $\kappa_x$ is built from the N-uniform constants $\chi(\varepsilon)$ and $c_{\text{struct}}$ ({doc}`03_cloning`). This validates the Fragile Gas as a scalable, physically meaningful model.
:::

### 1.3. Overview of the Proof Strategy and Document Structure

The proof constructs the centered-control argument through five main stages, illustrated in the diagram below:

```{mermaid}
graph TD
    subgraph "Legend"
        L1["Definition/Concept"]:::stateStyle
        L2["Lemma"]:::lemmaStyle
        L3["Theorem"]:::theoremStyle
    end

    subgraph "Foundations (§2)"
        A["<b>§2: Cluster Structure</b><br>Target set I_k = U_k ∩ H_k<br>Complement J_k"]:::stateStyle
    end

    subgraph "Variance Analysis (§3)"
        B["<b>§3.1: Variance Decomposition</b><br>Var(S_k) = f_I Var(I_k) + f_J Var(J_k)<br>+ f_I f_J ||μ(I_k) - μ(J_k)||²"]:::lemmaStyle
        C["<b>§3.2: Centered W₂ Bound</b><br>W_{2,x}²(tilde μ_{x,1}, tilde μ_{x,2})<br>≤ Var_x(S_1) + Var_x(S_2)"]:::lemmaStyle
    end

    subgraph "Keystone Core (§4)"
        D["<b>§4.1: Quantitative Keystone Lemma</b><br>χ(ε), g_max(ε) from {doc}03_cloning"]:::lemmaStyle
        E["<b>§4.2: Positional Variance Drift</b><br>𝔼[Δ V_{x,proxy}] ≤ -κ_x V_{x,proxy} + C_x"]:::theoremStyle
    end

    subgraph "Main Result (§5-6)"
        F["<b>§5: Centered W₂ Control</b><br>V_{x,struct} ≤ V_{x,proxy}<br>Proxy drift yields control"]:::stateStyle
        G["<b>§6: Structural/Barycenter Split</b><br>Full W₂ via kinetic + cloning"]:::theoremStyle
    end

    subgraph "Analysis (§7-8)"
        H["<b>§7: Comparison</b><br>No q_min; no alignment axiom"]:::stateStyle
        I["<b>§8: Explicit Constants</b><br>χ, g_max, κ_x derived"]:::stateStyle
    end

    A --> B
    B --> C
    D --> E
    C --> E
    E --> F
    F --> G
    G --> H
    G --> I

    classDef stateStyle fill:#4a5f8c,stroke:#8fa4d4,stroke-width:2px,color:#e8eaf6
    classDef axiomStyle fill:#8c6239,stroke:#d4a574,stroke-width:2px,stroke-dasharray: 5 5,color:#f4e8d8
    classDef lemmaStyle fill:#3d6b4b,stroke:#7fc296,stroke-width:2px,color:#d8f4e3
    classDef theoremStyle fill:#8c3d5f,stroke:#d47fa4,stroke-width:3px,color:#f4d8e8
```

**Proof Architecture**:

**Section 2 (Cluster Structure)**: We recall the **target set** $I_k = U_k \cap H_k(\varepsilon)$ (unfit and high-error walkers) and its **complement** $J_k$, using the exact same clustering algorithm (Definition 6.3.1) and unfit set definition (Definition 7.6.1.0) from {doc}`03_cloning`.

**Section 3 (Variance + Centered $W_2$ Bound)**: We prove the variance decomposition ({prf:ref}`lem-variance-decomposition`) and show that the centered positional Wasserstein term is bounded by the sum of internal variances ({prf:ref}`lem-centered-w2-variance-bound`).

**Section 4 (Keystone Core)**: We import the Quantitative Keystone Lemma and the positional variance drift theorem from {doc}`03_cloning`, yielding a direct drift bound for the variance proxy $V_{\text{x,proxy}}$.

**Section 5 (Centered $W_2$ Control)**: We combine the proxy drift with the bound $V_{\text{x,struct}} \le V_{\text{x,proxy}}$ to obtain N-uniform control of the centered positional component.

**Section 6 (Full $W_2$ Split)**: We combine the barycenter decomposition with the kinetic contraction results to explain how the full phase-space $W_2$ contracts when cloning and kinetic steps are composed.

**Section 7 (Comparison)**: We contrast the Keystone-based variance approach with the failed single-walker $q_{\min}$ strategy.

**Section 8 (Explicit Constants)**: We derive parameter-level expressions for $\chi(\varepsilon)$, $g_{\max}(\varepsilon)$, and $\kappa_x$.

**Key Proof Principles**:

1. **Variance proxy**: Control $V_{\text{x,struct}}$ via $V_{\text{x,proxy}} = \text{Var}_x(S_1) + \text{Var}_x(S_2)$
2. **Keystone constants**: Use $f_{UH}(\varepsilon)$, $p_u(\varepsilon)$, and $\chi(\varepsilon)$ from {doc}`03_cloning` (already N-uniform)
3. **No cross-swarm alignment**: Avoid brittle alignment assumptions and $q_{\min}$ arguments
4. **Framework consistency**: Use exact definitions from the Keystone Lemma proof

The result is a rigorous, self-contained proof of centered Wasserstein-2 **control via a variance proxy** (with a closed drift bound under structural dominance), with explicit N-uniform constants.

---

## 2. Cluster Structure

### 2.1. Cluster Structure Definitions

We first recall the cluster-based partition from {doc}`03_cloning`.

:::{prf:definition} Target Set and Complement
:label: def-target-complement

For a swarm $S_k$ with alive set $\mathcal{A}_k$, define:

**Target Set** (from {doc}`03_cloning`, Section 8.2):

$$
I_k(\varepsilon) := U_k \cap H_k(\varepsilon)

$$
where:
- $U_k$ is the unfit set (Definition 7.6.1.0, line 4499): walkers with fitness $\leq$ mean
- $H_k(\varepsilon)$ is the unified high-error set (Definition 6.3, line 2351): outlier clusters in phase space

**Complement Set**:

$$
J_k(\varepsilon) := \mathcal{A}_k \setminus I_k(\varepsilon)

$$

**Population fractions** (all-alive regime, so $|\mathcal{A}_k| = N$):

$$
f_I(\varepsilon) := \frac{|I_k|}{N}, \quad f_J(\varepsilon) := \frac{|J_k|}{N} = 1 - f_I(\varepsilon)

$$

**Guaranteed lower bound** (Theorem 7.6.1, line 4572):

$$
f_I(\varepsilon) \geq f_{UH}(\varepsilon) > 0 \quad \text{(N-uniform)}

$$
:::

:::{prf:remark} All-Alive Normalization
:label: rem-all-alive-normalization

The cloning operator outputs all-alive swarms, so throughout this document we work in the all-alive regime $|\mathcal{A}_k| = N$. This keeps the empirical measure normalization consistent with the $W_2$ formulation and aligns $f_{UH}(\varepsilon)$ with the lower bound proven in {doc}`03_cloning` (where $k = N$ in the all-alive state).
:::

:::{prf:remark} Why These Sets?
:label: rem-why-target-sets

The target set $I_k$ represents the walkers that are:
1. **Unfit** ($U_k$): Lower than average fitness → high cloning probability
2. **High-error** ($H_k$): Geometrically outliers → contribute to structural error

By Theorem 7.6.1 ({doc}`03_cloning`, Section 7.6.2), the Stability Condition guarantees a **non-vanishing overlap** between these sets. This is the crucial population that:
- Is **targeted** by the cloning mechanism (unfit)
- **Causes** the structural error (high-error)

The Keystone proof exploits this **correctly-targeted** population.
:::

:::{prf:remark} Empirical Measures and Framework Properties
:label: rem-empirical-measures

**Notational Precision**: This document analyzes the $N$-particle empirical measures $\mu_1, \mu_2$, which are discrete probability measures supported on $N$ walkers. The clustering algorithm, fitness function $F(x)$, and potential landscape are properties defined at the population level.

**Variance Notation**: $V_{\text{struct}}$ denotes the hypocoercive structural error between centered **phase-space** measures (as in {doc}`03_cloning`). We also use the positional structural term
$V_{\text{x,struct}} := W_{2,x}^2(\tilde{\mu}_{x,1}, \tilde{\mu}_{x,2})$ for centered positional marginals and the variance proxy
$V_{\text{x,proxy}} := \text{Var}_x(S_1) + \text{Var}_x(S_2)$. $\text{Var}_x(S_k)$ denotes the internal positional variance of swarm $k$.

**Relationship to Continuum Limit**: The fitness function $F(x)$ and its valley structure are properties of the continuum state space $\mathcal{X}$, while the clusters $I_k, J_k$ are finite-sample objects constructed from the empirical distribution. The proofs in this document use properties of the limiting landscape (e.g., Confining Potential axiom, fitness valleys) to reason about finite-sample cluster behavior.

**Approximation Errors**: For finite $N$, there are approximation errors $O(1/\sqrt{N})$ when estimating continuum properties (like the potential $F(x)$) from empirical measures. These errors are absorbed into:
1. The noise term $C_x = \frac{g_{\max}(\varepsilon)}{4} + 4d\delta^2$ in the variance-proxy drift inequality
2. The clustering threshold $\varepsilon$, which depends on $N$ implicitly through the error tolerance

**N-Uniformity Justification**: The key result is that these finite-sample approximation errors do not affect the *sign* or *N-independence* of the drift coefficient $\kappa_x > 0$. This is because:
- The clustering algorithm thresholds (Definition 6.3) are calibrated to maintain $O(1)$ cluster fractions
- The framework axioms (Confining Potential, Environmental Richness) provide $O(1)$ landscape features that dominate the finite-sample noise
- All critical bounds ($f_{UH}, p_u, \chi, g_{\max}$) are proven N-uniform in {doc}`03_cloning`

This remark clarifies that while the analysis is formally at the $N$-particle level, the use of continuum landscape properties is justified by the framework's built-in error control mechanisms.
:::

### 2.2. No Cross-Swarm Alignment Assumption

This document does not assume any cross-swarm alignment or matching axiom. All geometric guarantees are imported from the Keystone Lemma chain in {doc}`03_cloning`. The only cross-swarm coupling used later is the standard independent coupling for bounding $W_{2,x}^2$ by internal variances (Lemma {prf:ref}`lem-centered-w2-variance-bound`), which requires no alignment structure.

---

## 3. Variance Decomposition and Centered Wasserstein Bound

### 3.1. Within-Swarm Variance Decomposition

We first establish how variance decomposes with respect to the cluster partition.

:::{prf:lemma} Variance Decomposition by Clusters
:label: lem-variance-decomposition

For a swarm $S_k$ partitioned into $I_k$ (target) and $J_k$ (complement) with population fractions $f_I = |I_k|/N$ and $f_J = |J_k|/N$:

$$
\text{Var}_x(S_k) = f_I \text{Var}_x(I_k) + f_J \text{Var}_x(J_k) + f_I f_J \|\mu_x(I_k) - \mu_x(J_k)\|^2

$$

where:
- $\text{Var}_x(I_k) = \frac{1}{|I_k|} \sum_{i \in I_k} \|x_i - \mu_x(I_k)\|^2$ (within-target variance)
- $\text{Var}_x(J_k) = \frac{1}{|J_k|} \sum_{j \in J_k} \|x_j - \mu_x(J_k)\|^2$ (within-complement variance)
- $\mu_x(I_k) = \frac{1}{|I_k|} \sum_{i \in I_k} x_i$ (target barycenter)
- $\mu_x(J_k) = \frac{1}{|J_k|} \sum_{j \in J_k} x_j$ (complement barycenter)

**Proof:**

Standard variance decomposition. The total variance is:

$$
\text{Var}_x(S_k) = \frac{1}{N} \sum_{i=1}^N \|x_i - \bar{x}_k\|^2

$$

where $\bar{x}_k = \frac{1}{N}\sum_{i=1}^N x_i = f_I \mu_x(I_k) + f_J \mu_x(J_k)$.

Expand:

$$
\begin{aligned}
N \cdot \text{Var}_x(S_k) &= \sum_{i \in I_k} \|x_i - \bar{x}_k\|^2 + \sum_{j \in J_k} \|x_j - \bar{x}_k\|^2 \\
&= \sum_{i \in I_k} \|x_i - \mu_x(I_k) + \mu_x(I_k) - \bar{x}_k\|^2 + \sum_{j \in J_k} \|x_j - \mu_x(J_k) + \mu_x(J_k) - \bar{x}_k\|^2
\end{aligned}

$$

Using $\|a + b\|^2 = \|a\|^2 + 2\langle a, b\rangle + \|b\|^2$ and $\sum_{i \in I_k} (x_i - \mu_x(I_k)) = 0$:

$$
\begin{aligned}
&= \sum_{i \in I_k} \|x_i - \mu_x(I_k)\|^2 + |I_k| \|\mu_x(I_k) - \bar{x}_k\|^2 \\
&\quad + \sum_{j \in J_k} \|x_j - \mu_x(J_k)\|^2 + |J_k| \|\mu_x(J_k) - \bar{x}_k\|^2
\end{aligned}

$$

Now, $\mu_x(I_k) - \bar{x}_k = \mu_x(I_k) - f_I \mu_x(I_k) - f_J \mu_x(J_k) = f_J (\mu_x(I_k) - \mu_x(J_k))$.

Similarly, $\mu_x(J_k) - \bar{x}_k = -f_I (\mu_x(I_k) - \mu_x(J_k))$.

Therefore:

$$
\begin{aligned}
N \cdot \text{Var}_x(S_k) &= |I_k| \text{Var}_x(I_k) + |I_k| f_J^2 \|\mu_x(I_k) - \mu_x(J_k)\|^2 \\
&\quad + |J_k| \text{Var}_x(J_k) + |J_k| f_I^2 \|\mu_x(I_k) - \mu_x(J_k)\|^2 \\
&= |I_k| \text{Var}_x(I_k) + |J_k| \text{Var}_x(J_k) + (|I_k| f_J^2 + |J_k| f_I^2) \|\mu_x(I_k) - \mu_x(J_k)\|^2
\end{aligned}

$$

Using $|I_k| = f_I N$ and $|J_k| = f_J N$:

$$
|I_k| f_J^2 + |J_k| f_I^2 = N f_I f_J^2 + N f_J f_I^2 = N f_I f_J (f_J + f_I) = N f_I f_J

$$

Dividing by $N$ gives the result. □
:::



### 3.2. Centered Positional Wasserstein Bound

We bound the centered positional Wasserstein term by the internal variances of the two swarms. This avoids any cross-swarm alignment assumptions.

:::{prf:lemma} Centered Positional Wasserstein Bound
:label: lem-centered-w2-variance-bound

Let $\tilde{\mu}_{x,1}$ and $\tilde{\mu}_{x,2}$ be the centered positional empirical measures of two all-alive swarms. Then:

$$
V_{\text{x,struct}} = W_{2,x}^2(\tilde{\mu}_{x,1}, \tilde{\mu}_{x,2}) \leq \text{Var}_x(S_1) + \text{Var}_x(S_2) = V_{\text{x,proxy}}.

$$

**Proof.**

Let $X \sim \tilde{\mu}_{x,1}$ and $Y \sim \tilde{\mu}_{x,2}$ be independent. Because both measures are centered, $\mathbb{E}[X] = \mathbb{E}[Y] = 0$, so:

$$
\mathbb{E}\|X - Y\|^2 = \mathbb{E}\|X\|^2 + \mathbb{E}\|Y\|^2 = \text{Var}_x(S_1) + \text{Var}_x(S_2).

$$

The independent coupling is an admissible transport plan, so the optimal transport cost is no larger than this value. □
:::

:::{prf:lemma} Barycenter Decomposition of Wasserstein-2
:label: lem-wasserstein-barycenter-decomposition

For two empirical measures $\mu_1, \mu_2$ on phase space $z = (x, v)$ with finite second moments, let $\bar{z}_k := \int z \, d\mu_k$ and define centered measures $\tilde{\mu}_k := (z - \bar{z}_k)_\# \mu_k$. Then:

$$
W_2^2(\mu_1, \mu_2) = \|\bar{z}_1 - \bar{z}_2\|^2 + W_2^2(\tilde{\mu}_1, \tilde{\mu}_2)

$$

**Proof.**

For any coupling $\pi \in \Gamma(\mu_1, \mu_2)$,

$$
\int \|z_1 - z_2\|^2 \, d\pi = \|\bar{z}_1 - \bar{z}_2\|^2 + \int \|(z_1 - \bar{z}_1) - (z_2 - \bar{z}_2)\|^2 \, d\pi

$$

because the cross term vanishes by centering. The map $(z_1, z_2) \mapsto (z_1 - \bar{z}_1, z_2 - \bar{z}_2)$ is a bijection between couplings of $\mu_1, \mu_2$ and couplings of $\tilde{\mu}_1, \tilde{\mu}_2$, so taking the infimum yields the claim. □
:::

:::{prf:remark} Interpretation of the Decomposition
:label: rem-variance-wasserstein-interpretation

The phase-space Wasserstein-2 distance splits into:

- **Barycenter term**: $\|\bar{z}_1 - \bar{z}_2\|^2$ (location + velocity mismatch)
- **Centered term**: $W_2^2(\tilde{\mu}_1, \tilde{\mu}_2)$ (shape/structure mismatch)

Cloning controls the **centered positional** component $V_{\text{x,struct}}$ through the variance proxy $V_{\text{x,proxy}}$ (Lemma {prf:ref}`lem-centered-w2-variance-bound`). In {doc}`03_cloning`, the structural error satisfies
$V_{\text{struct}} \geq \lambda_2 W_2^2(\tilde{\mu}_1, \tilde{\mu}_2) \geq \lambda_2 V_{\text{x,struct}}$
for an N-uniform $\lambda_2 > 0$, so proxy control yields N-uniform control of a centered component of phase-space $W_2$. The barycenter and velocity components are handled by the kinetic operator.
:::

:::{prf:remark} Structural-Dominance Regime (Optional)
:label: rem-structural-dominance

If the barycenter term is already controlled, for example if there exists an N-uniform $c_{\text{dom}} > 0$ such that

$$
\|\bar{z}_1 - \bar{z}_2\|^2 \leq c_{\text{dom}} V_{\text{x,struct}},

$$

then the proxy control in Section 5 combines with the decomposition above to yield geometric $W_2$ contraction. In practice this regime is obtained after composing with $\Psi_{\text{kin}}$ (see {doc}`05_kinetic_contraction` and {doc}`06_convergence`).
:::

---

## 4. Keystone-Driven Positional Variance Contraction

We now import the Keystone Lemma and the positional variance drift bound from {doc}`03_cloning`. These results provide the N-uniform contraction mechanism used in this document.

### 4.1. Quantitative Keystone Lemma (Recall)

:::{prf:lemma} N-Uniform Quantitative Keystone Lemma (Positional Component)
:label: lem-quantitative-keystone-w2

Under the foundational axioms of {doc}`03_cloning`, there exist $R^2_{\text{spread}} > 0$, $\chi(\varepsilon) > 0$, and $g_{\max}(\varepsilon) \ge 0$, all independent of $N$, such that for any pair of swarms $(S_1, S_2)$:

$$
\frac{1}{N}\sum_{i \in I_{11}} (p_{1,i} + p_{2,i})\|\Delta\delta_{x,i}\|^2 \ge \chi(\varepsilon) V_{\text{struct}} - g_{\max}(\varepsilon)

$$

This is Lemma 8.1.1 in {doc}`03_cloning` ({prf:ref}`lem-quantitative-keystone`).
:::

### 4.2. Positional Variance Drift

:::{prf:theorem} Positional Variance Proxy Drift
:label: thm-positional-variance-proxy

Define the variance proxy
$V_{\text{x,proxy}} := \text{Var}_x(S_1) + \text{Var}_x(S_2)$.
In the all-alive regime, $V_{\text{x,proxy}}$ agrees with the $N$-normalized variance component $V_{\text{Var},x}(S_1) + V_{\text{Var},x}(S_2)$ from {doc}`03_cloning`, and the cloning operator satisfies:

$$
\mathbb{E}[\Delta V_{\text{x,proxy}}] \leq -\kappa_x V_{\text{x,proxy}} + C_x

$$

with N-uniform
$\kappa_x = \frac{\chi(\varepsilon)}{4} c_{\text{struct}}$
and
$C_x = \frac{g_{\max}(\varepsilon)}{4} + C_{\text{jitter}}$.
Here $c_{\text{struct}} > 0$ is the structural-variance link constant from {doc}`03_cloning` (Section 10.3.6), and $C_{\text{jitter}} = 4 d \delta^2$ is a conservative bound from the positional cloning jitter.

**Reference**: This is a direct restatement of {doc}`03_cloning`, Theorem 10.3.1 ({prf:ref}`thm-positional-variance-contraction`), specialized to the all-alive regime.
:::

:::{prf:remark} Jitter Scale Convention
:label: rem-jitter-scale

$\delta$ is the positional jitter scale in the cloning update. In the Euclidean Gas implementation, one typically sets $\delta = \sigma_x$ (or $\delta = \sqrt{\tau}\,\sigma_x$ for a discretized step), but the analysis keeps $\delta$ explicit.
:::

---

## 5. From Variance Contraction to Centered $W_2$ Control

We now combine the proxy drift with the centered Wasserstein bound.

:::{prf:proposition} Centered Positional Control via Variance Proxy
:label: prop-centered-w2-control

Under the conditions of Theorem {prf:ref}`thm-positional-variance-proxy`, the centered positional Wasserstein term satisfies:

$$
\mathbb{E}\left[V_{\text{x,struct}}(S_1', S_2')\right] \leq (1 - \kappa_x) V_{\text{x,proxy}}(S_1, S_2) + C_x.

$$

**Proof.**
By Lemma {prf:ref}`lem-centered-w2-variance-bound`, $V_{\text{x,struct}} \le V_{\text{x,proxy}}$. Apply Theorem {prf:ref}`thm-positional-variance-proxy` and take expectations. □
:::

:::{prf:remark} Closed Drift for $V_{\text{x,struct}}$
:label: rem-closed-drift-vxstruct

Without additional alignment structure, the bound above is **one-sided**: it controls $V_{\text{x,struct}}$ by a contractive proxy but does not produce a closed drift inequality in $V_{\text{x,struct}}$ alone. A regime-specific dominance assumption (Assumption {prf:ref}`ass-structural-dominance`) yields a closed drift bound (Corollary {prf:ref}`cor-closed-drift-vxstruct`).
This additional assumption is **not** required for the main control result.
:::

:::{prf:assumption} Structural-Dominance Regime (Positional)
:label: ass-structural-dominance

There exists an N-uniform constant $c_{\text{proxy}} \ge 1$ such that, at the times of interest,

$$
V_{\text{x,proxy}} \le c_{\text{proxy}} V_{\text{x,struct}}.
$$

**Interpretation**: the centered shape mismatch dominates the internal variance. This is a high-mismatch regime; it typically fails when the swarms are already nearly aligned.
:::

:::{prf:remark} Sufficient Geometric Condition
:label: rem-structural-dominance-sufficient

If the centered supports satisfy a separation condition, the dominance constant can be made explicit. Suppose both centered supports are contained in a ball of radius $R$ (e.g., $R \le D_{\text{valid}}$) and have minimal separation
$\operatorname{dist}(\operatorname{supp}\tilde{\mu}_{x,1}, \operatorname{supp}\tilde{\mu}_{x,2}) \ge D > 0$.
Then $\text{Var}_x(S_k) \le R^2$ and $V_{\text{x,struct}} \ge D^2$, so

$$
V_{\text{x,proxy}} \le 2 R^2 \le \frac{2 R^2}{D^2} V_{\text{x,struct}}.
$$

Thus the assumption holds with $c_{\text{proxy}} = 2 (R/D)^2$. This illustrates that the dominance regime corresponds to **strong shape mismatch** (large $D$ relative to $R$).
:::

:::{prf:corollary} Closed Drift Under Structural Dominance
:label: cor-closed-drift-vxstruct

Assume {prf:ref}`ass-structural-dominance` and Theorem {prf:ref}`thm-positional-variance-proxy`. Then:

$$
\mathbb{E}[\Delta V_{\text{x,struct}}] \le -\kappa_{\text{eff}} V_{\text{x,struct}} + C_x,
\qquad
\kappa_{\text{eff}} := 1 - (1-\kappa_x) c_{\text{proxy}}.
$$

In particular, if $c_{\text{proxy}} < 1/(1-\kappa_x)$, then $\kappa_{\text{eff}} > 0$ and the centered positional error contracts geometrically. The correction term is linear in $V_{\text{x,struct}}$, so larger mismatch yields stronger expected correction. When the mismatch is small and the dominance condition fails, the kinetic step provides the remaining contraction.
:::

---

## 6. Full $W_2$ Contraction After the Kinetic Step

The full phase-space $W_2$ contraction is obtained by combining the centered control above with the kinetic operator's barycenter and velocity contraction.

:::{prf:theorem} Structural/Barycenter Split for Full $W_2$
:label: thm-full-w2-split

Let $\mu_1, \mu_2$ be the empirical phase-space measures of two swarms. Then:

$$
W_2^2(\mu_1, \mu_2) = \|\bar{z}_1 - \bar{z}_2\|^2 + W_2^2(\tilde{\mu}_1, \tilde{\mu}_2).

$$

Cloning controls the centered positional component via Proposition {prf:ref}`prop-centered-w2-control`. The kinetic operator $\Psi_{\text{kin}}$ contracts the barycenter and velocity components ({doc}`05_kinetic_contraction`). Therefore the composed dynamics $\Psi_{\text{kin}} \circ \Psi_{\text{clone}}$ yields full phase-space $W_2$ contraction as in {doc}`06_convergence`.
:::

---

## 7. Comparison with Single-Walker Approach

### 7.1. Why Single-Walker Approach Failed

**Original approach** (in previous version):
```
Track individual pairs (i, π(i))
Need: min probability q_min over all matchings
Problem: q_min ~ 1/(N!) → 0 as N → ∞
Result: N-uniformity BROKEN
```

**Keystone approach** (this document):
```
Track variance proxy V_{x,proxy} = Var_x(S_1) + Var_x(S_2)
Use: Keystone constants χ(ε), g_max(ε) to get drift
No cross-swarm alignment assumptions required
Result: N-uniformity PRESERVED
```

### 7.2. Advantages Summary

| Aspect | Single-Walker | Keystone-Based |
|--------|---------------|---------------|
| **Coupling** | Individual matching with q_min | Variance proxy; no matching requirement |
| **Geometry** | Per-walker alignment (brittle) | No cross-swarm alignment needed |
| **Proof method** | Dynamic (survival probability) | Keystone lemma + variance drift |
| **N-uniformity** | BROKEN (q_min → 0) | ✓ PROVEN (Keystone constants from {doc}`03_cloning`) |
| **Framework consistency** | Ad-hoc definitions | Uses exact definitions from Chapters 6-8 |

---

## 8. Explicit Constants and Derived Bounds

### 8.1. Contraction Constant Components

We express each constant in terms of framework parameters and explicit bounds from {doc}`03_cloning`.

1. **High-error fraction** (Chapter 6):

$$
f_H(\varepsilon) := \min\left(f_O,\; f_{H,\text{cluster}}(\varepsilon)\right)
$$

with

$$
f_O = \frac{(1-\varepsilon_O) R_h^2}{D_h^2}, \qquad
f_{H,\text{cluster}}(\varepsilon) = \frac{(1-\varepsilon_O)\left(R^2_{\text{var}} - (D_{\text{diam}}(\varepsilon)/2)^2\right)}{D_{\text{valid}}^2},
$$

and $D_{\text{diam}}(\varepsilon) = c_d \varepsilon$.
Here $D_h^2 := D_x^2 + \lambda_v D_v^2$ is the hypocoercive diameter.

2. **Stability-gap margin** (Theorem 7.5.2.4, {doc}`03_cloning`):

$$
\Delta_{\log}(\varepsilon) :=
\beta \ln\left(1 + \frac{\kappa_{d',\text{mean}}(\varepsilon)}{g_{A,\max}+\eta}\right)
-
\alpha \ln\left(1 + \frac{\kappa_{\mathrm{rescaled}}(L_R D_{\text{valid}})}{\eta}\right),
\qquad \Delta_{\log}(\varepsilon) > 0
$$

which implies a mean fitness gap

$$
\Delta_{\text{fit}}(\varepsilon) \ge V_{\text{pot,min}}\left(e^{\Delta_{\log}(\varepsilon)} - 1\right).
$$

3. **Unfit-high-error overlap** (explicit conservative bound):

$$
f_{UH}(\varepsilon) \ge f_H(\varepsilon) \cdot
\frac{\Delta_{\text{fit}}(\varepsilon)}{V_{\text{pot,max}} - V_{\text{pot,min}}}
$$

with $V_{\text{pot,min}} = \eta^{\alpha+\beta}$ and $V_{\text{pot,max}} = (g_{A,\max} + \eta)^{\alpha+\beta}$.

4. **Unfit fraction** (Lemma 7.6.1.1, {doc}`03_cloning`):

$$
f_U(\varepsilon) = \frac{\kappa_{V,\text{gap}}(\varepsilon)}{2\left(V_{\text{pot,max}} - V_{\text{pot,min}}\right)}.
$$

5. **Cloning pressure** (Lemma 8.3.2 / Section 8.6.1.1, {doc}`03_cloning`):

$$
p_u(\varepsilon) = \min\left(1,\; \frac{1}{p_{\max}} \cdot
\frac{\Delta_{\min}(\varepsilon, f_U, f_F, k)}{V_{\text{pot,max}} + \varepsilon_{\text{clone}}}\right),
\qquad
\Delta_{\min}(\varepsilon, f_U, f_F, k) := \frac{f_F f_U}{(k-1)(f_F + f_U^2/f_F)} \kappa_{V,\text{gap}}(\varepsilon).
$$

The N-uniform lower bound implied by Theorem 8.7.1 in {doc}`03_cloning` is used throughout this document.
In the all-alive regime used here, $k = N$.

6. **High-error concentration constant** (Lemma 8.4.1 in {doc}`03_cloning`):

$$
c_H(\varepsilon) := \min\left\{1-\varepsilon_O, \frac{(1-\varepsilon_O)\left(R^2_{\text{var}} - (D_{\text{diam}}(\varepsilon)/2)^2\right)}{R^2_{\text{var}}}\right\},
\qquad
c_{\text{err}}(\varepsilon) = \frac{c_H(\varepsilon)}{4}.
$$

7. **Error offset** (Lemma 8.4.1 in {doc}`03_cloning`):

$$
g_{\text{err}}(\varepsilon) = \left(\frac{c_H(\varepsilon)}{2} + 5\right) D_{\text{valid}}^2.
$$

8. **Keystone feedback coefficient**:

$$
\chi(\varepsilon) = p_u(\varepsilon) \cdot c_{\text{err}}(\varepsilon).
$$

9. **Keystone offset** (Section 8.6.2, {doc}`03_cloning`):

$$
g_{\max}(\varepsilon) = \max\left(p_u(\varepsilon) \cdot g_{\text{err}}(\varepsilon),\; \chi(\varepsilon) R_{\text{spread}}^2\right).
$$

10. **Positional variance drift constants** (Theorem 10.3.1, {doc}`03_cloning`):

$$
\kappa_x = \frac{\chi(\varepsilon)}{4} c_{\text{struct}}, \qquad
C_x = \frac{g_{\max}(\varepsilon)}{4} + 4 d \delta^2.
$$

Here $c_{\text{struct}} > 0$ is the structural-variance link constant from {doc}`03_cloning` (Section 10.3.6); in balanced all-alive regimes, $c_{\text{struct}} = 1/2$ is a conservative choice.

### 8.2. Convergence Rate (Proxy)

The variance proxy satisfies geometric decay:

$$
\mathbb{E}[V_{\text{x,proxy}}(t)] \leq (1 - \kappa_x)^t V_{\text{x,proxy}}(0) + \frac{C_x}{\kappa_x}.
$$

By Lemma {prf:ref}`lem-centered-w2-variance-bound`, this yields:

$$
\mathbb{E}[V_{\text{x,struct}}(t)] \leq \mathbb{E}[V_{\text{x,proxy}}(t)].
$$

Full phase-space $W_2$ contraction follows after composing with the kinetic operator, which contracts the barycenter and velocity components ({doc}`05_kinetic_contraction`, {doc}`06_convergence`).

### 8.3. Comparison with KL-Convergence

The KL-convergence framework ({doc}`15_kl_convergence`) may provide faster convergence rates via entropy methods. The centered/structural Wasserstein-2 control proven here is complementary:

- **Centered positional $W_2$ control (via variance proxy)**: Geometric proxy decay, explicit constants, suitable for mean-field limit
- **KL contraction**: Entropy-based, potentially faster, uses LSI theory

Both approaches are valid; the Keystone-based variance proxy control provides an independent verification of convergence with explicit N-uniform constants.

---

## 9. Conclusion and Future Work

### 9.1. Main Achievements

This document establishes **centered positional $W_2$ control** for the cloning operator using a **Keystone-based variance proxy** that:

1. ✅ **Avoids q_min problem**: No dependence on minimum matching probability
2. ✅ **Leverages Keystone bounds**: Constants sourced from {doc}`03_cloning` Chapters 6-8
3. ✅ **No alignment axiom**: No cross-swarm geometric alignment assumptions required
4. ✅ **N-uniform throughout**: All constants independent of N
5. ✅ **Framework-consistent**: Uses exact cluster definitions from the Keystone Lemma chain

### 9.2. Open Questions

1. **Optimal constants**: Can $\chi(\varepsilon)$ or $c_{\text{struct}}$ be tightened to improve $\kappa_x$?
2. **Closed drift for $V_{\text{x,struct}}$**: Identify regimes where $V_{\text{x,proxy}} \le c_{\text{proxy}} V_{\text{x,struct}}$ holds.
3. **Adaptive extensions**: How does viscous coupling affect the proxy decay rate?
4. **Numerical validation**: Swarm simulations to verify proxy decay and combined kinetic+cloning $W_2$ contraction

### 9.3. Relation to Framework

This result enables:
- **Propagation of Chaos** ({doc}`09_propagation_chaos`): N-particle system → mean-field limit
- **Mean-Field Convergence** ({doc}`08_mean_field`): Measure-level contraction
- **Combined with kinetic contraction**: Full Wasserstein contraction for alternating operator $\Psi_{\text{kin}} \circ \Psi_{\text{clone}}$

---

## References

**Primary**: {doc}`03_cloning` Chapters 6-8 (Keystone Principle) and Chapter 10 (variance drift)

**Key Results Used**:
- Definition 6.3 (line 2351): Unified High-Error and Low-Error Sets
- Definition 7.6.1.0 (line 4499): Unfit Set
- Theorem 7.6.1 (line 4572): Unfit-High-Error Overlap (f_UH > 0)
- Lemma 8.3.2 (line 4881): Cloning Pressure on Unfit Set (p_u > 0)
- Lemma 8.4.1: Error Concentration in the Target Set ($c_{\text{err}}, g_{\text{err}}$)
- Lemma 8.1.1: Quantitative Keystone Lemma ($\chi, g_{\max}$)
- Theorem 10.3.1: Positional Variance Contraction
- Theorem 7.5.2.4: Stability Condition (fitness ordering)
- Theorem 8.7.1 (line 5521): N-Uniformity of Keystone Constants

**Secondary**:
- {doc}`01_fragile_gas_framework`: Axioms
- {doc}`15_kl_convergence`: Alternative convergence analysis

---

**Document Status**: COMPLETE (Keystone-Based Proxy Control)

**Next Steps**: Numerical validation + comparison with KL-convergence rates
