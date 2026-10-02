# Wasserstein-2 Control from Signed Keystone Accounting

## 0. TLDR

**Centered positional control under cloning**: We work with phase-space $W_2$ on $z=(x,v)$ and the barycenter decomposition
$W_2^2(\mu_1, \mu_2) = \|\bar{z}_1 - \bar{z}_2\|^2 + W_2^2(\tilde{\mu}_1, \tilde{\mu}_2)$. For the centered positional marginals, let
$V_{\text{x,struct}} := W_{2,x}^2(\tilde{\mu}_{x,1}, \tilde{\mu}_{x,2})$ and define the variance proxy
$V_{\text{x,proxy}} := \text{Var}_x(S_1) + \text{Var}_x(S_2)$.
Lemma {prf:ref}`lem-centered-w2-variance-bound` shows $V_{\text{x,struct}} \le V_{\text{x,proxy}}$.
The source positional theorem gives an $N$-uniform reset bound
$\mathbb E V_{\text{x,proxy}}'\le C_{\rm reset}$.
The Keystone Lemma separately gives an $N$-uniform lower bound on
error-weighted cloning pressure. Its conversion to signed drift uses
the actual donor, collision and kinetic terms in
{prf:ref}`thm-slc-signed-complete-update`.

**No alignment axiom**: The pressure estimate does not require a minimum
matching probability. The full-update calculation uses the specified
common-source coupling and keeps its mismatch costs.

**Complete-update contraction requires coupled estimates**: The variance proxy bounds centered positional discrepancy but can remain positive when two laws coincide. Full phase-space contraction requires matched component estimates for the coupled cloning and kinetic updates, followed by a verified composition inequality. See {doc}`06a_structural_landscape_convergence` for that accounting.

**Explicit constants**: The Keystone coefficient and its finite-size
correction are in {prf:ref}`thm-slcn-keystone-power`; the reset constant
is in {prf:ref}`thm-positional-variance-proxy`. Neither is by itself
a full-update contraction coefficient.

**Dependencies**: {doc}`03_cloning`, {doc}`02_euclidean_gas`

## 1. Introduction

### 1.1. Goal and Scope

This document records the centered Wasserstein decomposition, the
source Keystone pressure estimate and the cloning variance reset.
The signed complete-update derivation that combines pressure with
donor transfer, collisions and kinetics is
{prf:ref}`thm-slc-signed-complete-update`. Its TV conversion for
sampled marked positions is {prf:ref}`thm-slc-keystone-tagged-tv`.

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
The cloning stage gives the reset bound

$$
\mathbb E\Delta V_{\text{x,proxy}}
\le -V_{\text{x,proxy}}+C_{\rm reset},

$$

whose offset may cover the entire input range. The bound is not a
strict contraction attributable to Keystone pressure. By coercivity
in {doc}`03_cloning`, the hypocoercive structural error satisfies
$V_{\text{struct}} \geq \lambda_2 W_2^2(\tilde{\mu}_1, \tilde{\mu}_2) \geq \lambda_2 V_{\text{x,struct}}$. The barycenter term $\|\bar{z}_1 - \bar{z}_2\|^2$ is handled by the kinetic operator.

The Keystone pressure coefficient is independent of $N$ because its
coverage proof uses normalized error over all geometric clusters.
The signed donor and kinetic balance must be evaluated before that
coefficient becomes a complete-update rate.

The scope here is the centered decomposition and the exact input
supplied by the Keystone and reset theorems. The complete-update
calculation retains the Keystone mechanism in
{doc}`06a_structural_landscape_convergence`.

### 1.2. Why Wasserstein-2 Contraction Matters

Centered Wasserstein-2 control via the variance proxy quantifies one contribution of cloning to the convergence programme. Its relation to a population limit depends on separate consistency and stability estimates.

**Connection to Mean-Field Theory**: The finite-horizon results in {doc}`09_propagation_chaos` use consistency, moment control and continuity of the actual fixed-step population map. They do not require an N-uniform contraction rate. A matching N-uniform two-swarm contraction can additionally control accumulated errors over long times, as described in {doc}`06a_structural_landscape_convergence`. A variance reset alone does not supply that two-law estimate.

**Role in Convergence Theory**: The Fractal Gas alternates cloning and kinetics. Its keystone–kinetic proof architecture combines complementary estimates under their declared hypotheses:

- **Cloning operator**: Supplies an $N$-uniform Keystone pressure bound
  and a separate positional reset bound. Their signed donor balance is
  retained before any rate is claimed.
- **Kinetic operator**: Supplies dissipation and cross-component estimates for an appropriate coupled observable.

The complete-update analysis in
{doc}`06a_structural_landscape_convergence` retains the cross-component
terms and the landscape-dependent defects. The reset bound supplies
moment control; it is not substituted for the signed Keystone term.

**Complementary to KL-Convergence**: {doc}`15_kl_convergence` studies entropy and functional inequalities for its specified laws and dynamics. Comparing those estimates with Wasserstein control requires matching their hypotheses and targets. Neither a static LSI nor the variance proxy alone establishes contraction for the complete gas.

:::{important}
**What N-uniformity supplies**

The Keystone pressure coefficient is independent of $N$, with its
explicit self-exclusion correction. That fact is retained in
(SCK.5)--(SCK.6). It does not imply an $N$-uniform drift
coefficient for the variance proxy or an $N$-uniform full-swarm TV
mixing rate. Uniform-time conclusions use the complete signed update,
not this one-sided proxy bound.
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
        D["<b>§4.1: Quantitative Keystone Lemma</b><br>χ(ε), g_max(ε) from the 03_cloning chapter"]:::lemmaStyle
        E["<b>§4.2: Positional Reset</b><br>𝔼[V′_{x,proxy}] ≤ C_reset"]:::theoremStyle
    end

    subgraph "Main Result (§5-6)"
        F["<b>§5: Centered W₂ Control</b><br>V_{x,struct} ≤ V_{x,proxy}<br>Proxy drift yields control"]:::stateStyle
        G["<b>§6: Structural/Barycenter Split</b><br>Components requiring coupled estimates"]:::theoremStyle
    end

    subgraph "Analysis (§7-8)"
        H["<b>§7: Comparison</b><br>No q_min; no alignment axiom"]:::stateStyle
        I["<b>§8: Explicit Constants</b><br>Keystone pressure and reset tracked"]:::stateStyle
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

**Section 4 (Keystone Core)**: We import the Quantitative Keystone
Lemma and the separate positional reset theorem from {doc}`03_cloning`.

**Section 5 (Centered $W_2$ Control)**: We combine the reset bound
with $V_{\text{x,struct}}\le V_{\text{x,proxy}}$ for moment control;
the signed Keystone--kinetic update supplies the rate calculation.

**Section 6 (Full $W_2$ Split)**: We identify the barycenter and centered phase-space components whose coupled estimates must close under cloning–kinetic composition.

**Section 7 (Comparison)**: We contrast the Keystone-based variance approach with the failed single-walker $q_{\min}$ strategy.

**Section 8 (Explicit Constants)**: We distinguish pressure and reset
constants from a complete-update rate.

**Key Proof Principles**:

1. **Variance proxy**: Control $V_{\text{x,struct}}$ via $V_{\text{x,proxy}} = \text{Var}_x(S_1) + \text{Var}_x(S_2)$
2. **Keystone constants**: Use $f_{UH}(\varepsilon)$, $p_u(\varepsilon)$, and $\chi(\varepsilon)$ from {doc}`03_cloning` (already N-uniform)
3. **No cross-swarm alignment**: Avoid brittle alignment assumptions and $q_{\min}$ arguments
4. **Framework consistency**: Use exact definitions from the Keystone Lemma proof

The result is a rigorous centered Wasserstein decomposition and
reset bound, with the $N$-uniform Keystone pressure carried into
the complete-update proof cited above.



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

**Finite-population correction**: The complete measurement-averaged
Keystone estimate in {prf:ref}`thm-keystone-discharged-averaged-pressure`
has an explicit $N^{-2}$ self-exclusion term. It is not a generic
$O(N^{-1/2})$ approximation of the reward or force. The actual
finite-swarm measurements and common empirical normalizers are used
throughout that proof. Population-limit sampling errors belong to the
separate consistency analysis in {doc}`09_propagation_chaos`.

**N-uniformity**: The positive Keystone pressure coefficient does
not contain $N$. This claim concerns pressure, not a closed drift
coefficient for the variance proxy. The complete signed donor and
kinetic terms are retained in
{prf:ref}`thm-slc-signed-complete-update`.
:::

### 2.2. No Cross-Swarm Alignment Assumption

This document does not assume any cross-swarm alignment or matching axiom. All geometric guarantees are imported from the Keystone Lemma chain in {doc}`03_cloning`. The only cross-swarm coupling used later is the standard independent coupling for bounding $W_{2,x}^2$ by internal variances (Lemma {prf:ref}`lem-centered-w2-variance-bound`), which requires no alignment structure.



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

then this term is bounded by the positional discrepancy. This inequality alone
does not control the centered joint position–velocity law or close the proxy
drift in the initial Wasserstein distance. A full contraction conclusion uses
matched coupled component estimates and {prf:ref}`thm-slc-defect-composition`.
:::



## 4. Keystone-Driven Positional Variance Contraction

We import the Keystone pressure estimate and the separate positional reset bound from {doc}`03_cloning`. The $N$-uniform pressure coefficient enters the signed complete-update calculation; the reset bound controls a moment without determining the sign of that calculation.

### 4.1. Quantitative Keystone Lemma (Recall)

:::{prf:lemma} N-Uniform Quantitative Keystone Lemma (Positional Component)
:label: lem-quantitative-keystone-w2

Under the foundational axioms of {doc}`03_cloning`, there exist $R^2_{\text{spread}} > 0$, $\chi(\varepsilon) > 0$, and $g_{\max}(\varepsilon) \ge 0$, all independent of $N$, such that for any pair of swarms $(S_1, S_2)$:

$$
\frac{1}{N}\sum_{i \in I_{11}} (p_{1,i} + p_{2,i})\|\Delta\delta_{x,i}\|^2 \ge \chi(\varepsilon) V_{\text{struct}} - g_{\max}(\varepsilon)

$$

This is Lemma 8.1.1 in {doc}`03_cloning` ({prf:ref}`lem-quantitative-keystone`).
:::

### 4.2. Positional variance and the signed Keystone bridge

:::{prf:theorem} Positional variance reset for the cloning stage
:label: thm-positional-variance-proxy

Define the variance proxy
$V_{\text{x,proxy}} := \text{Var}_x(S_1) + \text{Var}_x(S_2)$.
Suppose the eligible frozen donor positions in each swarm have diameter
at most $D_x$ and use the canonical recipient jitter
$\sigma_{\rm clone}$. The actual cloning proposal obeys

$$
\mathbb E V_{\text{x,proxy}}'
\le D_x^2+2(1-1/N)d\sigma_{\rm clone}^2=:C_{\rm reset},
\qquad
\mathbb E\Delta V_{\text{x,proxy}}
\le-V_{\text{x,proxy}}+C_{\rm reset}.
$$

This is precisely the two-swarm sum of
{prf:ref}`thm-positional-variance-contraction`. Its $N$-uniform
coefficient is a reset coefficient with a possibly large offset;
it is **not** $\chi(\varepsilon)c_{\rm struct}/4$ and does not
turn Keystone pressure into signed drift. The latter calculation is
(SCK.1)--(SCK.3) in
{prf:ref}`thm-slc-signed-complete-update`, with its marked version
{prf:ref}`thm-slc-marked-keystone-port`.

*Proof.* The cited single-swarm theorem gives
$\mathbb E\operatorname{Var}_x(S_s')
\le D_x^2/2+(1-1/N)d\sigma_{\rm clone}^2$ for each $s=1,2$.
Add the two inequalities and subtract the entering proxy. $\square$
:::

:::{prf:remark} Jitter Scale Convention
:label: rem-jitter-scale

$\sigma_{\rm clone}$ is the recipient-position jitter in cloning.
It is independent of the final kinetic position-noise scale
$\sigma_x\sqrt h$; identifying them would change both the update
and its constant.
:::



## 5. From Variance Contraction

The reset estimate gives a moment bound. The Keystone route to a
rate instead carries the exact signed donor term through collision
and kinetics before testing a landscape's regional constants.

:::{prf:proposition} Centered Positional Control via Variance Proxy
:label: prop-centered-w2-control

Under the conditions of {prf:ref}`thm-positional-variance-proxy`,
the centered positional Wasserstein term satisfies

$$
\mathbb E V_{\text{x,struct}}(S_1',S_2')\le C_{\rm reset}.
$$

*Proof.* Lemma {prf:ref}`lem-centered-w2-variance-bound` gives
$V_{\text{x,struct}}\le V_{\text{x,proxy}}$ at the output.
Apply the preceding reset theorem and take expectations. $\square$
:::

:::{prf:remark} Closed Drift for $V_{\text{x,struct}}$
:label: rem-closed-drift-vxstruct

This reset bound may cover the entire range of input positional
variance. It therefore yields no strict centered-$W_2$ contraction
rate. The rate calculation uses the Keystone pressure with the
donor, barycenter, collision and kinetic terms of the actual update,
as in {prf:ref}`thm-slc-keystone-tagged-tv` and
{prf:ref}`thm-slc-marked-keystone-port`.
:::



## 6. Full $W_2$ Contraction After the Kinetic Step

The full phase-space decomposition identifies what the signed
Keystone--kinetic calculation must control. The reset bound is a
moment estimate; it is not substituted as a strict contraction.

:::{prf:theorem} Structural/Barycenter Split for Full $W_2$
:label: thm-full-w2-split

Let $\mu_1, \mu_2$ be the empirical phase-space measures of two swarms. Then:

$$
W_2^2(\mu_1, \mu_2) = \|\bar{z}_1 - \bar{z}_2\|^2 + W_2^2(\tilde{\mu}_1, \tilde{\mu}_2).

$$

The identity holds for the ordinary Euclidean phase-space cost; for a fixed
positive-definite quadratic cost, use its norm in both terms. It identifies the
components required by a complete two-swarm contraction estimate. A centered
positional variance proxy alone is not that estimate. When coupled component
bounds control all these terms for the same full kernel, their composition is
{prf:ref}`thm-slc-defect-composition`; its population-limit transfer is
{prf:ref}`thm-slc-contraction-transfer` in the matching metric.

*Proof.* Every coupling of the two measures corresponds bijectively, by separate
translation, to a coupling of their centered measures. Write the displacement as
$z_1-z_2=(\bar z_1-\bar z_2)+(\tilde z_1-\tilde z_2)$. The cross term integrates
to zero because both centered marginals have zero mean. The mean-displacement
term is constant over couplings. Taking the infimum proves the identity; the
same calculation applies to a fixed quadratic form. $\square$
:::



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
Track the error-weighted cloning pressure with N-normalized errors.
Carry the signed donor, barycenter, collision and kinetic terms.
Result: the Keystone coefficient is N-uniform; a full-update rate
requires the displayed signed balance to close.
```

### 7.2. Advantages Summary

| Aspect | Single-Walker | Keystone-Based |
|--------|---------------|---------------|
| **Coupling** | Individual matching with $q_{\min}$ | Actual common-source plan and signed update |
| **Geometry** | Per-walker alignment | Error-weighted coverage of all declared clusters |
| **Proof method** | Minimum matching probability | Keystone pressure plus donor and kinetic accounting |
| **N-uniformity** | Lost in the matching minimum | Proved for pressure; full-update rate is a separate calculation |
| **Kernel** | Depends on the proposed matching | Uses the canonical accepted graph and complete update |



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

10. **Positional variance reset constant**
({prf:ref}`thm-positional-variance-contraction`):

$$
C_{\rm reset}=D_x^2+2(1-1/N)d\sigma_{\rm clone}^2.
$$

The Keystone constants in items 1--9 are pressure constants. The
reset constant here is a moment constant. There is no derived
$\kappa_x=\chi c_{\rm struct}/4$ in the source positional theorem.

### 8.2. Reset control and the complete-update rate

The reset estimate gives, for each cloning output,

$$
\mathbb E V_{\text{x,proxy}}'\le C_{\rm reset}.
$$

The centered positional Wasserstein term therefore obeys

$$
\mathbb E V_{\text{x,struct}}'\le C_{\rm reset}.
$$

The signed complete-update rate calculation is (SCK.3)--(SCK.6)
in {prf:ref}`thm-slc-signed-complete-update`. Its direct sampled-row
TV conversion is {prf:ref}`thm-slc-keystone-tagged-tv`. They retain
the reward-dependent donor flux, barycenter, collisions and kinetic
terms that the reset bound does not evaluate.

### 8.3. Comparison with KL-Convergence

The KL-convergence framework ({doc}`15_kl_convergence`) may provide faster convergence rates via entropy methods. The centered/structural Wasserstein-2 control proven here is complementary:

- **Centered positional $W_2$ control (via variance proxy)**: A
  reset bound for the cloning stage
- **KL contraction**: Entropy-based, potentially faster, uses LSI theory

Each approach retains its own hypotheses. The proxy estimate controls one geometric component; it is not an independent proof of complete-law convergence.



## 9. Conclusion and Future Work

### 9.1. Main Achievements

This document establishes the centered positional $W_2$ decomposition,
the cloning reset bound and the $N$-uniform Keystone pressure input:

1. ✅ **Avoids q_min problem**: No dependence on minimum matching probability
2. **Keystone pressure**: Constants sourced from {doc}`03_cloning`
3. ✅ **No alignment axiom**: No cross-swarm geometric alignment assumptions required
4. **N-uniform input**: The pressure coefficient is independent of $N$;
   the finite-size correction is explicit
5. ✅ **Framework-consistent**: Uses exact cluster definitions from the Keystone Lemma chain

### 9.2. Open Questions

1. **Signed rate evaluation**: Evaluate the donor and kinetic terms in
   (SCK.3) with the declared regional landscape profiles.
2. **Phase-resolved control**: Apply the resulting rate only on the
   population classes certified by those profiles.

### 9.3. Relation to Framework

This result contributes positional control to the larger programme:

- **Finite-horizon propagation of chaos** ({doc}`09_propagation_chaos`) also requires the actual population map's consistency and continuity estimates.
- **Long-time population analysis** ({doc}`08_mean_field`) requires additional stability or phase-identification arguments.
- **Complete-update contraction** requires matched cloning–kinetic coupling estimates and a composition criterion, with the landscape dependence retained as in {doc}`06a_structural_landscape_convergence`.



## References

**Primary**: {doc}`03_cloning` Chapters 6-8 (Keystone Principle) and Chapter 10 (variance drift)

**Key Results Used**:
- Definition 6.3 (line 2351): Unified High-Error and Low-Error Sets
- Definition 7.6.1.0 (line 4499): Unfit Set
- Theorem 7.6.1 (line 4572): Unfit-High-Error Overlap (f_UH > 0)
- Lemma 8.3.2 (line 4881): Cloning Pressure on Unfit Set (p_u > 0)
- Lemma 8.4.1: Error Concentration in the Target Set ($c_{\text{err}}, g_{\text{err}}$)
- Lemma 8.1.1: Quantitative Keystone Lemma ($\chi, g_{\max}$)
- Theorem 10.3.1: Positional variance reset and exact drift
- Theorem 7.5.2.4: Stability Condition (fitness ordering)
- Theorem 8.7.1 (line 5521): N-Uniformity of Keystone Constants

**Secondary**:
- {doc}`01_fragile_gas_framework`: Axioms
- {doc}`15_kl_convergence`: Alternative convergence analysis



**Document Status**: COMPLETE (Keystone-Based Proxy Control)

**Next Steps**: Numerical validation + comparison with KL-convergence rates
