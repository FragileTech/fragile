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

:::{div} feynman-prose
**Completed population-independent alive-law relaxation**:
In the conservative current-frame gas with no viscosity or history term,
{prf:ref}`thm-slcw-finite-uniform-law` proves exact relaxation to each
finite swarm's invariant law, with a coefficient and rate independent of
$N$, for the stated bounded-reward and force-center regime. Active cloning
remains present, and the explicit positive-exponent interval includes
realized fitness ties. The transport target is the law of the random alive
empirical measure, not a prescribed pairing of sample paths.

For an unbounded quadratic-growth raw reward,
{prf:ref}`thm-slcw-alive-uniform-law` instead gives a population-independent
time rate toward the stationary population law, with an explicit finite-particle
error tending to zero as $N$ increases. It can use the same potential for
force and reward. These results require the displayed bounded completed
force-center profile; they do not certify the original dense-viscosity preset.
The terminal-box QSD theorem below has its separate survival-conditioned scope.
:::

**Explicit constants**: The averaged Keystone coefficient, velocity remainder
and $N^{-2}$ self-exclusion correction are stated in Section 8.1; the reset constant
is in {prf:ref}`thm-positional-variance-proxy`. Neither is by itself
a full-update contraction coefficient.

**Completed finite-population convergence**: Under the explicit force, noise and timestep tests in {prf:ref}`def-w2-finite-population-qsd-regime`, the existing terminal-box gas converges to its full marked QSD conditional on nonextinction, including exact fitness ties. Its alive-sampled phase-space marginal converges in $W_2$. The rate is explicit and may depend on $N$; it does not require a positive realized fitness gap.

**Dependencies**: {doc}`03_cloning`, {doc}`02_euclidean_gas`, {doc}`18_coupled_gas_discharge`

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

:::{div} feynman-prose
The measurement-averaged Keystone coefficient is independent of $N$ because
its complete-coverage proof uses normalized error over every geometric cluster.
The complete signed calculation retains donor transfer, noise, collisions,
kinetics, and terminal marking. To turn it into an alive-swarm transport bound,
we must use the actual alive normalization, discard dead coordinates, and
account for optimal matching between the surviving populations. A lower bound
on the cost of one prescribed coupling supplies no lower bound on that
optimum. A survival-conditioned comparison uses the normalized surviving
output laws. Disabling deaths instead defines a different transition; its
transport estimate must use that transition throughout.

The distinction also matters for centering. A pure translation changes the
barycenters while leaving the centered empirical clouds identical. Its
fixed-label positional cost is positive, but its centered transport cost
vanishes. The exact targets are kept separate in
{prf:ref}`rem-w2-global-quadratic-obstructions`.

The positive law results use a different comparison. Two-update kinetic
smoothing can make rows agree exactly, and the finite sampled-normalization
and component calculation bounds how active cloning can undo that agreement.
With bounded reward, this yields an exact population-independent rate toward
the finite-swarm invariant law. With a raw quadratic-growth reward, the
weighted population estimate and finite-particle transfer give a rate toward
the mean-field stationary law plus a vanishing population error. The formal
scopes are recorded in {prf:ref}`rem-w2-completed-conservative-alive-law`.

The terminal-box theorem instead uses kernel smoothing and survival
conditioning at fixed population size. Extending a population-independent
alive-law rate to the original dense-viscosity and killed update still needs
a proof for that complete transition.

For a local rate, picture two clouds near a stable potential well. A force
coercive on the region containing both clouds can pull their positions toward
each other. Near a ridge, the force linearization can push them apart. The
Rastrigin potential has regions of both signs of curvature, so arbitrary
entering states need not move closer at every update. A shared phase label
alone supplies no quantitative rate: the force, velocities, copying, noise,
and survival terms need verified bounds on the region used by the theorem.
:::

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
        D["<b>§4.1: Averaged Keystone Input</b><br>χ*, positional threshold, velocity remainder<br>and N⁻² self-exclusion correction"]:::lemmaStyle
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
    D --> G
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

**Section 2 (Cluster Structure)**: We recall the realized target $I_k=U_k\cap H_k(\varepsilon)$ and its complement from {prf:ref}`def-unfit-set` and {prf:ref}`def-unified-high-low-error-sets`. A positive target fraction requires a signed realized fitness gap; it is not assumed in the measurement-averaged route.

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
2. **Keystone constants**: Use the complete measurement-averaged estimate in {prf:ref}`thm-keystone-discharged-averaged-pressure`, with its velocity remainder and finite-population correction. The realized-target route requires the additional hypotheses stated in Section 8.2.
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

Condition on the entering swarm and its complete realized fitness vector
$V_i$. Let $\mu_V=N^{-1}\sum_iV_i$ and put

$$
U_k=\{i\in\mathcal A_k:V_i\leq\mu_V\},\qquad
I_k(\varepsilon)=U_k\cap H_k(\varepsilon),\qquad
J_k(\varepsilon)=\mathcal A_k\setminus I_k(\varepsilon).
$$

Here $U_k$ is {prf:ref}`def-unfit-set`, and the geometrically defined
$H_k(\varepsilon)$ is {prf:ref}`def-unified-high-low-error-sets`. In the
all-alive regime, define $f_I=|I_k|/N$ and $f_J=1-f_I$.

A positive lower bound requires additional input. If $H_k,L_k$ partition the
alive rows, have fractions $f_H,f_L>0$, and their realized fitness means obey
$\mu_{V,L}-\mu_{V,H}\geq\Delta_V>0$, then
{prf:ref}`thm-unfit-high-error-overlap-fraction` gives

$$
f_I\geq\frac{f_Hf_L\Delta_V}{R_V},\qquad
R_V=\max_iV_i-\min_iV_i>0.
$$

Population-independent lower bounds on $f_H,f_L,\Delta_V$ and an upper bound
on $R_V$ give a population-independent overlap. A symmetric realized fitness
vector can have $I_k=H_k$ and zero cloning probability, so overlap alone
supplies no pressure bound.
:::

:::{prf:remark} All-Alive Normalization
:label: rem-all-alive-normalization

The input comparisons in this document concern all-alive swarms, $|\mathcal A_k|=N$. Their alive-law and fixed-slot empirical normalizations therefore coincide. The cloning proposal revives dead recipients when donors exist; extinction and the killed transition require separate treatment. The partially alive version of the averaged pressure estimate retains alive mass and unmatched-label terms in {prf:ref}`thm-keystone-discharged-averaged-pressure`.
:::

:::{prf:remark} Why These Sets?
:label: rem-why-target-sets

The target set combines a geometric error criterion with a below-mean fitness
criterion. Its contribution to the pressure sum requires both a positive-part
fitness comparison and error carried by the same labels. Section 8.2 states
these two conditional estimates explicitly. The main measurement-averaged
route in Section 8.1 covers every geometric cluster and retains every fitness
tie, so it needs no positive realized target overlap.
:::

:::{prf:remark} Empirical Measures and Framework Properties
:label: rem-empirical-measures

**Notational Precision**: This document analyzes the discrete $N$-particle empirical measures $\mu_1,\mu_2$. The fitness vector uses the actual sampled companions and shared empirical normalizers. It is not a deterministic function of one position $x$. The fixed reward landscape and realized geometric clusters enter its conditional law.

**Variance Notation**: $V_{\text{struct}}$ denotes the hypocoercive structural error between centered **phase-space** measures (as in {doc}`03_cloning`). We also use the positional structural term
$V_{\text{x,struct}} := W_{2,x}^2(\tilde{\mu}_{x,1}, \tilde{\mu}_{x,2})$ for centered positional marginals and the variance proxy
$V_{\text{x,proxy}} := \text{Var}_x(S_1) + \text{Var}_x(S_2)$. $\text{Var}_x(S_k)$ denotes the internal positional variance of swarm $k$.

**Relationship to Continuum Limit**: The reward landscape is fixed, while fitness and clusters depend on the current finite swarm. The pressure estimates use that finite-swarm law. Passing to a continuum fitness operator requires the separate consistency estimates in {doc}`09_propagation_chaos`.

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

The centered Wasserstein bound uses the independent transport plan in {prf:ref}`lem-centered-w2-variance-bound`. The pressure estimates use a comparison matching fixed from the entering states; any such matching is an admissible plan, with its actual positional and velocity errors retained. Their conversion to a full-update rate uses the specified common-source transition coupling in {prf:ref}`thm-slc-signed-complete-update`, including its mismatch costs.



## 3. Variance Decomposition and Centered Wasserstein Bound

### 3.1. Within-Swarm Variance Decomposition

We first establish how variance decomposes with respect to the cluster partition.

:::{prf:lemma} Variance Decomposition by Clusters
:label: lem-variance-decomposition

For a swarm $S_k$ partitioned into nonempty sets $I_k$ (target) and $J_k$ (complement), with population fractions $f_I=|I_k|/N$ and $f_J=|J_k|/N$:

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

Dividing by $N$ gives the result. For an empty part, omit its barycenter and variance; the identity reduces to the variance of the other part. $\square$
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



## 4. Keystone Pressure and Positional Variance Reset

We import the Keystone pressure estimate and the separate positional reset bound from {doc}`03_cloning`. The $N$-uniform pressure coefficient enters the signed complete-update calculation; the reset bound controls a moment without determining the sign of that calculation.

### 4.1. Quantitative Keystone Lemma (Recall)

:::{div} feynman-prose
The average is taken after the complete fitness vector has been sampled.
A tied vector contributes zero live-to-live selection. Other measurement
outcomes can supply pressure even in a symmetric cloud. The complete-coverage
estimate counts those outcomes and keeps the cost of excluding self donors.
It also keeps the velocity error: positional copying cannot dissipate a purely
velocity discrepancy by itself.
:::

:::{prf:lemma} Measurement-averaged Keystone input with explicit remainders
:label: lem-quantitative-keystone-w2

Use the fixed kernel, bounded entering alive region, complete reward regularity,
and configured positive diversity exponent of
{prf:ref}`lem-keystone-complete-coverage-constants`. Consider two all-alive
entering swarms with a fixed comparison matching. Define

$$
W=\frac1N\sum_i\|\Delta\delta_{x,i}\|^2,\qquad
D_v=\frac1N\sum_i\|\Delta\delta_{v,i}\|^2,\qquad
Q=\frac1N\sum_i(p_{1,i}+p_{2,i})\|\Delta\delta_{x,i}\|^2.
$$

Condition all expectations on the complete entering states. For any
$\eta>0$, put $c_v=\lambda_v+b^2/(4\eta)$, where the positive-definite
hypocoercive cost is
$q(\Delta x,\Delta v)=\|\Delta x\|^2+\lambda_v\|\Delta v\|^2+
b\langle\Delta x,\Delta v\rangle$, with $b^2<4\lambda_v$.
The positive constants $C_0,E_{\max},M_r,W_0$ defined in Section 8.1 give

$$
\chi_*=\frac{C_0W_0^2}{2E_{\max}^2M_r^2}>0,
$$

and, at every population size,

$$
\mathbb E Q\geq
\frac{\chi_*}{1+\eta}V_{\mathrm{struct}}
-\chi_*\left[W_0+\frac{c_vD_v}{1+\eta}\right]
-\frac{C_0E_{\max}}{N^2}.
$$

The remainders remain explicit. The coefficient is population independent;
a positive statewise lower bound requires that its structural contribution
exceed these remainders.

*Proof.* The all-alive case of
{prf:ref}`thm-keystone-discharged-averaged-pressure` gives
$\mathbb E Q\geq\chi_*(W-W_0)-C_0E_{\max}/N^2$.
The fixed comparison matching is an admissible transport plan. Applying
$|b|\|\Delta x\|\|\Delta v\|\leq\eta\|\Delta x\|^2+
b^2\|\Delta v\|^2/(4\eta)$ to that plan gives
$V_{\mathrm{struct}}\leq(1+\eta)W+c_vD_v$.
Solve for $W$ and substitute into the pressure estimate. This argument uses
the actual velocity remainder and includes every measurement outcome.
$\square$
:::

The optional realized-target form is {prf:ref}`lem-quantitative-keystone`.
Its selection and error-concentration hypotheses are spelled out in Section 8.2.

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



## 5. Centered Positional Control from the Reset Bound

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



## 6. Components Required for Full $W_2$ Contraction

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

:::{div} feynman-prose
The Keystone estimate tracks error-weighted cloning pressure with
$N$-normalized errors and a coefficient independent of $N$. The full update
also contains donor, barycenter, collision, kinetic, and terminal-mark terms.
The fixed-slot quadratic counterexamples in
{prf:ref}`rem-w2-global-quadratic-obstructions` retain their array and coupling
scopes. The alive empirical $W_2$ permits optimal row matching and uses the
surviving mass. Its population-independent estimate remains unresolved.
:::

### 7.2. Advantages Summary

:::{div} feynman-added
| Aspect | Single-Walker | Keystone-Based |
|--------|---------------|---------------|
| **Coupling** | Individual matching with $q_{\min}$ | Actual common-source plan and signed update |
| **Geometry** | Per-walker alignment | Error-weighted coverage of all declared clusters |
| **Proof method** | Minimum matching probability | Keystone pressure plus donor and kinetic accounting |
| **N-uniformity** | Lost in the matching minimum | Proved for pressure; alive-$W_2$ rate remains unresolved |
| **Kernel** | Depends on the proposed matching | Uses the canonical accepted graph and complete update |
:::



## 8. Explicit Constants and Derived Bounds

### 8.1. Complete measurement-averaged pressure constants

:::{div} feynman-prose
We first compute constants for the actual random measurement step. A nearby
donor can receive a far measurement while the recipient receives a near one.
Their shared normalizer preserves the difference. The diversity advantage
must exceed the reward variation between those two nearby positions. Covering
the whole feature region then turns this local comparison into an estimate
for every error-carrying label. This route works without claiming that a large
total variance separates the means of two preselected geometric groups.
:::

:::{prf:definition} Constants for the complete averaged pressure estimate
:label: def-w2-averaged-keystone-constants

Use the pipeline and entering-region hypotheses of
{prf:ref}`lem-keystone-complete-coverage-constants`. Write actual fitness as
$F_i=A_i f(u_i)$, where $A_i$ is the retained reward factor and $f$ is the fixed
positive strictly increasing diversity factor. Retain the shared empirical
normalizer and independent measurement companions. The configured diversity
exponent is positive. Put

$$
D_0=2\sqrt{R_x^2+\lambda_{\mathrm{alg}}R_v^2},\quad
B_f=\max(R_x,\sqrt{\lambda_{\mathrm{alg}}}R_v),\quad
m_x=\frac{R_x^2}{(R_x+B_x)^2}>0,
$$

for the configured squashed features, with entering alive positions
$|x|\leq B_x$ and velocities $|v|\leq V_{\max}$. The feature cube is
$[-B_f,B_f]^{2d}$. Use the matching unsquashed constants from the cited lemma
when that comparison is configured. The actual Gaussian bandwidths give

$$
\kappa_D=e^{-D_0^2/(2\sigma_D^2)},\qquad
\kappa_C=e^{-D_0^2/(2\sigma_C^2)}.
$$

Every distinct eligible donor has probability at least
$\kappa_D/(k-1)$ or $\kappa_C/(k-1)$ in its respective law. Let
$\delta_D\geq0$ be the distance floor and $\varepsilon_s>0$ the configured
diversity standardization floor. Define

$$
D_m=\sqrt{D_0^2+\delta_D^2}-\delta_D,\quad
s_*=\sqrt{D_m^2/4+\varepsilon_s^2},\quad
Z_*=D_m/\varepsilon_s.
$$

The reward-factor lower bound $A_->0$, diversity upper bound $f_+<\infty$,
and fitness upper bound $F^*<\infty$ are the actual pipeline bounds. Assume the
complete configured reward is $L_R$-Lipschitz on the entering region, uniformly
over the stated family. Let $\varepsilon_r>0$ be its standardization floor,
$H$ its fixed powered reward rescaling, and

$$
m_z=\min\left\{m_x,
\frac{\sqrt{\lambda_{\mathrm{alg}}}R_v^2}{(R_v+V_{\max})^2}\right\},\qquad
Z_R=\operatorname{osc}(R)/\varepsilon_r,\qquad
L_H=\max_{|t|\leq Z_R}|H'(t)|.
$$

Shared reward centering and the regularized scale give

$$
|A_i-A_j|\leq L_A|z_i-z_j|,\qquad
L_A=\frac{L_HL_R}{\varepsilon_r m_z}.
$$

The reward bound includes every term used by the algorithm. A boundary penalty
requires this same uniform regularity on the declared entering region; a bound
on the objective alone does not suffice.

Choose $E_{\max}>0$ bounding $\|\Delta\delta_{x,i}\|^2$; if both entering
position domains have diameter at most $D_x$, $E_{\max}=4D_x^2$ suffices.
Fix an analysis threshold $0<W_0\leq E_{\max}$ with
$m_x^2W_0/4<D_0^2$, and set

$$
v_0=m_x^2W_0/4,\quad h_f=\sqrt{v_0/2},\quad
\rho_f=\frac{v_0/2}{D_0^2-v_0/2}>0,
$$

$$
\Delta_f=\sqrt{h_f^2+\delta_D^2}-\sqrt{h_f^2/4+\delta_D^2},\quad
t_f=\Delta_f/s_*,\quad
\omega_f=\min_{u\in[-Z_*,Z_*-t_f]}[f(u+t_f)-f(u)]>0.
$$

Strict increase on this fixed compact interval proves positivity for the
configured map; it requires no gain limit. Put

$$
r=\begin{cases}
\min\{h_f/2,A_-\omega_f/(2f_+L_A)\},&L_A>0,\\
h_f/2,&L_A=0,
\end{cases}\qquad
\gamma_0=A_-\omega_f/2,
$$

$$
a_0=\min\left\{1,\frac{\gamma_0}
{p_{\max}(F^*+\varepsilon_{\mathrm{clone}})}\right\},\qquad
C_0=\kappa_C\kappa_D^2\rho_f a_0>0,
$$

$$
M_r=\left\lceil\frac{2B_f\sqrt{2d}}r\right\rceil^{2d},\qquad
\chi_*=\frac{C_0W_0^2}{2E_{\max}^2M_r^2}>0.
$$

These constants depend on the fixed configuration, complete reward,
entering region and analysis threshold. They are independent of $N$.
:::

:::{prf:proposition} All-population averaged pressure bound
:label: prop-w2-averaged-keystone-constants

Under {prf:ref}`def-w2-averaged-keystone-constants`, for the all-alive quantities
$W,Q$ of {prf:ref}`lem-quantitative-keystone-w2`,

$$
\mathbb E Q\geq\chi_*(W-W_0)-\frac{C_0E_{\max}}{N^2}.
$$

For the same fixed comparison plan and every $\eta>0$, this gives

$$
\mathbb E Q\geq\frac{\chi_*}{1+\eta}V_{\mathrm{struct}}
-\chi_*\left[W_0+\frac{(\lambda_v+b^2/(4\eta))D_v}{1+\eta}\right]
-\frac{C_0E_{\max}}{N^2}.
$$

In particular, $D_v$ can be replaced by $4V_{\max}^2$ when each entering
velocity has norm at most $V_{\max}$. Retaining its actual value gives the
sharper estimate.

*Proof.* If $W\geq W_0$, select the swarm with the larger positional variance.
Since $W\leq2\sum_s\operatorname{Var}_x(S_s)$, it has positional variance at
least $W/4$ and feature variance at least $m_x^2W/4\geq v_0$.
{prf:ref}`lem-keystone-near-neighbor-pressure` gives its averaged row pressure
$\overline p_i\geq C_0[n_i(r)/(N-1)]^2$.
{prf:ref}`thm-keystone-complete-error-coverage` sums this estimate over every
geometric cluster and every comparison label. For $N\geq2$ it gives

$$
\mathbb E Q\geq C_0W
\left[\frac{(NW/(E_{\max}M_r)-1)_+}{N-1}\right]^2
\geq C_0W\left(\frac W{E_{\max}M_r}-\frac1N\right)_+^2.
$$

Using $(a-b)_+^2\geq a^2/2-b^2$, $W\geq W_0$ and $W\leq E_{\max}$,

$$
\mathbb E Q\geq\frac{C_0W^3}{2E_{\max}^2M_r^2}
-\frac{C_0W}{N^2}
\geq\chi_*W-\frac{C_0E_{\max}}{N^2}.
$$

For $W<W_0$, nonnegativity proves the affine bound. At $N=1$, centered
positional discrepancies vanish and the same low-error branch applies.
The matching/Young estimate proved in
{prf:ref}`lem-quantitative-keystone-w2` gives the structural version.
Finally, centering is an orthogonal projection on the fixed all-alive matching,
so $D_v\leq N^{-1}\sum_i|v_{1,i}-v_{2,i}|^2\leq4V_{\max}^2$.
$\square$
:::

### 8.2. Optional realized-target pressure constants

:::{div} feynman-prose
For a particular realized fitness vector we can also use a target population.
That route needs a signed gap between the actual geometric groups. Total
variance can certify how many values lie below their overall mean, but it
cannot say which geometric group contains them. The signed gap and the error
carried by the selected labels must both be established. A complete tie makes
the positive comparison hypothesis unavailable; its zero selection still
belongs to the averaged estimate above.
:::

:::{prf:proposition} Conditional target pressure with all margins stated
:label: prop-w2-realized-target-keystone

Fix a family of all-alive entering comparisons and complete realized fitness
vectors with $N\geq2$ and $v_*\leq V_i\leq v^*$, $0<v_*<v^*$. Let
$R_*=v^*-v_*$. Above a declared structural threshold $R_{\mathrm{spread}}^2$,
suppose the following bounds hold uniformly at this same conditional stage.

1. In a chosen swarm, the actual high-error and complementary geometric sets
   $H,L$ have fractions $f_H\geq\underline f_H>0$ and
   $f_L\geq\underline f_L>0$. Their actual arithmetic fitness means obey
   $\mu_{V,L}-\mu_{V,H}\geq\Delta_V>0$.
   A sufficient verification, from
   {prf:ref}`thm-stability-condition-final-corrected`, is

   $$
   \delta_{\log}=\beta D_*-\alpha A_*,\qquad
   \delta=\delta_{\log}-B_H/(2v_*^2)>0,\qquad
   \Delta_V=v_*(e^\delta-1),
   $$

   where $D_*\leq\mathbb E_L\log d'-\mathbb E_H\log d'$,
   $A_*\geq\mathbb E_H\log r'-\mathbb E_L\log r'$, and
   $\operatorname{Var}_H(V)\leq B_H$ refer to those same realized empirical
   populations. The bounds in {prf:ref}`prop-corrective-signal-bound` and
   {prf:ref}`prop-log-reward-gap-axiom-bound` may verify $D_*,A_*$ when their
   own hypotheses hold. An ordered-coupling bound requires its actual
   ordering hypothesis. If a total-to-group variance argument is used,
   {prf:ref}`lem-variance-to-mean-separation` requires a separately proved within-group
   variance bound and the corrective orientation.

2. The actual donor law satisfies $K_i(j)\geq a_*/(N-1)$ for $j\ne i$,
   with $a_*>0$. For Gaussian weights on a comparison domain of diameter
   $D_z$ and configured width $\sigma_C$, one may use
   $a_*=\exp[-D_z^2/(2\sigma_C^2)]$. The complete realized fitness variance
   satisfies $s_V^2\geq s_*^2>0$; the preceding mean-gap premise supplies
   the valid choice $s_*^2=\underline f_H\underline f_L\Delta_V^2$.

3. Put $T=I_{11}\cap U_k\cap H$. Let
   $S_k=\sum_i\|\delta_{x,k,i}\|^2$. For positive constants $c_H,a_x$ and
   finite nonnegative constants $b_x,M_j,B_T$, require

   $$
   \sum_{i\in H}\|\delta_{x,k,i}\|^2\geq c_HS_k,\qquad
   S_k/N\geq a_xV_{\mathrm{struct}}-b_x,
   $$

   $$
   \frac1N\sum_{i\in H}\|\delta_{x,j,i}\|^2\leq M_j,\qquad
   \frac1N\sum_{i\in H\setminus T}\|\Delta\delta_{x,i}\|^2\leq B_T.
   $$

   Here $j$ is the other swarm. The second inequality retains the actual
   velocity and hypocoercive cross-term remainder when full phase-space
   structural error is used. For a complete-linkage cluster diameter $D_c$
   and positional variance threshold $R_{\mathrm{var}}^2>D_c^2/2$,
   {prf:ref}`lem-variance-concentration-Hk` supplies
   $c_H=(1-\varepsilon_O)(1-D_c^2/(2R_{\mathrm{var}}^2))$ when the
   actual high-error clusters capture at least a fraction $1-\varepsilon_O$
   of the positional between-cluster variance.

Then valid constants are

$$
f_{UH}=\frac{\underline f_H\underline f_L\Delta_V}{R_*},\qquad
f_U=f_F=\frac{s_*^2}{2R_*^2},
$$

$$
B_{\mathrm{acc}}=\max\{R_*,p_{\max}(v^*+\varepsilon_{\mathrm{clone}})\},\qquad
p_u=\frac{a_*s_*^2}{2R_*B_{\mathrm{acc}}},
$$

$$
c_{\mathrm{err}}=c_Ha_x/2,\qquad
g_{\mathrm{err}}=c_Hb_x/2+M_j+B_T,\qquad
\chi=p_uc_{\mathrm{err}},
$$

$$
g_{\max}=\max\{p_ug_{\mathrm{err}},\chi R_{\mathrm{spread}}^2\}.
$$

The symbols $f_U,f_F$ here denote lower bounds on the corresponding fractions.
The clipping denominator $B_{\mathrm{acc}}$ retains both the largest fitness
difference and the largest acceptance denominator. With these constants,

$$
Q\geq\chi V_{\mathrm{struct}}-g_{\max}.
$$

*Proof.* {prf:ref}`thm-unfit-high-error-overlap-fraction` yields
$|H\cap U_k|/N\geq f_Hf_L\Delta_V/R_V\geq f_{UH}$; the factor $f_L$
comes from the difference between the global and high-error means.
The exact variance decomposition gives
$s_V^2\geq f_Hf_L\Delta_V^2$.
{prf:ref}`lem-unfit-fraction-lower-bound` gives the stated unfit and fit
fraction bounds with the squared range in the denominator.
{prf:ref}`lem-unfit-cloning-pressure` gives $p_{k,i}\geq p_u$ on $T$.
It sums the positive differences over donors before dividing by their
probability normalizer, so no uncompensated factor $1/(N-1)$ remains.
{prf:ref}`lem-error-concentration-target-set` gives
$N^{-1}\sum_{i\in T}\|\Delta\delta_{x,i}\|^2\geq
c_{\mathrm{err}}V_{\mathrm{struct}}-g_{\mathrm{err}}$.
Multiplying by $p_u$ proves the high-error bound. Below the declared threshold,
$Q\geq0\geq\chi V_{\mathrm{struct}}-g_{\max}$.
These are precisely the inputs to {prf:ref}`lem-quantitative-keystone`.
$\square$
:::

The conditional proposition can be averaged when its hypotheses hold almost
surely with the same constants. If they hold only on selected measurement
events, their probabilities and their correlation with the entering error must
remain in the estimate. A lower bound on expected log-fitness alone does not
establish the conditional premises. The complete averaged route in Section 8.1
provides an alternative that includes all measurement events explicitly.

### 8.3. Reset control and the complete-update rate

For eligible input positions of diameter at most $D_x$ and actual recipient jitter $\sigma_{\mathrm{clone}}$, the reset constant is

$$
C_{\mathrm{reset}}=D_x^2+2(1-1/N)d\sigma_{\mathrm{clone}}^2.
$$

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

### 8.4. Comparison with KL-Convergence

The KL-convergence framework ({doc}`15_kl_convergence`) may provide faster convergence rates via entropy methods. The centered/structural Wasserstein-2 control proven here is complementary:

- **Centered positional $W_2$ control (via variance proxy)**: A
  reset bound for the cloning stage
- **KL contraction**: Entropy-based, potentially faster, uses LSI theory

Each approach retains its own hypotheses. The proxy estimate controls one geometric component; it is not an independent proof of complete-law convergence.



(sec-w2-dimension-moment-certificates)=
### 8.5. Dimension and normalized-moment certificates

:::{prf:lemma} Dimension-aware reset on the actual live donor support
:label: lem-w2-dimension-aware-cloning-reset

Condition on a complete entering state and realized fitness. In one swarm of
$N$ output rows, every retained or revived positional source belongs to the
finite entering **live** support $A\subset\mathbb R^d$. Its affine dimension
is at most $r\le d$ and its diameter is $D$. Put $D=0$ when $r=0$. Write
$p_i$ for the actual probability that recipient $i$ copies and receives
independent centered jitter of covariance $\sigma_J^2 I_d$; mandatory
revivals have $p_i=1$. Jitter is independent of source selection. Then

$$
\mathbb E\operatorname{Var}_x(S')
\le \frac{r}{2(r+1)}D^2
 +(1-1/N)d\sigma_J^2\bar p,
\qquad \bar p=N^{-1}\sum_i p_i\le1.
$$

For two swarms, sum their separate geometric terms and jitter terms. In
particular, a collapsed support contributes zero geometric term. An exact
upper bound on affine dimension may replace $r$ because $r/(r+1)$ is
increasing. Isotropic jitter retains the ambient dimension $d$ even when
the source support has smaller affine dimension.

*Proof.* Work in the affine hull of $A$. Its smallest enclosing ball exists.
Its center belongs to the convex hull of its contact points: otherwise a
separating direction moves the center closer to every contact point, and
compactness of the finite support permits a strictly smaller ball. Choose a
convex representation of the center using at most $r+1$ contact points.
Such a representation follows by eliminating a coefficient along an affine
dependence whenever more than $r+1$ coefficients are positive. If the
resulting coefficients are $a_j$, the squared ball radius satisfies

$$
R_A^2=\frac12\sum_{j,k}a_ja_k|x_j-x_k|^2
\le\frac{D^2}{2}\left(1-\sum_j a_j^2\right)
\le\frac{rD^2}{2(r+1)}.
$$

For every realized copying cloud, its variance is at most its mean squared
distance from this ball center, hence at most $R_A^2$. This step does not
assume independent donor choices. Centered independent recipient jitter
adds exactly $(1-1/N)d\sigma_J^2\bar p$ to expected empirical variance;
the copying/jitter cross term has zero expectation. Dead stored coordinates
are not members of $A$. $\square$
:::

:::{prf:lemma} Reset from normalized donor occupation moments
:label: lem-w2-normalized-occupation-reset

At the same conditional stage, let $R_{ij}$ be the actual probability that
output row $i$ copies positional source $j$ before jitter, including its
probability of retaining itself. Thus $R$ is stochastic and has zero columns
on dead sources. Define

$$
q_j=\frac1N\sum_iR_{ij},\qquad
m_q=\sum_jq_jx_j,\qquad
W_q=\sum_jq_j|x_j-m_q|^2.
$$

Then, with arbitrary dependence between the source choices,

$$
\mathbb E\operatorname{Var}_x(S')
\le W_q+(1-1/N)d\sigma_J^2\bar p.
$$

If recipient source choices are conditionally independent, put
$m_i=\sum_jR_{ij}x_j$ and
$s_i^2=\sum_jR_{ij}|x_j-m_i|^2$. The exact identity is

$$
\mathbb E\operatorname{Var}_x(S')
=W_q-\frac1{N^2}\sum_i s_i^2
 +(1-1/N)d\sigma_J^2\bar p.
$$

These are normalized second-moment certificates. They require no maximum
particle norm and apply to unbounded configurations with finite moments.
A state-uniform or law-averaged use requires a corresponding moment bound;
the finite-state identity does not supply confinement by itself.

*Proof.* Let $Z_i$ denote the copied position before jitter. The identity
$\operatorname{Var}(Z)=N^{-1}\sum_i|Z_i|^2-|\bar Z|^2$ gives
$\mathbb E\operatorname{Var}(Z)=W_q-
\mathbb E|\bar Z-m_q|^2\le W_q$. Independence gives
$\mathbb E|\bar Z-m_q|^2=N^{-2}\sum_i s_i^2$.
Add the same centered-jitter contribution as in
{prf:ref}`lem-w2-dimension-aware-cloning-reset`. The column law $q$ is a
probability distribution on live sources, so no dead storage term or
unnormalized factor $N$ remains. $\square$
:::

:::{prf:lemma} Curvature and reward moments for quadratic-cosine landscapes
:label: lem-w2-curvature-reward-moment

For $d\ge1$, $\kappa\ge0$, $a\in\mathbb R$, and $\omega\ge0$, let

$$
U(x)=\frac\kappa2|x|^2+a\sum_{b=1}^d(1-\cos(\omega x_b)),
\qquad F=-\nabla U.
$$

The Hessian eigenvalues belong to
$[\kappa-|a|\omega^2,\kappa+|a|\omega^2]$, and the global force
Lipschitz constant is $L_F=\kappa+|a|\omega^2$, independent of $d,N$.
For any probability coupling $\pi$ of measures with finite second moments,
write $M_{2,x}=\int|x|^2d\pi$, $M_{2,y}=\int|y|^2d\pi$, and
$C_\pi=\int|x-y|^2d\pi$. Then

$$
\left|\int U(x)d\pi-\int U(y)d\pi\right|
\le\int|U(x)-U(y)|d\pi
\le\frac{L_F}{\sqrt2}\sqrt{(M_{2,x}+M_{2,y})C_\pi}.
$$

The same bound applies to reward $R=-U$. Its dimension dependence is in the
actual normalized moments, rather than an added $\sqrt d$ maximum-coordinate
factor. If a genuine box $[-L,L]^d$ is part of the kernel, its exact reward
Lipschitz constant is $\sqrt d\max_{|t|\le L}|\kappa t+a\omega\sin(\omega t)|$.
Evaluate the endpoints and the stationary points satisfying
$\cos(\omega t)=-\kappa/(a\omega^2)$ when that equation is admissible.
An alternative bound separates confinement from the bounded periodic term.
For $C_b=\int|x_b-y_b|^2d\pi$ it is

$$
\left|\int U(x)d\pi-\int U(y)d\pi\right|
\le\frac\kappa{\sqrt2}\sqrt{(M_{2,x}+M_{2,y})\sum_bC_b}
 +|a|\sum_b\min(2,\omega\sqrt{C_b}).
$$

It does not charge dimensions in which the coupling has no displacement.

*Proof.* The Hessian is diagonal with entries
$\kappa+a\omega^2\cos(\omega x_b)$, proving the global curvature bound.
Since $\nabla U(0)=0$, $|\nabla U(z)|\le L_F|z|$. Integration along the
segment from $y$ to $x$ yields
$|U(x)-U(y)|\le(L_F/2)|x-y|(|x|+|y|)$.
Cauchy--Schwarz and $(|x|+|y|)^2\le2(|x|^2+|y|^2)$ prove the moment
estimate. Separability gives the stated exact box constant.
For the separated estimate, apply the quadratic moment argument to
$\kappa|x|^2/2$ and use both
$|\cos s-\cos t|\le2$ and $|\cos s-\cos t|\le|s-t|$ in each coordinate,
followed by Cauchy--Schwarz under the probability coupling. $\square$
:::

:::{prf:lemma} Dense Gaussian viscosity from donor covariance moments
:label: lem-w2-viscous-moment-derivative

Fix $n$ live rows and frozen velocities. For row-normalized viscosity let
$P_i$ be the actual excluded-self Gaussian donor law of width $\rho>0$.
Denote its position and velocity means by $\bar x_i,\bar v_i$, and put

$$
s_{x,i}^2=\sum_jP_i(j)|x_j-\bar x_i|^2,\quad
s_{v,i}^2=\sum_jP_i(j)|v_j-\bar v_i|^2,\quad
r_i^2=\sum_jP_i(j)|x_i-x_j|^2.
$$

For a perturbation $h$ with maximum row norm $\|h\|_\infty$, the position
derivative of the actual viscous force satisfies

$$
|(D_xF^{\rm visc}h)_i|
\le\frac\nu{\rho^2}s_{v,i}(s_{x,i}+r_i)\|h\|_\infty.
$$

For $n\le2$, its row-normalized position derivative is zero. If, on the
same genuine analysis event, $|x_j|\le R$ and $|v_j|\le V_c$ for every
row, an $n$-independent majorant for $n>2$ is

$$
C_{x,\infty}^{\rm row}
=\frac{3\sqrt3}{2}\frac{\nu V_cR}{\rho^2}.
$$

For count normalization, in the population Hilbert norm
$\|h\|_{2,n}^2=n^{-1}\sum_i|h_i|^2$, a valid global position derivative
majorant under $|v_i|\le V_c$ is

$$
C_{x,2}^{\rm count}=2\nu V_c e^{-1/2}/\rho.
$$

This last constant is not a maximum-row bound. The existing count
maximum-row constant $4\nu V_c e^{-1/2}/\rho$ retains its separate norm.

*Proof.* Differentiation of the row ratio gives
$D_xF_i^{\rm visc}h=\nu\sum_jP_i(j)(v_j-\bar v_i)
D\log K_{ij}h$. Expand
$D\log K_{ij}h=-\rho^{-2}\langle x_i-x_j,h_i-h_j\rangle$.
The common term $\langle x_i,h_i\rangle$ cancels. The remaining $h_i$
term is bounded by $s_{v,i}s_{x,i}\|h\|_\infty$, and the $h_j$ term by
$s_{v,i}r_i\|h\|_\infty$, using weighted Cauchy--Schwarz in both cases.
If $u=|\bar x_i|/R\in[0,1]$, then
$s_{x,i}\le R\sqrt{1-u^2}$,
$r_i\le R\sqrt{2+2u}$, and $s_{v,i}\le V_c$.
The maximum of $\sqrt{1-u^2}+\sqrt{2+2u}$ is $3\sqrt3/2$, attained
at $u=1/2$. A pair has only one donor, so its normalized ratio has zero
derivative.

For count normalization, symmetry gives for any vectors $g,h$ the unordered
edge identity for $\langle g,D_xF^{\rm visc}h\rangle$. Each Gaussian
weight derivative is bounded by $e^{-1/2}|h_i-h_j|/\rho$, and
$|v_j-v_i|\le2V_c$. Cauchy--Schwarz on unordered edges and
$\sum_{i<j}|h_i-h_j|^2=n\sum_i|h_i-\bar h|^2$ prove the Hilbert
constant. $\square$
:::

:::{prf:corollary} Sharper first-drift margins in their declared norms
:label: cor-w2-sharp-first-drift-margin

On the same prepared-position event of
{prf:ref}`def-w2-finite-population-qsd-regime`, take
$R=\sqrt d L_D+J$. The first drift $A_v-I$ has Lipschitz majorant

$$
\beta_{\rm row}^{\sharp}
=t^2\left(L_F+\frac{3\sqrt3}{2}\frac{\nu V_c(\sqrt d L_D+J)}{\rho^2}\right)
$$

in maximum-row norm, and

$$
\beta_{\rm count,2}^{\sharp}
=t^2(L_F+2\nu V_c e^{-1/2}/\rho)
$$

in the population Hilbert norm. Whenever the respective constant is below
one, the first drift is injective on the same convex domain and
$|\det DA_v|\ge(1-\beta^{\sharp})^{nd}$.
At the unchanged $d=3,N=200$ reference, the existing analytic bounds
$J<5.71$, $\sqrt3L_D<3.465$ give
$\beta_{\rm row}^{\sharp}<0.012$ and
$\beta_{\rm count,2}^{\sharp}<0.001$.

*Proof.* Apply {prf:ref}`lem-w2-viscous-moment-derivative` and the global
force derivative bound to $A_v=x+tv+t^2(F+F^{\rm visc})$.
A Lipschitz perturbation of identity below one is injective. Every eigenvalue
of its Jacobian perturbation has modulus at most that operator majorant, so
the determinant bound follows. The numerical strict bounds follow by direct
substitution of $t=0.02$, $\nu=0.3$, $V_c=4$, $\rho=L_F=1$ and the
displayed event radius estimates. $\square$

The row moment certificate itself does not require a position ball. The
uniform row corollary uses the existing Gaussian analysis event, not bounded
algorithmic noise. Its $J$ retains the original finite-population event
dependence. A population-uniform unbounded-space rate requires the additional
uniform conditional moment/operator hypotheses and confinement argument;
an observed second moment alone does not establish them. Downstream density
or spectral constants must be recomputed when a sharper margin is used.
:::

:::{prf:proposition} Regional structural profiles for quadratic-cosine forces
:label: prop-w2-regional-cosine-profile

Use the potential of {prf:ref}`lem-w2-curvature-reward-moment` and write
$F(x)=-\kappa x+e(x)$. Then

$$
M_d:=|a|\omega\sqrt d,\qquad L_e:=|a|\omega^2,
\qquad |e(x)|\le M_d,\qquad |U(x)|\le\frac\kappa2|x|^2+2|a|d.
$$

Let $A=\prod_b[\ell_b,u_b]$ be a declared convex analysis box, without
restricting the algorithm to that box. Exact cosine extrema on each interval
$[\omega\ell_b,\omega u_b]$ give a Hessian enclosure
$m_A I\preceq\nabla^2U\preceq M_A^{\rm Hess}I$. Put
$L_A=\max(|m_A|,|M_A^{\rm Hess}|)$,
$R_A^2=\sum_b\max(|\ell_b|,|u_b|)^2$ and
$\widehat M_A=\kappa R_A+M_d$. Let $L_{e,A}$ be the analogous maximum
absolute periodic Hessian entry. The profiles of
{prf:ref}`def-slc-profiles` admit the upper certificates

$$
\begin{aligned}
M_A&\le\widehat M_A,\qquad
\omega_A(r)\le\widehat\omega_A(r):=\min(L_A r,2\widehat M_A),\\
\omega_{e,A}(r)&\le\min(L_{e,A}r,2M_d),\qquad
D_A(L,r)\le[\widehat\omega_A(r)-Lr]_+,\\
J_A(k,r)&\le\min\left\{
[k-m_A]_+r^2,\ [k-\kappa+L_{e,A}]_+r^2,
\max_{0\le t\le r}[(k-\kappa)t^2+2M_dt]_+
\right\}.
\end{aligned}
$$

For $k<\kappa$, the radial certificate is valid on the entire unbounded
space:

$$
b_{\mathbb R^d}(k)\le\frac{M_d^2}{4(\kappa-k)}.
$$

If $M_d=0$ and $k\le\kappa$ it is zero. No finite global radial certificate
is inferred from this formula when $k\ge\kappa$ and $M_d>0$. A bounded
regional certificate is

$$
b_A(k)\le\sum_b\max_{0\le t\le\max(|\ell_b|,|u_b|)}
[(k-\kappa)t^2+|a|\omega t]_+.
$$

All constants are independent of population size. A positive $m_A$ certifies
pairwise restoration inside $A$; a nonpositive enclosure identifies a region
where this certificate supplies no strictly restoring coefficient. These
profiles do not supply residence, transition or donor-pressure probabilities.
The exterior and all Gaussian excursions retain their separate tail charges.

*Proof.* The periodic force is coordinatewise
$e_b=-a\omega\sin(\omega x_b)$, giving its amplitude and derivative bounds.
The Hessian entries are $\kappa+a\omega^2\cos(\omega x_b)$. Integrate this
Hessian along segments in the convex box to obtain the force and pairwise
moduli. The bounded-perturbation pairwise estimate follows from
$\langle x-y,e(x)-e(y)\rangle\le2M_d|x-y|$ and maximization over
$|x-y|\le r$; its alternative Lipschitz bound uses $L_{e,A}$. For the radial
profile,
$x\cdot F(x)+k|x|^2\le-(\kappa-k)|x|^2+M_d|x|$.
Completing the square gives the global constant. Coordinatewise maximization
gives the regional alternative. Finally $|1-\cos|\le2$ gives the reward
growth envelope. These estimates apply to defining suprema, not to finite
sample maxima. $\square$
:::

:::{prf:lemma} Propagated normalized moments with bounded nonquadratic force
:label: lem-w2-propagated-moment-budget

Let a uniformly chosen output source before jitter have
$\mathbb E|X_0|^2\le H_2$, and let its independent Gaussian clone jitter be
$I\sigma_JZ$ with $\mathbb EI=\bar p$. Assume the post-collision velocity
bound $V_c$, a dense nonnegative first viscous kick with $0\le t\nu\le1$,
and $F=-\kappa x+e$ with $|e|\le M_d$. Put
$t=h/2$, $c=e^{-\gamma h}$, $b=t(1+c)$ and
$a_x=1-\kappa t^2(1+c)$. With full standard Gaussian OU innovation of
amplitude $q$ and final independent position noise of amplitude $s$, define

$$
\begin{aligned}
B_Y&=|a_x|\sqrt{H_2}+bV_c+btM_d,&
\Sigma_Y&=a_x^2\sigma_J^2\bar p+t^2q^2,\\
B_W&=ct\kappa\sqrt{H_2}+cV_c+ctM_d,&
\Sigma_W&=c^2t^2\kappa^2\sigma_J^2\bar p+q^2.
\end{aligned}
$$

The prepared, OU-stage and final normalized moments satisfy

$$
\begin{aligned}
\mathbb E|X|^2&\le H_2+d\sigma_J^2\bar p,&
\mathbb E|Y|^2&\le(B_Y+\sqrt{d\Sigma_Y})^2,\\
\mathbb E|W|^2&\le(B_W+\sqrt{d\Sigma_W})^2,&
\mathbb E|x'|^2&\le(B_Y+\sqrt{d\Sigma_Y})^2+ds^2.
\end{aligned}
$$

The cap adds the deterministic final velocity bound $V_{\max}^2$ to the
last positional bound. These statements concern uniformly chosen output
slots before conditioning on final survival. Conditioning on alive output
requires a separately proved positive alive-mass bound. A uniform-in-time
use requires a propagated $H_2$ from confinement or Safe Harbor; a measured
finite-state occupation moment is only a conditional input certificate.

*Proof.* The first dense viscous kick is a convex combination of bounded
input velocities, also for count normalization whose row mass is at most
one. Denote that bounded result by $U$. Exact BAOAB assembly gives
$Y=a_xX_0+bU+bt e(X)+a_xI\sigma_JZ+tq\xi$ and
$W=-ct\kappa X_0+cU+ct e(X)-ct\kappa I\sigma_JZ+q\xi$.
Minkowski in normalized $L^2$ bounds their non-Gaussian terms by $B_Y,B_W$.
The centered Gaussian terms have the displayed second moments, even though
$e(X)$ and $U$ depend on the jitter. Apply Minkowski once more. The final
independent centered position innovation adds exactly $ds^2$; the velocity
cap bounds every final row. No maximum particle norm or truncated noise is
used. $\square$
:::

(sec-w2-finite-population-qsd)=
## 9. Completed Finite-Population Convergence in Fitness-Tie Regions

:::{div} feynman-prose
A fitness tie switches off competitive copying for that realization. It does
not switch off the independent kinetic noises. For the existing terminal-box
gas, those noises give every starting swarm a common part of its transition
law. The finite-population theorem below turns that common part into loss of
memory, including states with exactly equal fitness.

There are two normalizations to keep straight. We first condition the entire
run on the swarm still having an alive row. Inside each surviving swarm we
then choose one of its alive rows uniformly. This gives a probability law on
positions and velocities inside the configured box and velocity ball. Its
Wasserstein convergence follows from the full swarm's conditioned convergence.
The rate depends on population size and can be extremely small; the theorem
supplies a positive certificate, rather than a practical mixing-time estimate.
:::

:::{prf:definition} Primitive regime for the existing terminal-box kernel
:label: def-w2-finite-population-qsd-regime

Use the real-coordinate dense Viscous Euclidean Gas of
{prf:ref}`def-cgd-parameter-register`, with terminal box
$D=[-L_D,L_D]^d$, $L_D>0$, population $N\geq1$, dimension $d\geq1$, and
smooth cap $C_V(v)=Vv/(V+|v|)$, $V>0$. The consumed stages must agree with
that restriction of {prf:ref}`def-native-complete-execution-record`:
current independent companion draws, canonical regularized sampled fitness
and clipped gates, immutable-source copying and mandatory revival, independent
recipient jitter, one shared Haar rotation per accepted component, both dense
viscous BAOAB kicks, independent OU and final position innovations, final cap,
and terminal-only marking. Retain its positive widths and fitness floors.
Historical donors, elite retention, graph or geometry feedback, curl rotation,
adaptive diffusion, periodic wrapping and extra kinetic stages belong to other
kernels. The QSD state consists of the retained physical positions, velocities
and status marks. Passive readouts may be retained as their pushforwards; growing
trajectory-history clocks are outside this stationary state.

This kernel kills the swarm at $k=0$ alive rows and retains its declared
singleton convention. It is different from the $k<2$ cemetery convention of
{prf:ref}`def-cemetery-state`. All entering velocities, including those of dead
rows, are capped. Retained dead positions remain physical unbounded coordinates.

For the configured force assume the actual profile of
{prf:ref}`def-cgd-analytic-force-profile`: $F$ is globally real analytic,
$B_F=|F(0)|<\infty$, and $L_F=\sup_{x\in\mathbb R^d}\|DF(x)\|<\infty$.
The complete configured reward $R$ must be finite and continuous on the
closed alive region $\overline D\times\overline B_V$. This supplies the
bounded, continuous sampled-fitness map needed by the compact input-kernel
argument. The reference reward $R=-U$ satisfies this condition.
Let $h>0$, $\gamma\geq0$, and put

$$
t=h/2,\qquad c=e^{-\gamma h},\qquad
q^2=b_O^2\begin{cases}(1-e^{-2\gamma h})/(2\gamma),&\gamma>0,\\
h,&\gamma=0,
\end{cases}\qquad s=\sigma_{\mathrm{pos}}\sqrt h,
$$

with $q,s>0$. Here $\sigma_{\mathrm{pos}}$ denotes Chapter 18's final
position diffusion amplitude, separate from clone jitter $\sigma_J$.
Require $\sigma_J>0$ when $N>1$. Define

$$
\nu_{\mathrm{visc},N}=\begin{cases}\nu,&N\geq2,\\0,&N=1,\end{cases}
\qquad \nu\geq0,\quad \rho>0,\quad
V_c=(1+2|\alpha_{\mathrm{col}}|)V,\quad R_D=\sqrt d L_D,
$$

and $\chi_{\mathrm{norm}}=1$ for count normalization or $2$ for row
normalization. The B2 coercivity condition is

$$
\kappa_F=1-t^2L_F-\chi_{\mathrm{norm}}t\nu_{\mathrm{visc},N}>0.
$$

Choose $J_0=\sigma_J$ when $\sigma_J>0$, and any $J_0>0$ otherwise. Write
$p_0=G_d(J_0/\sigma_J)$ for positive jitter and $p_0=1$ otherwise, where
$G_d$ is standard Gaussian ball probability. The tagged-row survival floor is

$$
B_1(J_0)=(1+2t\nu_{\mathrm{visc},N})V_c
+t[B_F+L_F(R_D+J_0)],\qquad
A=L_D+J_0+t(1+c)B_1(J_0),
$$

$$
\sigma_h=\sqrt{t^2q^2+s^2},\qquad
a_F=p_0\left[
\Phi\left(\frac{L_D-A}{\sigma_h}\right)
-\Phi\left(\frac{-L_D-A}{\sigma_h}\right)\right]^d>0.
$$

All Gaussian draws remain unbounded. For the proof events choose

$$
r_*=\sqrt{2d[\log(12dN)-2\log a_F]},\qquad G=r_*,\qquad
J=\begin{cases}\sigma_Jr_*,&\sigma_J>0,\\J_0,&\sigma_J=0,\ N=1.
\end{cases}
$$

These choices give the discarded-event bound $p_{\mathrm{bad}}\leq a_F^2/2$
in {prf:ref}`cor-cgd-primitive-reference`. In addition require

$$
\beta_F=t^2(L_F+C_x)<1,\qquad
C_x=\begin{cases}
4\nu_{\mathrm{visc},N}V_c e^{-1/2}/\rho,&\mathrm{count},\\
16\nu_{\mathrm{visc},N}V_c(R_D+J)/\rho^2,&\mathrm{row},\ N>2,\\
0,&\mathrm{row},\ N\leq2.
\end{cases}
$$

Compute the one-update lower density $\mathfrak l_N(R,H)$, target
minorization $\epsilon_F$, two-update upper density $M_N$, and raw output
radii $R,H$ from {prf:ref}`def-cgd-analytic-force-profile` and
{prf:ref}`lem-cgd-two-update-density`, using these same primitive parameters
and radii. These are their explicit Gaussian density, determinant and finite
pattern-count formulas. Set

$$
\underline m_F=\min\left\{1,
\frac{\mathfrak l_N(R,H)a_F^2}{2M_N}\right\}>0,\qquad
\underline\delta_F=\epsilon_F\underline m_F\in(0,1),\qquad
\rho_F=1-\underline\delta_F\in(0,1).
$$

These are evaluated primitive bounds, rather than unknown spectral inputs.
Their comparison to the actual eigenfunction is proved in
{prf:ref}`thm-cgd-primitive-eigenfunction`. Every bound is for the fixed
finite population and the declared real-coordinate kernel.
:::

:::{prf:lemma} Total variation under bounded positive reweighting
:label: lem-w2-bounded-reweighting-tv

Use $\|\mu-\zeta\|_{\mathrm{TV}}=\sup_A|\mu(A)-\zeta(A)|$ for
probability measures. If $0<a\leq w\leq b<\infty$ and
$\mathcal R_w\mu=w\mu/\mu(w)$, then

$$
\|\mathcal R_w\mu-\mathcal R_w\zeta\|_{\mathrm{TV}}
\leq\frac ba\|\mu-\zeta\|_{\mathrm{TV}}.
$$

*Proof.* For a measurable set $A$, put $p=(\mathcal R_w\zeta)(A)$ and
$g=w(\mathbf1_A-p)$. Then $\zeta g=0$ and
$-pb\leq g\leq(1-p)b$, so its oscillation is at most $b$. Since
$\mu-\zeta$ has zero total mass,
$|(\mu-\zeta)g|\leq b\|\mu-\zeta\|_{\mathrm{TV}}$.
Divide by $\mu(w)\geq a$ to bound the difference of the two reweighted
probabilities of $A$, and take the supremum. $\square$
:::

:::{prf:theorem} Conditioned finite-swarm convergence and alive-marginal $W_2$
:label: thm-w2-finite-n-conditioned-convergence

Under {prf:ref}`def-w2-finite-population-qsd-regime`, the actual killed
kernel $Q$ has a unique full marked QSD $\nu_Q$ with
$\nu_QQ=\alpha_Q\nu_Q$, $0<\alpha_Q<1$. Every finite-time conditional law

$$
\Phi_n(\mu)=\frac{\mu Q^n}{\mu Q^n1}
$$

is defined for every initial probability on nonextinct terminally consistent
capped states. At every integer $n\geq0$,

$$
\|\Phi_n(\mu)-\nu_Q\|_{\mathrm{TV}}
\leq\min\{1,\underline m_F^{-1}\rho_F^n\},
$$

and the initial-sensitive estimate is

$$
\|\Phi_n(\mu)-\nu_Q\|_{\mathrm{TV}}
\leq\min\{1,\underline m_F^{-2}\rho_F^n
\|\mu-\nu_Q\|_{\mathrm{TV}}\}.
$$

For a nonextinct state define the alive-sampling probability kernel

$$
K_{\mathrm{alive}}(S,dz)=\frac1{k(S)}
\sum_{i:a_i=1}\delta_{(x_i,v_i)}(dz),\qquad
\lambda_n=\Phi_n(\mu)K_{\mathrm{alive}},\quad
\lambda_Q=\nu_QK_{\mathrm{alive}}.
$$

Its normalization is performed within each surviving swarm. With Euclidean
phase-space transport cost, set $D_{\mathrm{phase}}^2=4dL_D^2+4V^2$.
Then

$$
W_2^2(\lambda_n,\lambda_Q)
\leq D_{\mathrm{phase}}^2
\min\{1,\underline m_F^{-1}\rho_F^n\}.
$$

The same bound with $\underline m_F^{-2}\rho_F^n
\|\mu-\nu_Q\|_{\mathrm{TV}}$ holds. For positional marginals replace
$D_{\mathrm{phase}}^2$ by $4dL_D^2$. At physical times $T=nh$, the
$W_2^2$ exponential rate is $-\log\rho_F/h$, and the $W_2$ rate is
$-\log\rho_F/(2h)$. These rates need not be uniform in $N$.

*Proof.*

**1. Obtain the full-kernel spectral bounds from primitive estimates.**
{prf:ref}`thm-cgd-analytic-force-qsd` constructs the QSD and the positive
continuous eigenfunction $e$, with $Qe=\alpha_Qe$ and $\max e=1$, for
this exact kernel. {prf:ref}`thm-cgd-primitive-eigenfunction` proves
$e\geq\underline m_F$ by combining the one-update lower density and
two-update good-event upper density. Its tagged survival estimate gives
$Q1\geq a_F$, so $\mu Q^n1\geq a_F^n>0$.

**2. Keep the initial tilt and final normalization.** Define the actual
Doob kernel and its invariant law by

$$
P_e(S,dS')=\frac{Q(S,dS')e(S')}{\alpha_Qe(S)},\qquad
\pi_e=\mathcal R_e\nu_Q.
$$

Its common part has weight
$\epsilon_F\theta_0(e)/\alpha_Q\geq\epsilon_F\underline m_F
=\underline\delta_F$, since $\alpha_Q\leq1$. Splitting that common
part gives
$\|\eta P_e^n-\pi_e\|_{\mathrm{TV}}\leq
\rho_F^n\|\eta-\pi_e\|_{\mathrm{TV}}$.
The exact path identity is

$$
\mu Q^nf=\alpha_Q^n\mu(e)
\big[(\mathcal R_e\mu)P_e^n\big](f/e),
$$

and using the same identity for $f=1$ yields

$$
\Phi_n(\mu)=\mathcal R_{1/e}
\big[(\mathcal R_e\mu)P_e^n\big],\qquad
\nu_Q=\mathcal R_{1/e}\pi_e.
$$

Apply {prf:ref}`lem-w2-bounded-reweighting-tv` to $1/e\in
[1,\underline m_F^{-1}]$. The distance between the two initial tilted
probabilities is at most one, proving the uniform
$\underline m_F^{-1}\rho_F^n$ bound. Applying the same lemma to
$e\in[\underline m_F,1]$ additionally proves the initial-sensitive bound.
A probability TV distance is at most one, which gives both minima.

The use of two updates in Step 1 bounds $e$; the Doob common part in Step 2
belongs to one full update. Consequently the exponent is $n$, without
$\lfloor n/2\rfloor$.

**3. Pass to the alive-sampled marginal and transport cost.** Markov-kernel
data processing gives
$\|\lambda_n-\lambda_Q\|_{\mathrm{TV}}\leq
\|\Phi_n(\mu)-\nu_Q\|_{\mathrm{TV}}$.
Both marginal laws are supported on $D\times\overline B_V$. Their common
part has mass $1-\|\lambda_n-\lambda_Q\|_{\mathrm{TV}}$; couple that
part identically. Couple the remaining equal masses by their product
probabilities. The remaining squared displacement is at most the domain's
squared diameter $4dL_D^2+4V^2$. This constructs an admissible plan of cost
at most $D_{\mathrm{phase}}^2\|\lambda_n-\lambda_Q\|_{\mathrm{TV}}$.
The positional argument is identical. Finally,
$\rho_F^n=\exp[(\log\rho_F)T/h]$; take the square root for the $W_2$
rate. $\square$
:::

:::{prf:corollary} Nonempty primitive regime at the unchanged reference
:label: cor-w2-reference-fitness-degeneracy

The complete reference instance of {prf:ref}`def-cgd-existing-reference`
satisfies the preceding theorem for both its count and its separately declared
row normalization. In particular,

$$
\begin{gathered}
d=3,\quad N=200,\quad h=0.04,\quad\gamma=1,\quad b_O=1,\quad
\sigma_{\mathrm{pos}}=\sigma_J=0.1,\quad V=2,\quad
\alpha_{\mathrm{col}}=0.5,\\
\nu=0.3,\quad\rho=1,\quad F(x)=-x,\quad R(x,v)=-|x|^2/2,
\quad D=[-2,2]^3.
\end{gathered}
$$

The donor feature radii are $2$, $\lambda_{\mathrm{alg}}=1$, both donor
widths are $2$, distance floor is $10^{-3}$, reward/diversity logistic
amplitudes are $2$, their floors and standardization regularizers are $0.1$,
both exponents are $1$, the gate scale is $1$, and its denominator floor is
$10^{-6}$. All other consumed fields retain that existing reference record.
No fitness-separation condition is required.

*Proof.* The actual force is analytic with $L_F=1$, $B_F=0$. The exact B2
margins are

$$
\kappa_{F,\mathrm{count}}=0.9936>0.99,\qquad
\kappa_{F,\mathrm{row}}=0.9876>0.98.
$$

For the reference survival radius $J_0=0.1$, elementary exponential bounds
$e^{-0.04}<0.961$ and $1-e^{-0.08}>0.0768$ give
$B_1(J_0)<4.1193$, $A<2.262$, and $0.02038<\sigma_h<0.021$.
For the coordinate mean $A$, integrate the Gaussian density on
$[L_D-\sigma_h/13,L_D]$. Its standardized interval has length $1/13$,
and its furthest endpoint has magnitude at most

$$
\frac{A-L_D}{\sigma_h}+\frac1{13}
<\frac{0.262}{0.02038}+\frac1{13}<13.
$$

The coordinate survival probability is therefore greater than
$e^{-169/2}/(13\sqrt{2\pi})>e^{-88}$. The same interval is used at the
opposite endpoint for mean $-A$; the full interval probability is minimized
at these extreme means. Bounding the Gaussian density below on the unit ball gives
$p_0=G_3(1)\ge(4\pi/3)(2\pi)^{-3/2}e^{-1/2}>0.1$.
Thus $-\log a_F<3(88)+\log10<267$, and

$$
r_*^2<6(9+2\cdot267)=3258<57.1^2,\qquad J<5.71,
$$

using $\log(12dN)=\log7200<9$. These Gaussian-event radii satisfy the
proved discarded-mass condition. Hence, with $V_c=4$ and $t=0.02$,

$$
\beta_{F,\mathrm{count}}
=0.0004[1+4(0.3)(4)e^{-1/2}]<0.002,
$$

$$
\beta_{F,\mathrm{row}}
<0.0004[1+16(0.3)(4)(3.465+5.71)]<0.071<1.
$$

These inequalities prove closure without relying on rounded evaluator output.
The corresponding primitive rate is obtained by substituting the same record
and radii in {prf:ref}`def-w2-finite-population-qsd-regime`. $\square$
:::

:::{prf:corollary} A verified force-profile band containing a nonconvex landscape
:label: cor-w2-force-profile-band

Keep all scalar parameters and kernel conventions of
{prf:ref}`cor-w2-reference-fitness-degeneracy`, and use the actual globally
real-analytic force with $B_F=0$ and $L_F\le400$. Require the complete
configured reward to satisfy the continuity hypothesis in
{prf:ref}`def-w2-finite-population-qsd-regime`. Then
{prf:ref}`thm-w2-finite-n-conditioned-convergence` holds for either existing
viscous normalization. Valid bounds are

$$
\kappa_{F,\mathrm{count}}\ge0.834,\qquad
\kappa_{F,\mathrm{row}}\ge0.828,\qquad
\beta_{F,\mathrm{count}}<0.162,\qquad
\beta_{F,\mathrm{row}}<0.41.
$$

For example the configured standard Rastrigin potential
$U(x)=\sum_{\ell=1}^3[x_\ell^2+10(1-\cos(2\pi x_\ell))]$ has
$F=-\nabla U$, $B_F=0$, and $L_F\le2+40\pi^2<400$.
Its reward $R=-U$ is continuous. Thus this existing nonconvex landscape
falls within the displayed band when the remaining recorded parameters
agree. Global convexity and a unique minimum are unnecessary for this
finite-population QSD conclusion.
:::

:::{prf:proof}
The B2 inequalities follow directly from $t=0.02$, $\nu=0.3$, and
$L_F\le400$. For the same survival event $J_0=0.1$, use the bounds
$R_D<3.465$, $c<0.961$, and $\sigma_h>0.02038$ proved in the preceding
corollary. They give $B_1(J_0)<32.568$ and $A<3.378$.
Integration over $[L_D-\sigma_h/13,L_D]$ has standardized length $1/13$
and endpoints of magnitude at most

$$
\frac{A-L_D}{\sigma_h}+\frac1{13}
<\frac{1.378}{0.02038}+\frac1{13}<68.
$$

The coordinate survival probability is greater than
$e^{-68^2/2}/(13\sqrt{2\pi})>e^{-2316}$. Since $p_0>0.1$,
$-\log a_F<3(2316)+\log10<6951$. Consequently the specified radii obey
$r_*^2<6(9+2\cdot6951)=83466<289^2$ and $J<28.9$.
Therefore

$$
\beta_{F,\mathrm{count}}
<0.0004[400+4(0.3)(4)]<0.162,
$$

$$
\beta_{F,\mathrm{row}}
<0.0004[400+16(0.3)(4)(3.465+28.9)]
=0.4085632<0.41.
$$

The required primitive conditions and discarded-mass bound are verified.
For the example, differentiating the actual force gives a diagonal
Jacobian with entries $-2-40\pi^2\cos(2\pi x_\ell)$, giving the stated
global derivative bound and $F(0)=0$. $\square$
:::

:::{div} feynman-prose
This completes a particular convergence question: at fixed population size,
surviving runs forget their initial swarm and approach the full QSD, and the
alive-sampled position–velocity law converges in Wasserstein distance. Fitness
ties remain in every random measurement and gate calculation.

The alive marginal has bounded support because terminal status and the cap are
configured parts of this gas. A fixed-label marginal also retains dead positions;
those coordinates are unbounded and require their proved tail estimates before
making a transport comparison.

The full-array obstructions have concrete causes. Terminal status probabilities
can change to first order while an entering squared discrepancy changes to
second order. Singleton revival can spread one differing alive slot across
the stored array. The prescribed near-tie coupling can also pay a first-order
cost for different accepted copies. These statements retain labels, marks, or
a chosen coupling. The alive empirical transport problem discards dead rows
and optimizes the matching, so those lower bounds do not decide its rate.

Convergence of a law also asks a different question from contraction of two
sampled clouds. Kinetic smoothing can make runs forget their initial law even
when some updates separate nearby configurations. The conservative results
below complete this law-relaxation question with population-independent time
constants in their stated regimes. A population-independent survivor-block
comparison for the original killed gas remains separate; phase concentration
retains the requirements in
{prf:ref}`rem-native-stationary-residual`.
:::

:::{prf:remark} Scope of population-independent quadratic contraction
:label: rem-w2-global-quadratic-obstructions

For the complete terminal-box update, a global estimate

$$
\mathbb E\mathscr Q_{\rm mark}(S^+,\widetilde S^+)
\le (1-\kappa)\mathscr Q_{\rm mark}(S,\widetilde S)+r_N,
\qquad \kappa>0,\quad r_N\longrightarrow0,
$$

fails in the normalized fixed-slot quadratic of Chapter 18a. With a
positive status cost, {prf:ref}`prop-ku-terminal-mark-quadratic-obstruction`
proves failure on all-alive consensus inputs, under every coupling of
the separately survival-conditioned output laws. With zero status
cost, {prf:ref}`prop-ku-singleton-revival-uniform-obstruction` proves
failure on the global nonextinct class, again under every coupling.
For all-alive physical inputs, the particular maximal-source and shared-noise
coupling fails at near fitness ties by
{prf:ref}`thm-ku-allalive-source-coupling-obstruction`.

These conclusions concern the displayed full-array metric and coupling
scopes. They do not prove failure of optimal transport between alive
empirical measures or of transport between alive-sampled output laws.
The finite-population theorem above proves convergence by primitive
kernel comparison and eigenfunction reweighting. A population-independent
alive-swarm estimate may use another coupling in the same Wasserstein
metric, a phase-specific comparison with verified excursion control, or
a separately proved survivor-block estimate. A nonvanishing affine residual
can bound discrepancy around a floor, but does not prove decay to zero.
:::

:::{prf:definition} Alive empirical transport and survival-conditioned sampled laws
:label: def-w2-alive-transport-targets

For a nonextinct state $S$ with alive set $A(S)$ of size $M(S)$,
define its empirical alive phase-space probability by

$$
\mu_S^{\rm alive}=\frac1{M(S)}\sum_{i\in A(S)}\delta_{(x_i,v_i)},
\qquad
D_{\rm alive}(S,T)=W_{2,G}^2(\mu_S^{\rm alive},\mu_T^{\rm alive}),
$$

where $G$ is a fixed positive-definite phase-space quadratic form.
Let $H_N(S)=Q_N(S)/Q_N1(S)$ be the separately normalized
one-update law conditional on swarm nonextinction. Its alive-sampled
probability is

$$
\lambda_S=H_N(S)K_{\rm alive}
=\int\mu_{S'}^{\rm alive}\,H_N(S,dS').
$$

The quantities $D_{\rm alive}(S,T)$,
$\mathbb E_\Gamma D_{\rm alive}(S',T')$ for a coupling of the
output swarm laws, and $W_{2,G}^2(\lambda_S,\lambda_T)$ are distinct.
The full-array quadratic $\mathscr Q_{\rm mark}$ also charges retained
dead coordinates and, for a positive status coefficient, mark differences.
Disabling death instead defines a conservative update with all rows
alive and no survival normalization. Conditioning every row to remain
alive is a further, different normalization.
:::

:::{prf:proposition} A lower bound on a prescribed coupling does not bound optimal alive transport
:label: prop-w2-prescribed-coupling-scope

For two all-alive $N$-row states, write
$q_G(z-\widetilde z)=(z-\widetilde z)^\mathsf TG(z-\widetilde z)$.
Then

$$
D_{\rm alive}(S,T)
=\min_{\sigma\in\mathfrak S_N}\frac1N
  \sum_i q_G(z_i-\widetilde z_{\sigma(i)})
\le\frac1N\sum_i q_G(z_i-\widetilde z_i).
$$

For any coupling $\Gamma_0$ of the separately survivor-conditioned
output laws, or of conservative all-alive output laws, optimal transport between
the output swarm laws, with cost $D_{\rm alive}$, is at most
$\int D_{\rm alive}\,d\Gamma_0$. A positive lower bound on
$\int\mathscr Q_{\rm mark}\,d\Gamma_0$ gives no lower bound on
either of these optimal costs. In particular,
{prf:ref}`thm-ku-allalive-source-coupling-obstruction` does not
establish an obstruction to the alive transport targets above.
The averaged alive laws also satisfy

$$
W_{2,G}^2(\lambda_S,\lambda_T)
\le\inf_{\Gamma\in\Pi(H_N(S),H_N(T))}
     \int D_{\rm alive}(S',T')\,\Gamma(dS',dT').
$$

*Proof.* A coupling of two equally weighted empirical measures is a
doubly stochastic matrix divided by $N$. Its cost is linear in that
matrix, so a minimizing permutation exists by the finite assignment
decomposition; the identity permutation is an admissible competitor.
At the output-law level, the infimum over couplings is at most the
cost of $\Gamma_0$. Neither inequality transfers a lower bound on
the competitor's cost to a lower bound on the infimum. Applying
$K_{\rm alive}$ additionally removes dead coordinates and changes
the empirical normalization, so full-array lower bounds cannot be
imported without another comparison. For each pair of alive counts,
the finite transport polytope has fixed rational weights and finitely
many vertices. Selecting its first minimum-cost vertex gives a
measurable optimal empirical plan. Integrating that plan against any
coupling $\Gamma$ of the two survivor laws gives an admissible plan
for $\lambda_S,\lambda_T$, proving the last inequality.
Finally, conditioning a paired
raw law on joint survival need not give its two separately normalized
survivor laws: its first marginal is weighted by the conditional
survival probability of the second component. $\square$
:::

:::{prf:remark} Phase-specific contraction and convergence to a common law
:label: rem-w2-phase-versus-law-convergence

A one-update Wasserstein contraction is an estimate for the actual
transition on its stated class of inputs. The name of a common basin
or a common eventual equilibrium does not verify the required signed
force and coupling inequalities. The Rastrigin construction of
{prf:ref}`prop-ku-rastrigin-alive-transport-expansion` gives a strict
one-update increase in positional alive transport, with death disabled
and with nonextinction-conditioned alive sampling, even for two
consensus inputs lying in the same deterministic overdamped basin.

The fixed-population convergence theorem makes a different assertion.
If $\lambda_n^\mu,\lambda_n^\zeta$ are its alive-sampled laws from
two initial swarm laws and $\lambda_Q$ is the sampled-alive QSD law,
then

$$
W_2(\lambda_n^\mu,\lambda_n^\zeta)
\le W_2(\lambda_n^\mu,\lambda_Q)
   +W_2(\lambda_n^\zeta,\lambda_Q)\longrightarrow0.
$$

This follows from the two proved convergence bounds and the triangle
inequality. It does not require the distance to decrease at every
update and does not assert pathwise contraction of independently
sampled swarms. A phase-local geometric bound requires its stated
coercivity and coupling conditions along the iterates, with actual
excursions accounted for. A global rate uses separately verified
mixing conditions and can depend on interwell transition probabilities.
:::

:::{prf:remark} Completed population-independent conservative alive-law regimes
:label: rem-w2-completed-conservative-alive-law

The conservative current-frame canonical gas has completed positive
population-independent law-relaxation results. Both retain the actual
sampled standardization, Gaussian companion laws, simultaneous copying,
recipient jitter, accepted-component Haar collisions, full BAOAB step
and cap. Every row stays alive; there is no viscosity, history term or
survival normalization. Their primitive force and noise conditions include

$$
H_c=\sup_x|x+\eta F(x)|<\infty,\qquad
\operatorname{Lip}(F)<\infty,\qquad q,s>0,
\quad \eta=(h/2)^2(1+e^{-\gamma h}).
$$

For a continuous bounded raw reward, use the finite sampled-array
constants of {prf:ref}`lem-slcw-finite-preparation` and the
two-update minorization $\epsilon_f>0$ of
{prf:ref}`thm-slcw-finite-uniform-law`. When its explicit
$q_f=(1-\epsilon_f+e_2)(1+e_1)<1$ test holds, that theorem proves
exact convergence to the finite-swarm invariant law $\Pi_N$. If
$\Xi_{N,n}$ and $\Xi_N^*$ are the laws of the random alive empirical
measure at time $n$ and under $\Pi_N$, respectively, and
$\lambda_{N,n}^a,\lambda_N^{a,*}$ are their mean measures, then

$$
\mathcal W_{W_{2,G}}^2(\Xi_{N,n},\Xi_N^*),\qquad
W_{2,G}^2(\lambda_{N,n}^a,\lambda_N^{a,*})
\le C_fq_f^{\lfloor n/2\rfloor/2},\qquad n\ge1,\ N\ge2.
$$

The primitive $C_f,q_f$ are independent of $N$, and the error tends
to zero at each fixed $N$. The explicit nonempty interval of fixed
positive reward and diversity exponents is
{prf:ref}`cor-slcw-finite-positive-exponents`. Complete realized
fitness ties are included; no positive realized variance or cloning
pressure is assumed.

For a quadratic-growth unbounded raw reward, including $R=-U$ with
$F=-\nabla U$ under {prf:ref}`cor-slcw-same-potential`, the
weighted population and particle-transfer results instead yield
{prf:ref}`thm-slcw-alive-uniform-law`. With its stationary population
law $\pi$, explicit $N$-independent $C_G,B_w,q_w$, and explicit
$\varepsilon_N\to0$,

$$
\mathbb E W_{2,G}^2(\mu_{N,n}^a,\pi),\qquad
W_{2,G}^2(\lambda_{N,n}^a,\pi)
\le C_G\sqrt{B_w}\,q_w^{\lfloor(n-1)/2\rfloor/2}
       +C_G\sqrt{\varepsilon_N},\qquad n\ge1.
$$

Its nonempty positive-exponent interval is
{prf:ref}`cor-slcw-positive-exponents`; a fully evaluated same-potential
example is {prf:ref}`cor-slcw-concrete-alive-profile`.
The finite-particle term remains when comparing an empirical swarm
with a deterministic population law. It is not present in the preceding
exact relaxation between laws at the same $N$.

The bounded completed force-center condition is unnecessary for the
conservative harmonic result
{prf:ref}`thm-kuhw-active-exact-uniform-law`. Its actual finite
preparation, local and global weighted source-energy coupling, and
one-update kinetic minorization prove an explicit positive exponent
interval at the nonresonant original timestep. It gives

$$
\|\lambda_{N,n}^a-\lambda_{N,\infty}^a\|_{\rm TV}
\le A_{\mu_N}^*q_*^n,
\qquad
\mathcal W_{2,\mathrm{emp},G}^2
 (\mathscr L_{N,n},\mathscr L_{N,\infty}),\quad
W_{2,G}^2(\lambda_{N,n}^a,\lambda_{N,\infty}^a)
\le\frac{2\lambda_{\max}(G)}\eta A_{\mu_N}^*q_*^n.
$$

Here $q_*<1$ and the coefficients are independent of $N$ when the
entering averaged second moment is uniformly bounded. The complete
$h=0.04$, $F=-x$ profile is
{prf:ref}`cor-kuhw-original-step-active-profile`, with explicitly
configured bounded reward $-\tanh(|x|^2/2)$, active selection, and
viscosity and death disabled. This target is its exact finite-swarm
invariant alive law and has no particle floor.

For the harmonic force $F=-x$, a separate positive count-viscosity
result is {prf:ref}`thm-ku-count-small-viscosity-contraction`.
With cloning and death disabled,
{prf:ref}`cor-ku-count-kinetic-invariant-law` proves both alive-law
Wasserstein bounds with the same $N$-independent rate
$\mathfrak q=q_0+\nu C_\nu<1$, and an explicit invariant moment
bound. The primitive interval includes strictly positive viscosity
at $h=0.04$; {prf:ref}`cor-ku-original-step-positive-count` supplies
its register. This statement uses neither the bounded completed
force-center condition nor a particle floor. Its viscosity interval
does not certify $\nu=0.3$, and active cloning is a separate
condition. At $\nu=0.3$ the active harmonic fourth-moment drift
{prf:ref}`lem-ku-count-active-harmonic-drift` is proved under its
explicit positive-exponent bound; it supplies a moment estimate
rather than a law-relaxation rate.

The complete harmonic cap certificate
{prf:ref}`thm-rcap-harmonic-whole-update` improves the nonviscous
baseline without applying the cap alone to an unsuitable cross cost.
Its larger positive count-viscosity interval is
{prf:ref}`cor-rcap-positive-count-interval`, and its finite-swarm
invariant and alive-law bounds remain exact, without a particle floor,
when cloning and death are disabled. The diagnostic original-parameter
endpoint is approximately $0.000260436$; the exact primitive expression
certifies the theorem and still excludes $\nu=0.3$.

For active selection and strictly positive count viscosity,
{prf:ref}`thm-pvb-active-population-convergence` and
{prf:ref}`thm-vupt-uniform-time` now prove actual alive-observation
relaxation toward the conservative population stationary law.
With the explicit $q'<1$, $\varepsilon_N\to0$ and $C_G,C_*$ of
{prf:ref}`cor-vupt-alive-w2`,
$$
 \mathbb E W_{2,G}^2(\widehat\mu_n^N,\pi_{\nu,\theta}),\qquad
 W_{2,G}^2(\mathbb E\widehat\mu_n^N,\pi_{\nu,\theta})
 \le C_G\sqrt{\min\{1,C_*q'^n+\varepsilon_N\}}.
$$
The random empirical-measure law obeys the same estimate. The actual
sampled preparation and both correlated viscous kicks are retained;
entering swarms need no independence or exchangeability. The sufficient
regime includes $F=-x$, $h=0.04$, bounded configured reward
$-\tanh(|x|^2/2)$, explicit positive selection and viscosity intervals,
death disabled, and a uniform initial averaged eighth-moment budget.
This result has a population target and a particle floor; it does not
assert exact dense finite-array invariant mixing or certify $\nu=0.3$.

For the raw same-potential channels $F=-x$, $R=-|x|^2/2$,
{prf:ref}`cor-rqf-active-population` and
{prf:ref}`thm-rqpt-uniform-time` prove population attraction and actual
finite-swarm alive Wasserstein relaxation with an explicit vanishing
uniform-time particle floor, using the actual logistic derivative decay.
The optimal alive empirical-law and alive-sampled bounds are
{prf:ref}`cor-rqpt-alive-w2`, under the raw primitive positive selection
and count-viscosity intervals and initial averaged eighth-moment budget.
For the terminal-box marked population update,
{prf:ref}`thm-kpf-large-box-population-convergence` additionally proves
attraction to a unique stationary revival population law in a primitive
nonempty regime with a fixed sufficiently large box and positive selection
and viscosity. {prf:ref}`cor-kpf-current-alive-relaxation` gives its current
alive TV and Wasserstein relaxation. Mandatory revival, each environment's
alive normalizer and terminal death are retained. This marked population
law is not identified with a finite-swarm QSD or survivor law.

The actual surviving finite-swarm transfer is now proved in
{prf:ref}`thm-spt-uniform-surviving-law` and {prf:ref}`cor-spt-alive-w2`.
For the fixed large-box, positive weak-selection and count-viscosity regime,
its optimal current-alive empirical-law and swarm-first sampled bounds are
$$
 \mathbb E[W_{2,G}^2(\widehat\alpha_n^N,\pi_L^A)\mid\tau_N>n],\qquad
 W_{2,G}^2(\mathbb E[\widehat\alpha_n^N\mid\tau_N>n],\pi_L^A)
 \le\mathcal D_G^2\min\{1,2u_{N,n}/m_f+c_{\rm s}r_N\},
$$
where $u_{N,n}=\min\{1,C_{\rm pop}r_*^{n-1}+\varepsilon_N^{\rm s}\}$,
$r_*<1$ is independent of $N$, and
$\varepsilon_N^{\rm s},r_N\to0$. The proof retains each swarm's own
survival denominator and the actual alive empirical normalization; it
charges survival only over a recent restart window. Arbitrary retained
entering dead coordinates are allowed. Its target is the alive restriction
of the marked population stationary law, with a vanishing particle floor,
and is not identified with an exact finite-swarm QSD.

The same-potential raw quadratic surviving law is completed in
{prf:ref}`cor-rqk-surviving-alive-law`. Its unchanged channels are
$F=-x$, $R=-|x|^2/2$; all population endpoints are computed before the
fixed sufficiently large box is chosen. Only the finite weak comparison
uses the subsequent alive-box reward bounds. It gives the preceding
survival-conditioned optimal alive-law estimate with its own explicit
raw positive selection/viscosity interval and vanishing particle floor.

The row-normalized branch is now completed separately in
{prf:ref}`thm-rpf-marked-population`,
{prf:ref}`thm-rft-uniform-surviving-law` and
{prf:ref}`cor-rft-alive-wasserstein`, with the all-slot sampling order
in {prf:ref}`cor-rft-all-slot-alive-sample`. Its true empirical degrees,
exact self exclusion and uncapped correlated OU velocity numerators
are controlled by {prf:ref}`thm-rft-conditional-consistency` and
{prf:ref}`thm-rwm-population-modulus`. These yield the same form of
optimal physical alive-law bound toward the stationary row population
law, with an explicit vanishing uniform-time particle floor. The
unchanged raw channels $F=-x$, $R=-|x|^2/2$ and each swarm's own
current-survival denominator are retained. Its nonempty positive
selection/row-viscosity interval is
{prf:ref}`def-rpf-positive-endpoints`, after a fixed sufficiently
large box is chosen.

The original box and viscosity additionally have positive alive coverage
and exact recent-survival normalization in
{prf:ref}`thm-dsa-default-box-alive-floor` and
{prf:ref}`cor-dsa-recent-window-normalization`. At that viscosity the
actual count alignment dissipation and frozen-provider gaps are
{prf:ref}`thm-rcb-finite-dissipation` and
{prf:ref}`thm-rcb-source-box-frozen-gap`. Those results supply
confinement, normalization and common-provider mixing; an actual
own-provider nonlinear block is still needed for a default-preset law
rate. They do not replace that block by assumed signed feedback.

The original active dense-viscosity preset at $h=0.04$, $\nu=0.3$
still requires a population-uniform law estimate.
At its harmonic force and $L=2$, the actual velocity burn and
full-Gaussian provider accounts are now completed in
{prf:ref}`thm-rvb-population-burn`,
{prf:ref}`thm-rfp-first-provider-cap` and
{prf:ref}`thm-rfk-two-count-principal`. The last result retains
anisotropic principal losses $.00149\,d_X^2+.0721\,d_P^2$ through
both count kicks and the native cap; both actual spatial derivatives
remain separate signed terms. Actual finite marked consistency and
uniform-time empirical-provider moment budgets are completed in
{prf:ref}`thm-dmc-default-interface` and
{prf:ref}`thm-dev-current-provider-budget`, in their explicit weak
positive exponent interval. These proofs keep unrestricted dead
coordinates and each own recent-survival normalizer. They do not
absorb the full spatial/preparation/mark feedback and therefore do
not certify default alive-law attraction.
Default physical kinetic absorption is additionally complete on
arbitrary-position constant-velocity slices and pointwise velocity bands
in {prf:ref}`thm-dsa53-physical-shapes` and
{prf:ref}`cor-csb-optimal-law`. The exact correlated Gaussian tensor and
cap-loss identity are {prf:ref}`thm-dsg-signed-second-response`.
They close a signed, nonconstant inward-velocity family in
{prf:ref}`thm-dsg-inward-family`. Its prepared kinetic endpoints have
actual finite alive empirical-law and both sampled-law transport in
{prf:ref}`thm-siat-signed-alive-laws`, with each own denominator,
an $N$-independent coefficient, no floor and a stated minimum separation.
These classes are not asserted preserved by a full noisy active update.

The shrinking source-boundary envelopes are preserved in
{prf:ref}`thm-dsti-preserved-boundary`; they do not imply the fixed-width
interior criterion, whose proposed preservation is refuted in
{prf:ref}`prop-dsti-interior-nonpreservation`. The actual complete
full-dead feedback and chronological marked-law response are
{prf:ref}`thm-dlb-default-feedback` and
{prf:ref}`thm-dlb-delayed-response`. Their explicit absolute gain does
not absorb the default frozen gap. The exact delayed moment identity
{prf:ref}`thm-dsti-delayed-moments` keeps source/Haar, both count
and native-cap mixed residuals instead of assigning them a favorable sign.

The general prepared-input balance is exact in
{prf:ref}`thm-dbl68-ledger`: the harmonic loss, both signed provider
forms and complete cap loss remain together. The conditional native-cap
coercivity in {prf:ref}`thm-ccl-conditional-coercivity` retains their
correlation with the actual noisy provider, while
{prf:ref}`lem-ccl-first-force` uses the averaged velocity budget only
in independent environment factors. The actual OU trace and nonlinear
inward flux are {prf:ref}`thm-icb-stein-cap` and
{prf:ref}`cor-icb-cross-flux`. These identities and bounds do not yet
give a positive general signed margin or a delayed default law gap.
The exact recipient-jitter consumer
{prf:ref}`thm-fjt73-first-account` retains mismatched copied statuses
and the finite common-query and coincident-environment terms in the
complete first-force square. Its restricted inward-source linear sign
in {prf:ref}`cor-fjt73-inward-source` does not discharge that square
or the full second-provider/cap and preparation/marking account.
The full-Gaussian cap deficit
{prf:ref}`thm-gca74-cap-deficit` covers general noisy velocity laws
under its stated provider or prepared-array moments. It acts on
pre-OU-fixed vectors; the actual force remains correlated with the
cap matrix. The restriction formula
{prf:ref}`prop-gca74-own-restriction` retains both weighted removed
moments before each own survival or alive normalization.
The sharper centered operator
{prf:ref}`thm-ffo77-operator` passes the actual first-provider
positional threshold at $\nu=.3$. Its endpoint
{prf:ref}`cor-ffo77-position-endpoints` retains the velocity
displacement term and the declared population or fixed-array RMS
budgets. It is a physical positional estimate before marking;
the full phase and alive-law comparison remain separate.
The complete source-plan population capped spatial residual is
{prf:ref}`cor-nca76-complete-spatial-residual`. Its radial
first-force cancellation and actual second-provider bound retain the
noncommuting count operator and every Gaussian outcome.
The actual conditional mean-cap matrix and its full-jitter weighted
sector are now proved in {prf:ref}`thm-mcm79-lower` and
{prf:ref}`thm-mcm79-source-sector`. A root-uniform conditional lower
is impossible on actual supported revival preparations; this excludes
that estimate without refuting delayed law mixing.
The signed first-provider/cap auxiliary consumer
{prf:ref}`thm-sfc80-first-consumer` has positive population and
fixed finite-array margins under its declared prepared budgets.
Its pair $(R,DZ_b)$ is an auxiliary differential, and the entire
actual second response remains in
{prf:ref}`cor-sfc80-second-interface`; no endpoint transport map
is inferred from that auxiliary coefficient.
{prf:ref}`thm-wsd81-surviving-deficit` additionally bounds mixed
extinction and empirical-position losses and retains a paired cap
deficit after each own nonextinction division in its explicit
large-$N$, pathwise low-speed class. That class is not asserted
invariant. The complete second-provider signed term, actual
preparation/revival, terminal marking and separately normalized
alive readouts remain the default-law obligation.

For collapsed tied inputs, {prf:ref}`thm-tqp-sampled-position` gives an
exact size-independent sampled-position estimate at every separation.
{prf:ref}`thm-tqp-empirical-local-regularity` shows why local one-update
Lipschitz continuity of the random alive empirical-law readout is a
different issue. Its universal physical variance bound does not refute
delayed law mixing, population attraction or the completed uniform regimes.
For its force $F(x)=-x$, the unbounded conservative center profile is
$\sup_x|(1-\eta)x|=\infty$. The terminal-box result
{prf:ref}`thm-w2-finite-n-conditioned-convergence` remains a separate
completed theorem with its fixed-$N$, survival-conditioned rate.
:::

## 10. Conclusion and Future Work

### 10.1. Main Achievements

This document establishes the centered positional $W_2$ decomposition,
the cloning reset bound, the $N$-uniform Keystone pressure input, and the
fixed-population conditioned convergence theorem
{prf:ref}`thm-w2-finite-n-conditioned-convergence`. It also records the
completed conservative alive-law regimes of
{prf:ref}`rem-w2-completed-conservative-alive-law`:

1. ✅ **Avoids q_min problem**: No dependence on minimum matching probability
2. **Keystone pressure**: Constants sourced from {doc}`03_cloning`
3. ✅ **No alignment axiom**: No cross-swarm geometric alignment assumptions required
4. **N-uniform input**: The pressure coefficient is independent of $N$;
   the finite-size correction is explicit
5. **Finite-population convergence**: Explicit primitive tests certify conditioned full-swarm TV convergence and alive-marginal $W_2$ convergence, including fitness ties.
6. ✅ **Framework-consistent**: Uses the declared native update and exact cluster definitions from the Keystone Lemma chain
7. **Population-independent alive-law relaxation**: The conservative bounded-reward regime converges exactly to each finite-swarm stationary law; the raw-reward regime gives a uniform time rate toward the population law with a vanishing particle error.

### 10.2. Open Questions

:::{div} feynman-prose
Population-independent alive-law relaxation is proved in the conservative
regimes of {prf:ref}`rem-w2-completed-conservative-alive-law`. At the original
harmonic step size, active cloning with zero viscosity converges exactly to
the finite-swarm stationary law. Active cloning with small positive count
viscosity converges toward the population stationary law with a particle
floor that vanishes with $N$; the raw quadratic reward from the same harmonic
potential is included. These results keep both actual viscous kicks and
their correlated second provider. They require no positive realized fitness
variance and never require variance to exceed its maximum.

The marked population extension is also complete for a sufficiently large
fixed box and its explicit small positive parameter interval, including the
raw quadratic reward from the same harmonic potential. It includes
mandatory revival and the terminal mark, and its current-time alive law
approaches the alive restriction of the stationary revival law. The
finite-swarm step is also proved in that regime: conditioning on survival
through the observation time gives a geometric population term and a particle
floor that vanishes with $N$. It keeps each swarm's own normalized alive
measure and the survival-induced tilt on a recent window. The remaining
reference regime concerns viscosity $\nu=0.3$, box radius $L_D=2$, and the
row-normalized force where specified. Those values are outside the completed
law certificates. The full-array quadratic obstructions retain their stated
metric and coupling scopes.

A local contraction theorem needs quantitative force coercivity and control
of the other update terms on its stated population class. It must also justify
when the evolving laws remain in that class. Rastrigin's negative-curvature
regions prevent a common phase label from serving as that proof. The
finite-population QSD theorem is complete under its stated primitive hypotheses;
it does not assert that two arbitrary sampled swarms become closer at every
step.
:::

### 10.3. Relation to Framework

This result contributes positional control to the larger programme:

:::{div} feynman-prose
- **Finite-horizon propagation of chaos** ({doc}`09_propagation_chaos`) also requires the actual population map's consistency and continuity estimates.
- **Long-time population analysis** ({doc}`08_mean_field`) requires additional stability or phase-identification arguments.
- **Population-independent alive-law control** is completed in the stated
  conservative regimes. Its extension to the original dense-viscosity and
  killed gas requires the corresponding complete-law comparison.
:::



## References

**Primary**: {doc}`03_cloning` Chapters 6-8 (Keystone Principle) and Chapter 10 (variance drift)

**Key Results Used**:

- {prf:ref}`def-unified-high-low-error-sets` and {prf:ref}`def-unfit-set`:
  geometric populations and the realized unfit set
- {prf:ref}`thm-keystone-complete-error-coverage` and
  {prf:ref}`thm-keystone-discharged-averaged-pressure`:
  complete measurement-averaged pressure with self-exclusion correction
- {prf:ref}`thm-stability-condition-final-corrected` and
  {prf:ref}`thm-unfit-high-error-overlap-fraction`:
  conditional arithmetic gap and overlap, including their variance correction
- {prf:ref}`lem-unfit-cloning-pressure`,
  {prf:ref}`lem-error-concentration-target-set`, and
  {prf:ref}`lem-quantitative-keystone`: the optional realized-target route
- {prf:ref}`thm-positional-variance-contraction`: positional reset and exact drift
- {prf:ref}`thm-slc-signed-complete-update`: full signed cloning–kinetic accounting
- {prf:ref}`thm-cgd-analytic-force-qsd`, {prf:ref}`lem-cgd-two-update-density`,
  and {prf:ref}`thm-cgd-primitive-eigenfunction`: completed finite-population
  conditioned convergence with primitive constants
- {prf:ref}`thm-slcw-finite-uniform-law` and
  {prf:ref}`cor-slcw-finite-positive-exponents`: exact conservative finite-swarm
  alive-law relaxation with a population-independent rate
- {prf:ref}`thm-slcw-alive-uniform-law` and
  {prf:ref}`cor-slcw-concrete-alive-profile`: unbounded-raw-reward alive
  transport toward the population law, with explicit particle error

**Secondary**:
- {doc}`01_fragile_gas_framework`: Axioms
- {doc}`15_kl_convergence`: Alternative convergence analysis



**Document scope**: Centered positional reset, explicit Keystone pressure inputs, finite-population QSD convergence, and the completed conservative population-independent alive-law regimes.

:::{div} feynman-prose
**Remaining population-independent extension**: Carry the completed
conservative alive-law comparison to the original dense-viscosity or killed
transition, with its actual normalization and all feedback terms. A local
result additionally needs a justified population class and quantitative force
and update bounds. The full-array counterexamples retain their stated metric
and coupling scopes.
:::
