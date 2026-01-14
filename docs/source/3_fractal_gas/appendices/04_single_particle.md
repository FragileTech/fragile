# Single Walker Observables and Probability Fields

## 0. TLDR

**Single Walker Observable Stack**: We formalize the observable map of a single walker ({prf:ref}`def-walker`) inside a swarm ({prf:ref}`def-swarm-and-state-space`). The stack couples positional, velocity, and potential data through a fixed parameter set $\Theta_{\text{obs}}$ and produces the diversity, reward, and fitness channels used by the Fragile Gas.

**Diversity Channel Distribution**: The distribution $\mathcal{P}_{D(i)}$ of the diversity channel is an explicit sum over all companion maps produced by the sequential greedy pairing routine (including a single self-pair when $k$ is odd). Each term factors into the probability of the pairing realization and a deterministic evaluation of the regularized distance statistic.

**Conditional Cloning Probability Field**: For every potential companion $j$, the cloning pipeline produces a random variable $\Pi(i|j)$ whose law is inherited from diversity pairing realizations. The family $\{\mathcal{P}_{\Pi(i|j)}\}_{j}$ is the discrete probability field controlling local exploration pressure ({prf:ref}`def-cloning-probability`).

**Post-Cloning Mixture**: The position of walker $i$ after the cloning operator ({prf:ref}`def-cloning-operator-formal`) follows a mixed distribution with a Dirac mass at the original position and a Gaussian mixture centered at the companions. The weights are the joint probabilities of selecting each companion and executing a cloning action.

**Kinetic Coupling**: Convolving the post-cloning mixture with the BAOAB kinetic propagator ({prf:ref}`def-baoab-update-rule`) produces the full single-step transition kernel consistent with the kinetic operator ({prf:ref}`def-kinetic-operator-stratonovich`). This kernel directly yields the single walker death probabilities through boundary integrals over $\mathcal{X}_{\mathrm{valid}}$ ({prf:ref}`def-valid-state-space`).

## 1. Introduction

### 1.1. Goal and Scope

This chapter isolates the **single walker** perspective inside the Euclidean Gas. Starting from a fixed swarm configuration $S_t = ((x_i, v_i))_{i=1}^N$, we describe every random object that influences the evolution of a specific alive walker $i \in \mathcal{A}(S_t)$ ({prf:ref}`def-alive-dead-sets`). The objective is to turn the algorithmic pipeline—diversity measurement, reward assessment, cloning, and kinetic motion—into explicit probability distributions on $\mathbb{R}^d$ that can be used for both analysis and simulation.

### 1.2. Relationship to the Framework

The macroscopic drift and Lyapunov statements of Chapter 3 rely on aggregate statistics such as the positional variance or the hypocoercive Wasserstein distance. The present chapter supplies the microscopic objects that feed those statistics:

1.  The **diversity channel** captures local geometric information via the algorithmic distance ({prf:ref}`def-alg-distance`).
2.  The **reward channel** translates the energy landscape and velocity regularization into standardized scores.
3.  The **fitness potential** combines both channels multiplicatively to determine competitive outcomes.
4.  The **conditional cloning field** and **post-cloning mixture** describe the actual transitions used in {prf:ref}`def-cloning-operator-formal`.
5.  The **kinetic convolution** produces the law of the full update operator ({prf:ref}`def-swarm-update-procedure`) at the single walker level.

All expressions below are consistent with the Fragile Gas axioms ({prf:ref}`def-fragile-gas-axioms`) and adopt the same notation as Chapter 3.

### 1.3. Notation

*   $\mathcal{A}(S_t)$ denotes the set of alive walkers at time $t$ with cardinality $k = |\mathcal{A}(S_t)|$.
*   $c_j$ is the companion of walker $j$ in the companion map $M \in \mathcal{M}(\mathcal{A}(S_t))$; if $k$ is odd (or $k=1$), the leftover walker is mapped to itself, $c_j=j$.
*   $d_{\text{alg}}$ is the algorithmic distance with weight $\lambda_{\text{alg}}$ ({prf:ref}`def-algorithmic-distance-metric`).
*   $Z(\cdot)$ and $Z_r(\cdot)$ denote standardized statistics with a patched variance $\sigma'$, ensuring the denominators are strictly positive.
*   Bold symbols represent deterministic functions; calligraphic symbols denote distributions or fields.

## 2. The Single Walker Parameter Stack

We fix the parameter set

$$
\Theta_{\text{obs}} = \{ \lambda_{\text{alg}}, \epsilon_d, \epsilon_c, \epsilon_{\text{dist}}, \rho, \sigma_{\text{min}}, A, \eta, \alpha, \beta, p_{\max}, \epsilon_{\text{clone}}, \sigma_x, c_{v\_\text{reg}}, U(\cdot), \gamma, \beta_{\text{kin}}, \Delta t \}.

$$
The first block controls the diversity and reward channels, the second block fixes the cloning companion selection and thresholding rule, and the last block governs the kinetic step. In the reference implementation, a single companion-selection operator is typically reused for diversity and cloning, so $\epsilon_c = \epsilon_d$ unless separate operators are configured. Additional algorithm-specific constants (e.g., restitution coefficients or jitter variances) are absorbed into $\Theta_{\text{obs}}$ but are omitted from the notation when not required.

:::{definition} Observable Parameter Stack
:label: def-single-observable-stack

Given a swarm state $S_t$, the **observable stack** of walker $i$ is the tuple

$$
\mathsf{Obs}(i, S_t; \Theta_{\text{obs}}) := (\mathcal{P}_{D(i)}, R_i, \mathcal{P}_{V(i)}, \mathcal{F}_{\Pi(i)}, \mathcal{P}_{X'_i}, \mathcal{P}_{X''_i}),

$$
where each component is defined in Sections 3–7 below. Every map depends measurably on $S_t$ and on the algorithmic randomness of the diversity pairing, the cloning threshold, and the kinetic noise.
:::

## 3. Distribution of the Diversity Channel

### 3.1. Matching Probabilities

The **sequential greedy pairing** algorithm produces a perfect matching $M$ of $\mathcal{A}(S_t)$ when $k$ is even. When $k$ is odd, the implementation returns a companion map that is an involution with one fixed point: the leftover walker is mapped to itself, $c_i=i$, which keeps the distance measurement well-defined because of the $\epsilon_{\text{dist}}$ regularizer. When walker $i$ chooses among the unpaired set $U \subseteq \mathcal{A}(S_t)$, the selection probability for $j \in U \setminus \{i\}$ is

$$
P(C_i = j \mid i, U; \lambda_{\text{alg}}, \epsilon_d) = \frac{\exp\left(-\frac{d_{\text{alg}}(i, j; \lambda_{\text{alg}})^2}{2\epsilon_d^2}\right)}{\sum_{\ell \in U \setminus \{i\}} \exp\left(-\frac{d_{\text{alg}}(i, \ell; \lambda_{\text{alg}})^2}{2\epsilon_d^2}\right)}.

$$
If $U \setminus \{i\} = \varnothing$ (e.g., $k=1$), the implementation sets $c_i=i$.

Let $\mathcal{I}(M)$ denote the ordered list of walkers selected by the greedy algorithm as the "first" element of each pair. The probability of a pairing realization $M \in \mathcal{M}(\mathcal{A}(S_t))$ is the product of the sequential choices that realize those pairs, conditioned on the processing order used by the greedy algorithm (the fixed point, if present, contributes no factor):

$$
P(M \mid S_t; \lambda_{\text{alg}}, \epsilon_d) = \prod_{i \in \mathcal{I}(M)} P(C_i = c_i \mid i, U_i(M); \lambda_{\text{alg}}, \epsilon_d).

$$
where $U_i(M)$ records the remaining unmatched walkers when $i$ is processed.

:::{remark} Independent Companion Selection
If the diversity channel uses an independent softmax companion selector instead of mutual pairing, interpret $M$ as the vector of companion indices $(c_i)_{i \in \mathcal{A}(S_t)}$. Then

$$
P(M \mid S_t) = \prod_{i \in \mathcal{A}(S_t)} P(C_i = c_i \mid S_t).
$$

with the same self-pairing fallback when only one alive walker is available. The downstream definitions of $d_i$, $Z(i, M, S_t)$, and $\mathcal{P}_{D(i)}$ are unchanged.
:::

### 3.2. Regularized Distance Measurement

For every walker $j$, the regularized distance to its companion is

$$
d_j(M; \lambda_{\text{alg}}, \epsilon_{\text{dist}}) := \sqrt{\|x_j - x_{c_j}\|^2 + \lambda_{\text{alg}} \|v_j - v_{c_j}\|^2 + \epsilon_{\text{dist}}^2}.

$$
The regularization prevents degeneracy when walkers coincide and ensures differentiability of the subsequent statistics.

### 3.3. Standardization Regimes

Let $k = |\mathcal{A}(S_t)|$. Two regimes are supported:

*   **Global statistics ($\rho = \mathrm{None}$):**

    $$
    \mu_d(M, S_t) = \frac{1}{k} \sum_{j \in \mathcal{A}(S_t)} d_j(M; \cdot), \quad
    \sigma'_d(M, S_t; \sigma_{\text{min}}) = \sqrt{\frac{1}{k} \sum_{j \in \mathcal{A}(S_t)} (d_j - \mu_d)^2 + \sigma_{\text{min}}^2}.

    $$
*   **Localized statistics ($\rho < \infty$):**

    $$
    K_{\rho}(i, j) = \exp\left(-\frac{d_{\text{alg}}(i, j; \lambda_{\text{alg}})^2}{2\rho^2}\right), \quad
    \mu_{\rho, d}(i) = \frac{\sum_{j \in \mathcal{A}(S_t)} K_{\rho}(i, j) d_j}{\sum_{\ell \in \mathcal{A}(S_t)} K_{\rho}(i, \ell)},

    $$
    $$
    \sigma'_{\rho, d}(i; \sigma_{\text{min}}) = \sqrt{\frac{\sum_{j \in \mathcal{A}(S_t)} K_{\rho}(i, j) (d_j - \mu_{\rho, d}(i))^2}{\sum_{\ell \in \mathcal{A}(S_t)} K_{\rho}(i, \ell)} + \sigma_{\text{min}}^2}.

    $$

### 3.4. Diversity Distribution

The standardized score of walker $i$ is

$$
Z(i, M, S_t) = \begin{cases}
\dfrac{d_i(M; \cdot) - \mu_d(M, S_t)}{\sigma'_d(M, S_t; \sigma_{\text{min}})}, & \rho = \mathrm{None}, \\
\dfrac{d_i(M; \cdot) - \mu_{\rho, d}(i)}{\sigma'_{\rho, d}(i; \sigma_{\text{min}})}, & \rho < \infty.
\end{cases}

$$
After applying the logistic rescale and positivity floor, the diversity channel is

$$
\mathcal{D}_i(M, S_t) = \left( \frac{A}{1 + \exp(-Z(i, M, S_t))} + \eta \right)^{\beta}.

$$
The probability mass function of $\mathcal{D}_i$ is therefore

$$
\mathcal{P}_{D(i)}(v \mid S_t; \Theta_{\text{obs}}) = \sum_{M \in \mathcal{M}(\mathcal{A}(S_t))} \mathbf{1}_{\{\mathcal{D}_i(M, S_t) = v\}} \cdot P(M \mid S_t; \lambda_{\text{alg}}, \epsilon_d).

$$
Each atom corresponds to a companion map realization whose deterministic evaluation equals $v$.

## 4. Reward Channel and Fitness Potential

### 4.1. Raw Reward

The deterministic reward of walker $j$ uses the external potential $U$ and the velocity regularization coefficient $c_{v\_\text{reg}}$:

$$
r_j(S_t; U, c_{v\_\text{reg}}) = -U(x_j) - c_{v\_\text{reg}} \|v_j\|^2.

$$

### 4.2. Reward Standardization

The global statistics follow the same form as Section 3.3, using the full alive set. For a localized regime the same kernel $K_{\rho}$ is re-used:

$$
\mu_r(S_t) = \frac{1}{k} \sum_{j} r_j, \quad \sigma'_r(S_t; \sigma_{\text{min}}) = \sqrt{\frac{1}{k} \sum_j (r_j - \mu_r)^2 + \sigma_{\text{min}}^2},

$$
$$
\mu_{\rho, r}(i) = \frac{\sum_j K_{\rho}(i, j) r_j}{\sum_{\ell} K_{\rho}(i, \ell)}, \quad
\sigma'_{\rho, r}(i; \sigma_{\text{min}}) = \sqrt{\frac{\sum_j K_{\rho}(i, j) (r_j - \mu_{\rho, r}(i))^2}{\sum_{\ell} K_{\rho}(i, \ell)} + \sigma_{\text{min}}^2}.

$$
The standardized score $Z_r(i, S_t)$ follows from the corresponding means and variances.

### 4.3. Reward Channel Value

The reward channel is deterministic once $S_t$ is fixed:

$$
R_i(S_t; \Theta_{\text{obs}}) = \left( \frac{A}{1 + \exp(-Z_r(i, S_t))} + \eta \right)^{\alpha}.

$$

### 4.4. Fitness Potential Distribution

:::{proposition} Single Walker Fitness Distribution
:label: prop-single-fitness-distribution

The fitness potential $V_{\text{fit}}(i, S_t) = \mathcal{D}_i(S_t) \cdot R_i(S_t)$ has probability mass function

$$
\mathcal{P}_{V(i)}(v \mid S_t; \Theta_{\text{obs}}) = \mathcal{P}_{D(i)}\left( \frac{v}{R_i(S_t)} \Bigm| S_t; \Theta_{\text{obs}} \right).

$$
Therefore, $\mathcal{P}_{V(i)}$ inherits the atomic structure of $\mathcal{P}_{D(i)}$ scaled by the deterministic reward factor.
:::

## 5. Conditional Cloning Probability Field

### 5.1. Cloning Scores

Given a pairing realization $M$, the cloning score of walker $i$ relative to companion $j$ is ({prf:ref}`def-cloning-score`)

$$
s(i \mid j, M) = \frac{V_{\text{fit}}(j, M, S_t) - V_{\text{fit}}(i, M, S_t)}{V_{\text{fit}}(i, M, S_t) + \epsilon_{\text{clone}}}.

$$
This value becomes a random variable once $M$ is sampled.

### 5.2. Thresholding and Probabilities

Let $T_i \sim \mathrm{Uniform}(0, p_{\max})$ be the stochastic threshold. The clipped probability ({prf:ref}`def-cloning-probability`) is

$$
\pi_{\text{clip}}(s; p_{\max}) = \min\left\{1, \max\left\{0, \frac{s}{p_{\max}}\right\}\right\}.

$$
Conditioned on $M$ and a companion $j$, the cloning action indicator is $\mathbf{1}_{\{ s(i \mid j, M) > T_i \}}$ ({prf:ref}`def-cloning-decision`), and $\mathbb{P}(\text{clone} \mid s(i \mid j, M)) = \pi_{\text{clip}}(s(i \mid j, M); p_{\max})$.

### 5.3. Field Definition

The **conditional cloning probability field** of walker $i$ is the map

$$
\mathcal{F}_{\Pi(i)}(S_t; \Theta_{\text{obs}}): j \longmapsto \mathcal{P}_{\Pi(i\mid j)}(p \mid S_t; \Theta_{\text{obs}}),

$$
where

$$
\mathcal{P}_{\Pi(i\mid j)}(p \mid S_t; \Theta_{\text{obs}}) = \sum_{M \in \mathcal{M}(\mathcal{A}(S_t))} \mathbf{1}_{\{ \pi_{\text{clip}}(s(i \mid j, M); p_{\max}) = p \}} \cdot P(M \mid S_t; \lambda_{\text{alg}}, \epsilon_d).

$$
Its expectation

$$
\bar{p}(i \mid j) := \mathbb{E}_{M}[\pi_{\text{clip}}(s(i \mid j, M); p_{\max})]

$$
determines the mean contribution of $j$ to the total cloning probability $\pi_{\text{clone}}(i \mid S_t)$. Assuming the companion draw $C_i$ is independent of the diversity pairing $M$, this total probability is

$$
\pi_{\text{clone}}(i \mid S_t) = \sum_{j \in \mathcal{A}(S_t) \setminus \{i\}} P_{C_i}(j \mid S_t; \lambda_{\text{alg}}, \epsilon_c) \, \bar{p}(i \mid j),

$$
consistent with {prf:ref}`def-cloning-probability`. Here $P_{C_i}$ is the cloning companion selection distribution ({prf:ref}`def-cloning-companion-operator`).

## 6. Post-Cloning Position Distribution

### 6.1. Mixture Structure

Let $x_i$ denote the original position of walker $i$. After the cloning operator, the position $X'_i$ has the mixed distribution

$$
\mathcal{P}_{X'_i}(x' \mid S_t; \Theta_{\text{obs}}) = (1 - \pi_{\text{clone}}(i \mid S_t)) \cdot \delta(x' - x_i) + \sum_{j \in \mathcal{A}(S_t) \setminus \{i\}} w_{ij}^{\text{joint}} \cdot \mathcal{N}(x'; x_j, \sigma_x^2 I_d),

$$
where $\delta$ is the Dirac mass and $\mathcal{N}(\cdot; x_j, \sigma_x^2 I_d)$ is the isotropic Gaussian induced by the spatial jitter.

### 6.2. Joint Weights

The joint weights factorize as

$$
w_{ij}^{\text{joint}} = \bar{p}(i \mid j) \cdot P_{C_i}(j \mid S_t; \lambda_{\text{alg}}, \epsilon_c),

$$
and obey $\sum_{j} w_{ij}^{\text{joint}} = \pi_{\text{clone}}(i \mid S_t)$. Each Gaussian component is therefore weighed by the probability of selecting $j$ as a companion and subsequently cloning against $j$.

:::{remark} Interpretation as a Discrete Random Field

The function $x' \mapsto \mathcal{P}_{X'_i}(x')$ is a random field supported on the alive swarm. Locations near highly informative companions inherit large Gaussian weights, while inactive regions contribute only through the persistence mass. This view is convenient when studying local exploration pressure or extinction risk inside restricted domains.
:::

## 7. Coupling with the Kinetic Operator

### 7.1. BAOAB Convolution

Let $(X'_i, V'_i)$ denote the state immediately after cloning. The kinetic operator $\Psi_{\text{kin}}$ in Stratonovich form ({prf:ref}`def-kinetic-operator-stratonovich`) is implemented numerically through the BAOAB update ({prf:ref}`def-baoab-update-rule`). Let $\mathcal{K}_{\text{BAOAB}}^x(x'' \mid x', v')$ denote the positional marginal of the BAOAB kernel. In the reference implementation, this kernel is the pushforward of Gaussian noise through the full BAOAB map (including potential forces and any optional adaptive terms), so it need not be Gaussian when the force field is nonlinear. The full single-step transition law is the convolution

$$
\mathcal{P}_{X''_i}(x'' \mid S_t; \Theta_{\text{obs}}) = \iint \mathcal{K}_{\text{BAOAB}}^x(x'' \mid x', v') \, \mathcal{P}_{X'_i, V'_i}(x', v' \mid S_t) \; \mathrm{d}x' \, \mathrm{d}v',

$$
where the joint distribution $\mathcal{P}_{X'_i, V'_i}$ is induced by the cloning operator (its positional marginal is given in Section 6).

### 7.2. Death Probability Field

Let $\mathcal{X}_{\mathrm{valid}} \subset \mathbb{R}^d$ be the viable domain ({prf:ref}`def-valid-state-space`). For each companion $j$ let $E_{ij}$ denote the event that $C_i=j$ and the cloning action indicator equals 1. The conditional probability of exiting $\mathcal{X}_{\mathrm{valid}}$ after cloning to $j$ and applying the kinetic step is

$$
\pi_{\text{death}}(i \mid j, S_t) = \int_{\mathbb{R}^d \setminus \mathcal{X}_{\mathrm{valid}}} \mathcal{P}_{X''_i \mid E_{ij}, S_t}(x'') \, \mathrm{d}x'',

$$
where $\mathcal{P}_{X''_i \mid E_{ij}, S_t}$ is obtained by convolving the conditional post-cloning law of $(X'_i, V'_i)$ given $E_{ij}$ with the positional BAOAB kernel $\mathcal{K}_{\text{BAOAB}}^x$. The total death probability is the weighted sum

$$
\Pi_{\text{death}}(i \mid S_t) = (1 - \pi_{\text{clone}}(i \mid S_t)) \pi_{\text{death}}^{\text{persist}}(i) + \sum_{j} w_{ij}^{\text{joint}} \pi_{\text{death}}(i \mid j, S_t),

$$
with the persistence term computed by integrating the kinetic kernel based on the Dirac mass at $(x_i, v_i)$.

## 8. Parameter Glossary

| Parameter | Symbol | Unit | Role | Primary Reference |
| :-- | :-- | :-- | :-- | :-- |
| Algorithmic distance weight | $\lambda_{\text{alg}}$ | $[\text{dimensionless}]$ | Balances position and velocity terms in $d_{\text{alg}}$ | {prf:ref}`def-algorithmic-distance-metric` |
| Diversity interaction radius | $\epsilon_d$ | $[\text{distance}]$ | Controls softness of the pairing kernel | Section 3.1 |
| Cloning interaction radius | $\epsilon_c$ | $[\text{distance}]$ | Controls softness of the cloning companion kernel | {prf:ref}`def-cloning-companion-operator` |
| Distance regularizer | $\epsilon_{\text{dist}}$ | $[\text{distance}]$ | Prevents degeneracy of $d_j$ | Section 3.2 |
| Localization scale | $\rho$ | $[\text{distance}]$ | Chooses between global and local statistics | Section 3.3 |
| Variance patch | $\sigma_{\text{min}}$ | $[\text{dimensionless}]$ | Ensures positive denominators in Z-scores | Sections 3.3–4.2 |
| Logistic bound | $A$ | $[\text{dimensionless}]$ | Caps the rescaled channel output | Sections 3.4–4.3 |
| Positivity floor | $\eta$ | $[\text{dimensionless}]$ | Guarantees strictly positive channels | Sections 3.4–4.3 |
| Reward exponent | $\alpha$ | $[\text{dimensionless}]$ | Shapes the non-linearity of the reward channel | Section 4.3 |
| Diversity exponent | $\beta$ | $[\text{dimensionless}]$ | Controls sensitivity of the diversity channel | Section 3.4 |
| Reward velocity penalty | $c_{v\_\text{reg}}$ | $[\text{dimensionless}]$ | Penalizes high kinetic energy in $r_j$ | Section 4.1 |
| Potential function | $U(\cdot)$ | $[\text{dimensionless}]$ | Encodes the external energy landscape | Section 4.1 |
| Threshold scale | $p_{\max}$ | $[\text{dimensionless}]$ | Upper bound of the cloning threshold; sets clipping scale | Section 5.2 |
| Cloning denominator regularizer | $\epsilon_{\text{clone}}$ | $[\text{dimensionless}]$ | Prevents division by zero in $s(i \mid j, M)$ | Section 5.1 |
| Spatial jitter | $\sigma_x$ | $[\text{distance}]$ | Variance of positional perturbations during cloning | Section 6.1 |
| Kinetic friction | $\gamma$ | $[1/\text{time}]$ | BAOAB damping coefficient | Section 7.1 |
| Kinetic inverse temperature | $\beta_{\text{kin}}$ | $[\text{dimensionless}]$ | Scales the kinetic noise term | Section 7.1 |
| Time step | $\Delta t$ | $[\text{time}]$ | Temporal discretization of the kinetic operator | Section 7.1 |

The table collects the parameters that appear explicitly in the observable stack, enabling traceability between algorithmic implementation and the formal probability objects derived above.
