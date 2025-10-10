# Wasserstein Distance Convergence: Implementation Roadmap

**Purpose:** This document provides a detailed, implementation-ready roadmap for proving **exponential convergence in Wasserstein distance** for the Euclidean Gas algorithm. This is a **standalone, publishable result** that does NOT require curvature analysis or Fisher-Rao metric machinery.

**Status:** Implementation roadmap (ready to execute)

**Relationship to Existing Results:**
- [04_convergence.md](04_convergence.md): Total variation convergence via Foster-Lyapunov ✅
- [10_kl_convergence.md](10_kl_convergence/10_kl_convergence.md): KL-divergence convergence via LSI ✅
- **This document**: Wasserstein distance convergence via coupling methods (NEW)
- [16_fisher_rao_roadmap.md](16_fisher_rao_roadmap.md): Fisher-Rao convergence (future work, requires curvature)

**Key Insight:** Wasserstein convergence uses **coupling methods** and **contraction theory**, which are completely independent of the curvature/LSI/Fisher-Rao machinery. This makes it an ideal **immediate next step** for publication.

---

## 0. Executive Summary

### 0.1. What is Wasserstein Distance?

The **2-Wasserstein distance** $W_2(\mu, \nu)$ between probability measures is the optimal transport distance:

$$
W_2^2(\mu, \nu) := \inf_{\pi \in \Pi(\mu, \nu)} \int_{\mathcal{X} \times \mathcal{X}} \|x - y\|^2 \, d\pi(x, y)
$$

where $\Pi(\mu, \nu)$ is the set of couplings (joint distributions with marginals $\mu$ and $\nu$).

**Intuition:** $W_2(\mu, \nu)$ measures the "work" required to transport mass from $\mu$ to $\nu$.

**Why it matters:**
- **Geometric:** Captures spatial structure of distributions (unlike KL divergence, which is purely information-theoretic)
- **Stronger than TV:** Wasserstein convergence $\Rightarrow$ total variation convergence
- **Weaker than KL:** Can hold even when KL divergence is infinite
- **Applications:** Optimal transport, generative models (Wasserstein GANs), gradient flows

### 0.2. Why Pursue Wasserstein Convergence?

**Scientific Payoffs:**
- **Dual-metric convergence:** Complements existing KL/TV results with geometric perspective
- **Particle method analysis:** Natural metric for swarm/particle systems
- **Transport map insights:** Reveals how walkers "flow" toward the QSD
- **Robustness:** Wasserstein metric is more stable to tail behavior than KL divergence

**Technical Payoffs:**
- **Coupling proofs:** Elegant probabilistic arguments (constructive, intuitive)
- **Mean-field limit:** Essential for $N \to \infty$ analysis
- **Hessian-based diffusion:** Foundation for analyzing Adaptive Gas anisotropic noise

**Publication Impact:**
- **Standalone result:** Publishable independently (no curvature required!)
- **Completes trilogy:** TV + KL + Wasserstein = comprehensive convergence theory
- **Novel for swarm optimization:** First Wasserstein convergence proof for fitness-based particle methods
- **Opens path to:** Wasserstein gradient flow interpretation (future work)

### 0.3. Strategic Assessment

**Feasibility:** ★★★★★ **VERY HIGH**
- Coupling methods are well-established for Langevin dynamics
- No curvature analysis required
- Builds directly on existing Foster-Lyapunov framework

**Difficulty:** ★★☆☆☆ **LOW-MEDIUM**
- Technical but standard techniques
- Main challenge: handling cloning operator in coupling framework

**Time Estimate:** **4-8 weeks** for complete proof + write-up

**Payoff:** **HIGH**
- Publishable as standalone paper
- Foundational for future Fisher-Rao work (if curvature analysis succeeds)
- Immediate impact on particle method literature

**Recommendation:** ✅ **PURSUE IMMEDIATELY** — highest priority, ready to execute

---

## 1. Main Result (Target Theorem)

:::{prf:theorem} Wasserstein Convergence for the Euclidean Gas
:label: thm-wasserstein-convergence-main

For the N-particle Euclidean Gas with parameters satisfying the Foster-Lyapunov conditions of Theorem 8.1 in [04_convergence.md](04_convergence.md), there exist constants $\lambda_W > 0$ and $C_W < \infty$ such that for any initial swarm distribution $\mu_0 \in \mathcal{P}(\Sigma_N)$:

$$
W_2(\mu_t, \pi_{\text{QSD}}) \le C_W e^{-\lambda_W t} W_2(\mu_0, \pi_{\text{QSD}}) + R_{\infty}
$$

where:
- $\mu_t = \Psi_{\text{total}}^t(\mu_0)$ is the distribution after $t$ iterations
- $\pi_{\text{QSD}}$ is the unique quasi-stationary distribution
- $\lambda_W = \min(\gamma/2, \kappa_{\text{conf}}/4, \kappa_{\text{clone}})$ is the contraction rate
- $R_{\infty} = O(\delta)$ is the asymptotic bias from cloning noise
- $C_W = O(1)$ is a problem-dependent constant

**Parameter dependencies (explicit):**
- $\lambda_W$ increases with friction $\gamma$ and potential curvature $\kappa_{\text{conf}}$
- $R_{\infty}$ increases with cloning noise $\delta$ but can be made arbitrarily small
- Convergence holds for any swarm size $N \ge 2$

**Geometric interpretation:** The swarm distribution contracts toward the QSD at an exponential rate measured by the Wasserstein metric, with the rate determined by the Langevin friction and the confining potential.
:::

---

## 2. Proof Strategy Overview

The proof decomposes the total operator $\Psi_{\text{total}} = \Psi_{\text{kin}} \circ \Psi_{\text{clone}}$ and establishes Wasserstein contraction for each component separately, then combines them.

### 2.1. Three-Stage Proof Architecture

**Stage 1: Kinetic Operator Wasserstein Contraction**
- **Tool:** Synchronous coupling (reflection coupling for non-convex potential)
- **Result:** $W_2(\Psi_{\text{kin},*} \mu, \Psi_{\text{kin},*} \nu) \le e^{-\lambda_{\text{kin}} \tau} W_2(\mu, \nu)$
- **Challenge:** Handle non-uniformly-convex confining potential via localization
- **Section:** 3

**Stage 2: Cloning Operator Wasserstein Bound**
- **Tool:** Fitness-coupling (match walkers by fitness rank, bound displacement)
- **Result:** $W_2(\Psi_{\text{clone},*} \mu, \Psi_{\text{clone},*} \nu) \le (1 - \kappa_{\text{clone}}) W_2(\mu, \nu) + C_{\delta} \delta$
- **Challenge:** Cloning is non-local (discrete jumps); need careful coupling construction
- **Section:** 4

**Stage 3: Composition and Convergence to QSD**
- **Tool:** Seesaw mechanism (kinetic contracts where cloning expands, and vice versa)
- **Result:** Combined operator contracts in Wasserstein distance
- **Challenge:** Handle expansion from cloning noise via absorbing bias term
- **Section:** 5

### 2.2. Why This Works Without Curvature

**Key observation:** Wasserstein contraction proofs use **coupling methods**, not curvature bounds.

**Coupling = constructive proof:**
1. Given two distributions $\mu, \nu$, construct a **joint process** $(X_t, Y_t)$ such that:
   - $X_t \sim \Psi_t \mu$ and $Y_t \sim \Psi_t \nu$ (correct marginals)
   - $\mathbb{E}[\|X_t - Y_t\|^2]$ decreases over time (contraction)

2. By definition of Wasserstein distance:

$$
W_2^2(\Psi_t \mu, \Psi_t \nu) \le \mathbb{E}[\|X_t - Y_t\|^2]
$$

3. If the coupling contracts, so does Wasserstein distance.

**Curvature is NOT needed:**
- Curvature bounds are one sufficient condition for contraction
- But coupling methods work directly without curvature assumptions
- We only need: Lipschitz drift + friction + controlled noise

**What we DO need:**
- ✅ Coercive potential (already have: Axiom {prf:ref}`axiom-confining-potential`)
- ✅ Friction in kinetic operator (already have: parameter $\gamma > 0$)
- ✅ Bounded cloning displacement (already have: finite noise $\delta$)

---

## 3. Stage 1: Kinetic Operator Wasserstein Contraction

### 3.1. Strategy

**Goal:** Prove that $\Psi_{\text{kin}}(\tau)$ contracts Wasserstein distance between any two distributions.

**Approach:** Synchronous coupling for Langevin dynamics.

:::{prf:definition} Synchronous Coupling for Langevin
:label: def-synchronous-coupling

Given two initial conditions $(x_0, v_0)$ and $(x_0', v_0')$, evolve them using the **same Brownian motion**:

$$
\begin{aligned}
dx_t &= v_t \, dt, \quad dv_t = F(x_t) \, dt - \gamma v_t \, dt + \sigma_v \, dW_t \\
dx_t' &= v_t' \, dt, \quad dv_t' = F(x_t') \, dt - \gamma v_t' \, dt + \sigma_v \, dW_t
\end{aligned}
$$

The difference process $\Delta_t := (x_t - x_t', v_t - v_t')$ evolves as:

$$
d\Delta_x = \Delta_v \, dt, \quad d\Delta_v = (F(x_t) - F(x_t')) \, dt - \gamma \Delta_v \, dt
$$

(noise cancels!)
:::

### 3.2. Contraction Analysis

:::{prf:lemma} Kinetic Operator Wasserstein Contraction (Convex Case)
:label: lem-kinetic-wasserstein-convex

If the confining potential $U$ is $\kappa$-convex (i.e., $\text{Hess}(U) \succeq \kappa I_d$), then:

$$
W_2(\Psi_{\text{kin}}(\tau)_* \mu, \Psi_{\text{kin}}(\tau)_* \nu) \le e^{-\lambda \tau} W_2(\mu, \nu)
$$

where $\lambda = \min(\gamma, \kappa)$.
:::

:::{prf:proof}
**Step 1:** Use synchronous coupling. The difference dynamics are:

$$
\frac{d}{dt} \begin{pmatrix} \Delta_x \\ \Delta_v \end{pmatrix} = \begin{pmatrix} 0 & I_d \\ -\text{Hess}(U(\xi_t)) & -\gamma I_d \end{pmatrix} \begin{pmatrix} \Delta_x \\ \Delta_v \end{pmatrix}
$$

where $\xi_t$ is on the line segment between $x_t$ and $x_t'$ (mean value theorem for $F(x) - F(x')$).

**Step 2:** Define the Lyapunov function:

$$
V(\Delta) := \|\Delta_v\|^2 + \kappa \|\Delta_x\|^2 + 2\gamma \langle \Delta_x, \Delta_v \rangle
$$

**Step 3:** Compute the time derivative:

$$
\frac{dV}{dt} = -2\gamma \|\Delta_v\|^2 - 2\kappa \langle \Delta_x, F(x_t) - F(x_t') \rangle + 2\gamma \|\Delta_v\|^2 - 2\gamma^2 \langle \Delta_x, \Delta_v \rangle
$$

By convexity: $\langle \Delta_x, F(x_t) - F(x_t') \rangle \ge \kappa \|\Delta_x\|^2$.

After algebra: $\frac{dV}{dt} \le -2\lambda V$ where $\lambda = \min(\gamma, \kappa)$.

**Step 4:** By Grönwall's inequality:

$$
V(\Delta_\tau) \le e^{-2\lambda\tau} V(\Delta_0)
$$

Since $V(\Delta) \sim \|\Delta_x\|^2 + \|\Delta_v\|^2$ (by equivalence of norms), this implies:

$$
\mathbb{E}[\|\Delta_\tau\|^2] \le e^{-2\lambda\tau} \mathbb{E}[\|\Delta_0\|^2]
$$

By definition of Wasserstein distance:

$$
W_2^2(\Psi_{\text{kin}}(\tau)_* \mu, \Psi_{\text{kin}}(\tau)_* \nu) \le \mathbb{E}[\|\Delta_\tau\|^2] \le e^{-2\lambda\tau} W_2^2(\mu, \nu)
$$

:::

### 3.3. Extension to Coercive (Non-Convex) Potentials

**Problem:** The Euclidean Gas confining potential is **coercive** but **not uniformly convex**. In the interior of $\mathcal{X}_{\text{valid}}$, the Hessian may have zero or negative eigenvalues.

**Solution:** Localization + reflection coupling.

:::{prf:theorem} Kinetic Operator Wasserstein Contraction (Coercive Case)
:label: thm-kinetic-wasserstein-coercive

For the Euclidean Gas kinetic operator with potential $U$ satisfying Axiom {prf:ref}`axiom-confining-potential` (coercive), there exists a constant $\lambda_{\text{kin}} > 0$ such that:

$$
W_2(\Psi_{\text{kin}}(\tau)_* \mu, \pi_{\text{kin}}) \le e^{-\lambda_{\text{kin}} \tau} W_2(\mu, \pi_{\text{kin}})
$$

where $\pi_{\text{kin}}$ is the stationary distribution of $\Psi_{\text{kin}}$.

**Explicit constant:** $\lambda_{\text{kin}} = \min(\gamma/2, \alpha_U/R_U)$ where $\alpha_U, R_U$ are from the coercivity condition.
:::

:::{prf:proof}
**Proof strategy (sketch):**

1. **Localization:** Partition $\mathcal{X}_{\text{valid}}$ into:
   - **Interior region** $\mathcal{X}_{\text{int}}$: Where potential is approximately flat
   - **Boundary region** $\mathcal{X}_{\text{bdy}}$: Where coercivity dominates

2. **Interior coupling:** Use **reflection coupling** (Eberle 2016, Majka 2020):
   - Instead of synchronous coupling, reflect Brownian motions across the hyperplane perpendicular to $x_t - x_t'$
   - This creates contraction even without convexity (via friction)

3. **Boundary coupling:** Use standard synchronous coupling:
   - Coercivity ensures $\langle x - x', F(x) - F(x') \rangle \ge \alpha_U \|x - x'\|^2$ in $\mathcal{X}_{\text{bdy}}$
   - Strong contraction in this region

4. **Patchwork:** Combine the two couplings using a smooth transition
   - Contraction holds globally by weighted average

**Reference:** Majka (2020), "Transportation inequalities for non-globally dissipative SDEs with jumps via Malliavin calculus and coupling"
:::

**Implementation note:** The detailed proof requires ~10 pages of technical analysis. For the roadmap, we assume this result as a **black box** from the coupling literature.

### 3.4. Required Work for Stage 1

**Task 1.1 (1-2 weeks):** Literature review and adaptation
- Study Eberle (2016) reflection coupling for non-convex potentials
- Study Majka (2020) localization techniques
- Adapt to underdamped Langevin (position + velocity)

**Task 1.2 (2-3 weeks):** Proof development
- Define localization regions for the specific confining potential $U$
- Construct reflection coupling in interior
- Prove contraction in each region separately
- Combine using patchwork argument

**Task 1.3 (1 week):** Explicit constants
- Extract $\lambda_{\text{kin}}$ in terms of $\gamma, \alpha_U, R_U$
- Compute asymptotic behavior as $\gamma \to \infty$ (overdamped limit)

**Deliverable:** Theorem {prf:ref}`thm-kinetic-wasserstein-coercive` with complete proof

---

## 4. Stage 2: Cloning Operator Wasserstein Bound

### 4.1. Strategy

**Goal:** Bound the Wasserstein distance after applying the cloning operator.

**Challenge:** Cloning is a **non-local** operator—walkers can jump arbitrary distances. This breaks the usual coupling constructions.

**Key insight:** Cloning has two components:
1. **Selection:** Choose a companion walker based on fitness
2. **Displacement:** Move to companion's position + Gaussian noise

The **selection** step contracts (moves toward high-fitness regions), but the **noise** step expands.

### 4.2. Fitness Coupling

:::{prf:definition} Fitness Coupling for Cloning
:label: def-fitness-coupling

Given two swarms $\mathcal{S} = (w_1, \ldots, w_N)$ and $\mathcal{S}' = (w_1', \ldots, w_N')$, construct a coupling as follows:

1. **Sort by fitness:** Reindex both swarms so that:

$$
V_{\text{fit},1} \ge V_{\text{fit},2} \ge \cdots \ge V_{\text{fit},N}, \quad V_{\text{fit},1}' \ge V_{\text{fit},2}' \ge \cdots \ge V_{\text{fit},N}'
$$

2. **Match by rank:** Pair walker $i$ in $\mathcal{S}$ with walker $i$ in $\mathcal{S}'$

3. **Shared randomness:** Use the same random threshold $T_i$ and companion choice for paired walkers

4. **Shared Gaussian noise:** If both clone, use the same noise $\xi_i \sim \mathcal{N}(0, \delta^2 I_d)$
:::

**Intuition:** Walkers with similar fitness undergo similar cloning decisions, keeping them "coupled."

### 4.3. Wasserstein Bound for Cloning

:::{prf:lemma} Cloning Operator Wasserstein Bound
:label: lem-cloning-wasserstein-bound

Under the fitness coupling {prf:ref}`def-fitness-coupling`, the cloning operator satisfies:

$$
W_2^2(\Psi_{\text{clone},*} \mu, \Psi_{\text{clone},*} \nu) \le (1 - \kappa_{\text{clone}}) W_2^2(\mu, \nu) + C_N \delta^2
$$

where:
- $\kappa_{\text{clone}} \in (0, 1)$ is the contraction rate from fitness-based selection
- $C_N = O(N)$ is the number of walkers
- $\delta^2$ is the cloning noise variance
:::

:::{prf:proof}
**Step 1: Decompose displacement**

For paired walkers $(w_i, w_i')$ that both clone:

$$
\begin{aligned}
x_i^{\text{new}} &= x_{c(i)} + \xi_i \\
x_i'^{\text{new}} &= x_{c(i)}' + \xi_i
\end{aligned}
$$

The post-cloning distance is:

$$
\|x_i^{\text{new}} - x_i'^{\text{new}}\|^2 = \|x_{c(i)} - x_{c(i)}'\|^2
$$

(noise cancels!)

**Step 2: Companion distance bound**

Since companions are chosen by fitness rank and fitness is Lipschitz in position (by {prf:ref}`axiom-confining-potential`), we have:

$$
\|x_{c(i)} - x_{c(i)}'\| \le L_{\text{fit}} \|x_i - x_i'\|
$$

where $L_{\text{fit}} < 1$ is a contraction factor (high-fitness walkers are "attracted" toward each other).

**Step 3: Persist vs. clone cases**

- **Both persist:** $\|x_i^{\text{new}} - x_i'^{\text{new}}\|^2 = \|x_i - x_i'\|^2$ (no change)
- **Both clone:** $\|x_i^{\text{new}} - x_i'^{\text{new}}\|^2 \le L_{\text{fit}}^2 \|x_i - x_i'\|^2$ (contraction)
- **One clones, one persists:** Worst case, bounded by $\max(\|x_i - x_i'\|, \delta)$

**Step 4: Average over swarm**

$$
\begin{aligned}
W_2^2(\Psi_{\text{clone},*} \mu, \Psi_{\text{clone},*} \nu) &\le \mathbb{E}\left[\frac{1}{N} \sum_{i=1}^N \|x_i^{\text{new}} - x_i'^{\text{new}}\|^2\right] \\
&\le (1 - p_{\text{clone}} (1 - L_{\text{fit}}^2)) W_2^2(\mu, \nu) + O(p_{\text{mismatch}} \delta^2)
\end{aligned}
$$

where $p_{\text{clone}}$ is the cloning probability and $p_{\text{mismatch}}$ is the probability of mismatched cloning decisions.

Defining $\kappa_{\text{clone}} := p_{\text{clone}} (1 - L_{\text{fit}}^2)$ yields the result.
:::

**Important:** The $C_N \delta^2$ term is an **asymptotic bias**—it prevents exact convergence to $W_2 = 0$ but can be made arbitrarily small by choosing small $\delta$.

### 4.4. Required Work for Stage 2

**Task 2.1 (1 week):** Fitness Lipschitz constant
- Prove that fitness potential $V_{\text{fit}}$ is Lipschitz in position
- Extract explicit $L_{\text{fit}}$ in terms of algorithm parameters

**Task 2.2 (1-2 weeks):** Coupling construction
- Formalize the fitness coupling definition
- Prove that it is indeed a valid coupling (correct marginals)
- Handle edge cases (all walkers dead, ties in fitness)

**Task 2.3 (2 weeks):** Wasserstein bound proof
- Prove Lemma {prf:ref}`lem-cloning-wasserstein-bound` with detailed case analysis
- Compute explicit $\kappa_{\text{clone}}$ and $C_N$
- Show that $\kappa_{\text{clone}} > 0$ under algorithm parameter constraints

**Deliverable:** Lemma {prf:ref}`lem-cloning-wasserstein-bound` with complete proof

---

## 5. Stage 3: Composition and Main Theorem

### 5.1. Strategy

**Goal:** Combine the kinetic and cloning results to prove exponential convergence to $\pi_{\text{QSD}}$.

**Key idea:** Use the **seesaw mechanism** from [04_convergence.md](04_convergence.md)—where one operator expands, the other contracts.

### 5.2. Seesaw Composition

:::{prf:theorem} Wasserstein Convergence via Seesaw Mechanism
:label: thm-wasserstein-seesaw

Under the parameter conditions:

$$
\lambda_{\text{kin}} \tau > \log(1/(1 - \kappa_{\text{clone}}))
$$

(kinetic contracts faster than cloning expands), the composed operator satisfies:

$$
W_2(\Psi_{\text{total},*} \mu, \pi_{\text{QSD}}) \le \rho W_2(\mu, \pi_{\text{QSD}}) + R_{\infty}
$$

where:
- $\rho = e^{-\lambda_{\text{kin}} \tau} \sqrt{1 - \kappa_{\text{clone}}} < 1$ is the net contraction rate
- $R_{\infty} = O(\delta)$ is the asymptotic bias from cloning noise

**Iterating:**

$$
W_2(\mu_t, \pi_{\text{QSD}}) \le \rho^t W_2(\mu_0, \pi_{\text{QSD}}) + \frac{R_{\infty}}{1 - \rho}
$$

Exponential convergence to a $\delta$-neighborhood of $\pi_{\text{QSD}}$.
:::

:::{prf:proof}
**Step 1:** Apply cloning first:

$$
W_2(\Psi_{\text{clone},*} \mu, \Psi_{\text{clone},*} \pi_{\text{QSD}}) \le \sqrt{1 - \kappa_{\text{clone}}} W_2(\mu, \pi_{\text{QSD}}) + \sqrt{C_N} \delta
$$

**Step 2:** Apply kinetic operator:

$$
W_2(\Psi_{\text{kin},*} \Psi_{\text{clone},*} \mu, \Psi_{\text{kin},*} \Psi_{\text{clone},*} \pi_{\text{QSD}}) \le e^{-\lambda_{\text{kin}} \tau} W_2(\Psi_{\text{clone},*} \mu, \Psi_{\text{clone},*} \pi_{\text{QSD}})
$$

**Step 3:** Note that $\pi_{\text{QSD}}$ is invariant under $\Psi_{\text{total}} = \Psi_{\text{kin}} \circ \Psi_{\text{clone}}$, so:

$$
\Psi_{\text{kin},*} \Psi_{\text{clone},*} \pi_{\text{QSD}} = \pi_{\text{QSD}}
$$

**Step 4:** Combine:

$$
W_2(\Psi_{\text{total},*} \mu, \pi_{\text{QSD}}) \le e^{-\lambda_{\text{kin}} \tau} \sqrt{1 - \kappa_{\text{clone}}} W_2(\mu, \pi_{\text{QSD}}) + e^{-\lambda_{\text{kin}} \tau} \sqrt{C_N} \delta
$$

Define $\rho := e^{-\lambda_{\text{kin}} \tau} \sqrt{1 - \kappa_{\text{clone}}}$ and $R_{\infty} := e^{-\lambda_{\text{kin}} \tau} \sqrt{C_N} \delta$.

By assumption, $\rho < 1$ (seesaw condition).

**Step 5:** Iterate $t$ times:

$$
W_2(\mu_t, \pi_{\text{QSD}}) \le \rho^t W_2(\mu_0, \pi_{\text{QSD}}) + R_{\infty} \sum_{k=0}^{t-1} \rho^k \le \rho^t W_2(\mu_0, \pi_{\text{QSD}}) + \frac{R_{\infty}}{1 - \rho}
$$

Exponential convergence to the $\frac{R_{\infty}}{1 - \rho}$-neighborhood of $\pi_{\text{QSD}}$.
:::

### 5.3. Removing the Asymptotic Bias

**Observation:** The bias term $R_{\infty} = O(\delta)$ comes from cloning noise. As $\delta \to 0$, the bias vanishes, but cloning becomes less effective.

**Strategy:** For the **main theorem**, assume $\delta$ is chosen optimally to balance contraction and noise.

:::{prf:theorem} Exact Wasserstein Convergence (Main Result)
:label: thm-wasserstein-exact

For the Euclidean Gas with cloning noise $\delta$ satisfying:

$$
\delta \le \delta_* := \frac{\epsilon (1 - \rho)}{e^{-\lambda_{\text{kin}} \tau} \sqrt{C_N}}
$$

for any desired tolerance $\epsilon > 0$, we have:

$$
W_2(\mu_t, \pi_{\text{QSD}}) \le C_W e^{-\lambda_W t} W_2(\mu_0, \pi_{\text{QSD}}) + \epsilon
$$

where $\lambda_W = -\log(\rho)$ and $C_W = 1$.

**Interpretation:** By choosing $\delta$ sufficiently small (but still positive), we can achieve exponential convergence to within any desired $\epsilon$ of the QSD.

**Exact convergence:** In the limit $\delta \to 0$, we have exact exponential convergence (no bias).
:::

### 5.4. Required Work for Stage 3

**Task 3.1 (1 week):** Seesaw parameter condition
- Verify that Foster-Lyapunov parameters satisfy $\lambda_{\text{kin}} \tau > \log(1/(1 - \kappa_{\text{clone}}))$
- Compute explicit $\rho$ and $\lambda_W$ in terms of $\gamma, \kappa_{\text{conf}}, \kappa_{\text{clone}}$

**Task 3.2 (1 week):** Bias analysis
- Prove Theorem {prf:ref}`thm-wasserstein-seesaw` with detailed iteration argument
- Compute $R_{\infty}$ and verify it scales as $O(\delta)$

**Task 3.3 (1 week):** Main theorem proof
- Prove Theorem {prf:ref}`thm-wasserstein-exact`
- Derive optimal $\delta_*$ choice
- Show consistency with Foster-Lyapunov parameter regime

**Deliverable:** Main Theorem {prf:ref}`thm-wasserstein-convergence-main` with complete proof

---

## 6. Implementation Timeline

### 6.1. Phase-by-Phase Breakdown

**Phase 1 (Weeks 1-3): Kinetic Operator Contraction**
- **Week 1:** Literature review (Eberle, Majka, coupling methods)
- **Week 2-3:** Adapt reflection coupling to underdamped Langevin with coercive potential
- **Deliverable:** Theorem {prf:ref}`thm-kinetic-wasserstein-coercive`

**Phase 2 (Weeks 4-6): Cloning Operator Coupling**
- **Week 4:** Fitness Lipschitz constant and coupling construction
- **Week 5:** Wasserstein bound proof (case analysis)
- **Week 6:** Explicit constant computation
- **Deliverable:** Lemma {prf:ref}`lem-cloning-wasserstein-bound`

**Phase 3 (Weeks 7-8): Composition and Main Result**
- **Week 7:** Seesaw mechanism and iteration analysis
- **Week 8:** Main theorem proof and parameter optimization
- **Deliverable:** Theorem {prf:ref}`thm-wasserstein-convergence-main`

**Phase 4 (Weeks 9-10): Write-Up and Review**
- **Week 9:** Document structure, proofs, and examples
- **Week 10:** Gemini review, revisions, and finalization
- **Deliverable:** Publication-ready manuscript section

### 6.2. Parallel Work Opportunities

Tasks that can be done in parallel:
- **Task 1.1 (literature)** can start immediately
- **Task 2.1 (fitness Lipschitz)** can be done independently of Task 1.2
- **Task 3.1 (parameter conditions)** can be checked against existing Foster-Lyapunov results early

### 6.3. Risk Mitigation

**Risk 1:** Reflection coupling proof too technical
- **Mitigation:** Cite Majka (2020) as a black box; adapt conclusion only
- **Fallback:** Use weaker synchronous coupling with localization (longer proof but standard)

**Risk 2:** Fitness coupling doesn't preserve marginals
- **Mitigation:** Use maximal coupling as fallback (less tight bound but guaranteed valid)
- **Impact:** May increase constant $C_N$ but preserves exponential convergence

**Risk 3:** Seesaw condition fails for some parameter regimes
- **Mitigation:** Strengthen kinetic contraction (increase $\gamma$ or $\tau$)
- **Impact:** Trade-off between convergence rate and computational cost per iteration

---

## 7. Success Criteria

### 7.1. Minimum Success (Publishable)

- [ ] **Theorem {prf:ref}`thm-wasserstein-convergence-main`** with complete proof
- [ ] Explicit contraction rate $\lambda_W$ in terms of algorithm parameters
- [ ] Numerical validation on toy example (1D or 2D state space)
- [ ] All proofs pass Gemini mathematical review
- [ ] **Outcome:** Standalone paper on Wasserstein convergence for particle swarm optimization

### 7.2. Target Success (High-Impact Paper)

- [ ] All minimum success criteria
- [ ] **Coupling diagrams:** Visual illustrations of fitness coupling and seesaw mechanism
- [ ] **Numerical experiments:** Convergence rate plots for various parameter regimes
- [ ] **Comparison with KL/TV:** Show Wasserstein convergence is independent (can hold when KL diverges)
- [ ] **Mean-field limit:** Discussion of $N \to \infty$ behavior
- [ ] **Outcome:** High-quality paper for optimization or probability journal

### 7.3. Stretch Goals (Foundational Contribution)

- [ ] All target success criteria
- [ ] **Adaptive Gas extension:** Prove Wasserstein convergence for Hessian-based anisotropic diffusion
- [ ] **Optimal transport perspective:** Interpret cloning operator as Wasserstein gradient flow
- [ ] **Connection to Fisher-Rao:** If curvature analysis succeeds, unify with [16_fisher_rao_roadmap.md](16_fisher_rao_roadmap.md)
- [ ] **Outcome:** Major contribution bridging particle methods, optimal transport, and information geometry

---

## 8. Publication Strategy

### 8.1. Target Venues

**Tier 1 (Optimization):**
- *Mathematical Programming* (Series A or B)
- *SIAM Journal on Optimization*
- *Journal of Optimization Theory and Applications*

**Tier 1 (Probability):**
- *Annals of Applied Probability*
- *Probability Theory and Related Fields*
- *Electronic Journal of Probability*

**Tier 2 (Interdisciplinary):**
- *Journal of Machine Learning Research*
- *IEEE Transactions on Automatic Control*
- *Stochastic Processes and their Applications*

### 8.2. Manuscript Structure

**Title:** "Wasserstein Distance Convergence for Fitness-Based Particle Swarm Optimization"

**Abstract (150 words):**
- Problem: Convergence analysis for particle swarm methods
- Method: Coupling techniques for kinetic + cloning operators
- Result: Exponential Wasserstein convergence with explicit rates
- Impact: First geometric convergence proof for fitness-based swarms

**Sections:**
1. Introduction (4 pages)
   - Particle swarm optimization background
   - Euclidean Gas algorithm description
   - Main result statement
   - Literature review (Langevin dynamics, optimal transport, coupling methods)

2. Preliminaries (3 pages)
   - Wasserstein distance definition
   - Coupling theory basics
   - Euclidean Gas framework axioms

3. Kinetic Operator Contraction (6 pages)
   - Reflection coupling for non-convex potentials
   - Theorem {prf:ref}`thm-kinetic-wasserstein-coercive`
   - Explicit constant derivation

4. Cloning Operator Coupling (5 pages)
   - Fitness coupling construction
   - Lemma {prf:ref}`lem-cloning-wasserstein-bound`
   - Asymptotic bias analysis

5. Main Convergence Result (4 pages)
   - Seesaw composition theorem
   - Theorem {prf:ref}`thm-wasserstein-convergence-main`
   - Parameter optimization

6. Numerical Experiments (3 pages)
   - Toy examples validating convergence rates
   - Parameter sensitivity analysis

7. Conclusion and Future Work (2 pages)
   - Summary of contributions
   - Extensions to Adaptive Gas
   - Connection to Fisher-Rao convergence

**Total:** ~27 pages + references + appendix

### 8.3. Complementary Results

This Wasserstein paper **complements** the existing convergence trilogy:
- [04_convergence.md](04_convergence.md): **Total variation** convergence (weakest, easiest to prove)
- [10_kl_convergence.md](10_kl_convergence/10_kl_convergence.md): **KL divergence** convergence (information-theoretic, requires LSI)
- **This work:** **Wasserstein distance** convergence (geometric, requires coupling)

**Value proposition:** Each metric captures different aspects of convergence—together they provide a comprehensive understanding.

---

## 9. Open Questions and Future Directions

### 9.1. Immediate Extensions

**Question 1:** Can we prove Wasserstein convergence for the **Adaptive Gas** with Hessian-based diffusion?
- **Challenge:** Anisotropic noise breaks synchronous coupling
- **Approach:** Use non-Euclidean Wasserstein distance with Riemannian metric

**Question 2:** What is the **optimal cloning noise** $\delta_*$ that balances contraction and bias?
- **Challenge:** Trade-off between $\kappa_{\text{clone}}$ and $R_{\infty}$
- **Approach:** Variational optimization over $\delta$

**Question 3:** Does Wasserstein convergence hold in the **mean-field limit** $N \to \infty$?
- **Challenge:** Fitness coupling may break down for infinite swarms
- **Approach:** Propagation of chaos techniques from interacting particle systems

### 9.2. Connection to Fisher-Rao Roadmap

If the curvature investigation in [16_fisher_rao_roadmap.md](16_fisher_rao_roadmap.md) succeeds:
- Use Wasserstein convergence + HWI inequality → Fisher-Rao convergence
- This Wasserstein result becomes a **key building block**

If curvature analysis fails:
- Wasserstein convergence remains a **standalone publishable result**
- No dependency on curvature machinery

**Strategic value:** Wasserstein is useful **regardless** of Fisher-Rao outcome.

---

## 10. Conclusion

### 10.1. Summary

This roadmap provides a **complete, implementation-ready plan** for proving Wasserstein distance convergence for the Euclidean Gas algorithm.

**Key advantages:**
- ✅ **Independent of curvature:** Can be pursued immediately
- ✅ **Well-established techniques:** Coupling methods are standard
- ✅ **Publishable standalone:** Does not require other results
- ✅ **Complements existing work:** Fills gap between TV and KL convergence
- ✅ **Foundation for future work:** Enables Fisher-Rao analysis (if curvature holds)

**Timeline:** 8-10 weeks from start to publication-ready manuscript

**Recommendation:** **HIGHEST PRIORITY** — begin immediately.

---

## References

**Coupling Methods:**
1. Eberle, A. (2016). "Reflection couplings and contraction rates for diffusions." *Probability Theory and Related Fields*, 166(3-4), 851-886.
2. Majka, M. B. (2020). "Transportation inequalities for non-globally dissipative SDEs with jumps via Malliavin calculus and coupling." *Annals of Probability*, 48(3), 1103-1133.
3. Lindvall, T., & Rogers, L. C. G. (1986). "Coupling of multidimensional diffusions by reflection." *Annals of Probability*, 14(3), 860-872.

**Wasserstein Distance:**
4. Villani, C. (2009). *Optimal Transport: Old and New*. Springer.
5. Ambrosio, L., Gigli, N., & Savaré, G. (2008). *Gradient Flows in Metric Spaces and in the Space of Probability Measures*. Birkhäuser.

**Particle Methods:**
6. Del Moral, P. (2004). *Feynman-Kac Formulae: Genealogical and Interacting Particle Systems with Applications*. Springer.
7. Sznitman, A. S. (1991). "Topics in propagation of chaos." *École d'Été de Probabilités de Saint-Flour XIX*, 165-251.

**Framework Documents:**
- [01_fragile_gas_framework.md](01_fragile_gas_framework.md) — Foundational axioms
- [02_euclidean_gas.md](02_euclidean_gas.md) — Euclidean Gas specification
- [04_convergence.md](04_convergence.md) — Total variation convergence
- [10_kl_convergence.md](10_kl_convergence/10_kl_convergence.md) — KL-divergence convergence
- [16_fisher_rao_roadmap.md](16_fisher_rao_roadmap.md) — Fisher-Rao convergence (future work)
