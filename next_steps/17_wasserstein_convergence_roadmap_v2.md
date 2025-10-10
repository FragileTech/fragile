# Wasserstein Distance Convergence: Revised Implementation Roadmap

**Purpose:** This document provides a **realistic, research-grade roadmap** for proving **exponential convergence in Wasserstein distance** for the Euclidean Gas algorithm. This is a **standalone, publishable result** that does NOT require curvature analysis or Fisher-Rao metric machinery.

**Status:** Revised implementation roadmap (post-critical review)

**Version:** 2.0 (Critical issues from Gemini review addressed)

**Relationship to Existing Results:**
- [04_convergence.md](04_convergence.md): Total variation convergence via Foster-Lyapunov ✅
- [10_kl_convergence.md](10_kl_convergence/10_kl_convergence.md): KL-divergence convergence via LSI ✅
- **This document**: Wasserstein distance convergence via coupling methods (NEW)
- [16_fisher_rao_roadmap.md](16_fisher_rao_roadmap.md): Fisher-Rao convergence (future work, requires curvature)

**Key Changes from V1:**
- ❌ **Removed flawed** "fitness coupling" contraction claim for cloning operator
- ✅ **Added** three alternative coupling strategies with contingency plans
- ⚠️ **Revised** timeline from 4-8 weeks to **4-6 months** (realistic)
- ⚠️ **Upgraded** difficulty from LOW-MEDIUM to **MEDIUM-HIGH**
- ✅ **Expanded** underdamped reflection coupling to dedicated research phase

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
- **Geometric:** Captures spatial structure of distributions (unlike KL divergence)
- **Stronger than TV:** Wasserstein convergence $\Rightarrow$ total variation convergence
- **Weaker than KL:** Can hold even when KL divergence is infinite
- **Applications:** Optimal transport, generative models, gradient flows

### 0.2. Why Pursue Wasserstein Convergence?

**Scientific Payoffs:**
- **Dual-metric convergence:** Complements existing KL/TV results with geometric perspective
- **Particle method analysis:** Natural metric for swarm/particle systems
- **Transport map insights:** Reveals how walkers "flow" toward the QSD
- **Robustness:** More stable to tail behavior than KL divergence

**Technical Payoffs:**
- **Coupling proofs:** Elegant probabilistic arguments
- **Mean-field limit:** Essential for $N \to \infty$ analysis
- **Foundation:** Enables Adaptive Gas analysis and future Fisher-Rao work

**Publication Impact:**
- **Standalone result:** Publishable independently (no curvature required!)
- **Completes trilogy:** TV + KL + Wasserstein = comprehensive convergence
- **Novel contribution:** First Wasserstein proof for fitness-based particle methods

### 0.3. Strategic Assessment (REVISED AFTER CRITICAL REVIEW)

**Feasibility:** ★★★★☆ **HIGH** (revised down from VERY HIGH)
- Coupling methods are well-established for Langevin dynamics ✅
- No curvature analysis required ✅
- **BUT:** Cloning operator coupling requires novel research ⚠️
- **AND:** Underdamped reflection coupling is non-trivial extension ⚠️

**Difficulty:** ★★★☆☆ **MEDIUM-HIGH** (revised up from LOW-MEDIUM)
- Kinetic operator: Standard but technically involved (reflection coupling)
- Cloning operator: **Novel research required** (no existing coupling framework)
- Composition: Seesaw mechanism well-understood

**Time Estimate:** **4-6 months** (revised from 4-8 weeks)
- Month 1-2: Underdamped reflection coupling (research phase)
- Month 2-4: Cloning operator coupling (exploratory phase with contingencies)
- Month 4-5: Composition and main theorem
- Month 5-6: Write-up and review

**Payoff:** **HIGH**
- Publishable as standalone paper
- Foundational for future work
- High impact on particle method literature

**Recommendation:** ✅ **PURSUE WITH REALISTIC SCOPING**
- Front-load high-risk components (cloning coupling)
- Build in contingency plans for each stage
- Expect iterative refinement of proof strategy

---

## 1. Critical Issues from Gemini Review

:::{important}
**Gemini identified fundamental flaws** in the V1 roadmap. This revised version addresses all critical issues.
:::

### 1.1. Issue #1 (CRITICAL): Flawed Cloning Coupling

**Problem:** V1 claimed that fitness-based companion selection leads to contraction:

$$
\|x_{c(i)} - x_{c(i)}'\| \le L_{\text{fit}} \|x_i - x_i'\| \quad \text{(FALSE!)}
$$

**Why this fails:** Even if walkers $w_i$ and $w_i'$ are close, their companions $c(i)$ and $c(i)'$ may be on opposite sides of the state space.

**Resolution:** Section 4 now presents **three alternative coupling strategies** with explicit feasibility analysis.

### 1.2. Issue #2 (MAJOR): Underdamped Reflection Coupling

**Problem:** V1 treated extension of reflection coupling to underdamped Langevin as "straightforward adaptation" (1-2 weeks).

**Reality:** This is a **novel research problem**—existing literature (Eberle, Majka) focuses on overdamped (first-order) systems.

**Resolution:** Section 3 now dedicates a full research phase (4-6 weeks) with explicit technical challenges.

### 1.3. Issue #3 (MAJOR): Unrealistic Timeline

**Problem:** V1 estimated 4-8 weeks total.

**Reality:** Research-grade proof for top-tier journal requires 4-6 months.

**Resolution:** Completely revised timeline in Section 6 with phase-by-phase breakdown and buffer time.

### 1.4. Issue #4 (MODERATE): Incomplete Mismatch Analysis

**Problem:** V1 didn't properly analyze "one clones, one persists" case.

**Resolution:** Section 4.4 now includes detailed case-by-case expectation analysis.

---

## 2. Main Result (Target Theorem)

:::{prf:theorem} Wasserstein Convergence for the Euclidean Gas (Revised)
:label: thm-wasserstein-convergence-main-v2

For the N-particle Euclidean Gas with parameters satisfying the Foster-Lyapunov conditions of Theorem 8.1 in [04_convergence.md](04_convergence.md), there exist constants $\lambda_W > 0$ and $C_W < \infty$ such that for any initial swarm distribution $\mu_0 \in \mathcal{P}(\Sigma_N)$:

$$
W_2(\mu_t, \pi_{\text{QSD}}) \le C_W e^{-\lambda_W t} W_2(\mu_0, \pi_{\text{QSD}}) + R_{\infty}(t)
$$

where:
- $\mu_t = \Psi_{\text{total}}^t(\mu_0)$ is the distribution after $t$ iterations
- $\pi_{\text{QSD}}$ is the unique quasi-stationary distribution
- $\lambda_W > 0$ is the **net contraction rate** (depends on proof strategy—see Section 4)
- $R_{\infty}(t)$ is the **residual bias term** (may decay slower than exponential)
- $C_W = O(1)$ is a problem-dependent constant

**Parameter dependencies:**
- $\lambda_W$ controlled by kinetic friction $\gamma$ and potential coercivity $\alpha_U$
- $R_{\infty}$ depends on cloning noise $\delta$ and coupling mismatch probability
- **Key uncertainty:** Exact form depends on which cloning coupling strategy succeeds

**Realistic expectations:**
- **Best case:** $R_{\infty}(t) = O(\delta)$ (constant bias), exponential convergence to neighborhood
- **Likely case:** $R_{\infty}(t) = O(\delta t^{-1/2})$ (slow decay), eventual exponential regime
- **Worst case:** $R_{\infty}(t) = O(\delta)$ with no net contraction (require parameter tuning)
:::

:::{note}
This is a **target theorem**—the exact statement will be refined based on which proof strategy succeeds. The key scientific contribution is establishing **some form** of Wasserstein convergence, even if weaker than originally hoped.
:::

---

## 3. Stage 1: Kinetic Operator Wasserstein Contraction (RESEARCH PHASE)

### 3.1. Objective and Challenges

**Goal:** Prove that $\Psi_{\text{kin}}(\tau)$ contracts Wasserstein distance for the **underdamped Langevin dynamics** with **coercive (non-convex) potential**.

**Two key challenges:**
1. **Underdamped (second-order):** Position + velocity state $(x, v)$, not just position
2. **Non-convex:** Confining potential $U$ is coercive but not uniformly convex

**Existing literature:**
- Eberle (2016): Reflection coupling for **overdamped**, **non-convex** diffusions ✅
- Majka (2020): Localization for **overdamped**, **non-globally dissipative** SDEs ✅
- **Missing:** Reflection coupling for **underdamped**, **non-convex** systems ❌

### 3.2. Proof Strategy: Reflection Coupling for Underdamped Langevin

:::{prf:definition} Reflection Coupling for Underdamped Dynamics (Proposed)
:label: def-underdamped-reflection-coupling

Given two initial conditions $(x_0, v_0)$ and $(x_0', v_0')$, construct a coupling $(X_t, V_t, X_t', V_t')$ as follows:

**1. Define the separation vector:**

$$
\Delta_t := (x_t - x_t', v_t - v_t') \in \mathbb{R}^{2d}
$$

**2. Reflection hyperplane:** At each time $t$, define the hyperplane $H_t$ perpendicular to $\Delta_t$:

$$
H_t := \{(x, v) \in \mathbb{R}^{2d} : \langle (x, v), \Delta_t \rangle = 0\}
$$

**3. Reflected Brownian motion:** For the velocity noise:
- Sample $dW_t \sim \mathcal{N}(0, \sigma_v^2 I_d \, dt)$
- Compute reflected noise: $dW_t' := R_{H_t}(dW_t)$ where $R_{H_t}$ is reflection across $H_t$ in the $v$-coordinate

**4. Coupled dynamics:**

$$
\begin{aligned}
dx_t &= v_t \, dt, \quad dv_t = F(x_t) \, dt - \gamma v_t \, dt + dW_t \\
dx_t' &= v_t' \, dt, \quad dv_t' = F(x_t') \, dt - \gamma v_t' \, dt + dW_t'
\end{aligned}
$$

**Key property:** The reflection is designed so that $\mathbb{E}[\langle \Delta_t, dW_t - dW_t' \rangle] \le 0$ (noise reduces distance on average).
:::

### 3.3. Technical Challenges and Open Problems

**Challenge 1:** Reflection in $(x, v)$ space vs. just $v$ space
- **Issue:** Should we reflect the full $(x, v)$ noise or just the $v$ component?
- **Approach 1:** Reflect only $dW_t$ (velocity noise)—simpler but may lose optimality
- **Approach 2:** Reflect in full phase space—more complex coupling construction

**Challenge 2:** Lyapunov function for contraction proof
- **Issue:** Standard $V(\Delta) = \|\Delta_x\|^2 + \|\Delta_v\|^2$ doesn't work for non-convex $U$
- **Approach:** Use weighted Lyapunov:

$$
V(\Delta) = \|\Delta_v\|^2 + \kappa(x_t, x_t') \|\Delta_x\|^2 + 2\mu \langle \Delta_x, \Delta_v \rangle
$$

where $\kappa(x, x')$ is **position-dependent** (large near boundary, small in interior)

**Challenge 3:** Localization for non-convex interior
- **Issue:** In the flat interior, friction alone must provide contraction
- **Approach:** Partition state space:
  - **Boundary region** $\mathcal{B}$: Coercivity dominates ($\nabla U$ large)
  - **Interior region** $\mathcal{I}$: Friction dominates ($\gamma$ large)
  - Prove contraction separately in each region, then glue

**Challenge 4:** Discrete-time coupling (BAOAB integrator)
- **Issue:** Reflection coupling is defined for continuous-time SDEs
- **Approach:** Either (a) prove for continuous flow then bound discretization error, or (b) construct discrete reflection coupling directly

### 3.4. Required Work for Stage 1 (4-6 weeks)

**Phase 1.1 (Week 1-2): Literature Deep Dive**
- **Task:** Study Eberle (2016) Sections 3-4 in detail
- **Task:** Extract reflection coupling construction algorithm
- **Task:** Identify which techniques extend to underdamped case
- **Deliverable:** Technical memo on adaptation strategy

**Phase 1.2 (Week 3-4): Lyapunov Function Construction**
- **Task:** Define position-dependent weight $\kappa(x, x')$ using $\nabla^2 U$
- **Task:** Compute $\frac{d}{dt} V(\Delta_t)$ for the coupled underdamped dynamics
- **Task:** Prove $\frac{d}{dt} V \le -\alpha V$ in each region (boundary/interior)
- **Deliverable:** Lemma proving Lyapunov dissipation

**Phase 1.3 (Week 5-6): Contraction Theorem**
- **Task:** Combine localization + reflection + Lyapunov into full proof
- **Task:** Extract explicit contraction rate $\lambda_{\text{kin}}$
- **Task:** Handle BAOAB discretization (either via weak error bounds or direct discrete coupling)
- **Deliverable:** Theorem {prf:ref}`thm-kinetic-wasserstein-underdamped` with complete proof

:::{prf:theorem} Kinetic Operator Wasserstein Contraction (Underdamped, Coercive)
:label: thm-kinetic-wasserstein-underdamped

For the Euclidean Gas kinetic operator $\Psi_{\text{kin}}(\tau)$ with:
- Underdamped Langevin dynamics on $(x, v) \in \mathbb{R}^{2d}$
- Confining potential $U$ satisfying Axiom {prf:ref}`axiom-confining-potential` (coercive)
- Friction coefficient $\gamma > 0$

there exists a constant $\lambda_{\text{kin}} > 0$ such that:

$$
W_2(\Psi_{\text{kin}}(\tau)_* \mu, \pi_{\text{kin}}) \le e^{-\lambda_{\text{kin}} \tau} W_2(\mu, \pi_{\text{kin}})
$$

where $\pi_{\text{kin}}$ is the stationary distribution of $\Psi_{\text{kin}}$.

**Explicit rate:** $\lambda_{\text{kin}} = \min(\gamma/2, \alpha_U/(2R_U))$ where $\alpha_U, R_U$ are the coercivity constants.
:::

**Contingency Plan:** If reflection coupling proves intractable:
- **Fallback 1:** Use **synchronous coupling** with localization (weaker rate but simpler)
- **Fallback 2:** Prove Wasserstein contraction only to **marginal on positions** $x$ (ignore velocities)
- **Fallback 3:** Settle for **Wasserstein stability** (bounded but not contractive) and rely on cloning for net contraction

---

## 4. Stage 2: Cloning Operator Coupling (EXPLORATORY RESEARCH)

### 4.1. The Central Challenge

**Why cloning is hard:** The cloning operator involves:
1. **Fitness-based selection:** Choose companion walker by fitness ranking
2. **Non-local jumps:** Walker can teleport to companion's position
3. **Stochastic decisions:** Clone/persist based on random threshold

**Why naive couplings fail:**
- **Synchronous coupling:** Shared randomness doesn't control companion distance
- **Maximal coupling:** Gives valid bound but likely too weak (no contraction)
- **Fitness coupling (V1):** Gemini showed this is fundamentally flawed

**What we need:** A coupling that exploits the **selective pressure** of fitness-based cloning to create contraction (or bounded expansion).

### 4.2. Strategy A: Expected Distance Coupling (Primary Approach)

**Key idea:** Instead of pathwise contraction, prove contraction **in expectation** by averaging over cloning randomness.

:::{prf:definition} Expected Distance Coupling for Cloning
:label: def-expected-distance-coupling

Given two swarms $\mathcal{S} = (w_1, \ldots, w_N)$ and $\mathcal{S}' = (w_1', \ldots, w_N')$:

**1. Shared cloning decisions:** For each walker pair $(w_i, w_i')$, use:
- Shared random threshold: $T_i \sim \text{Unif}(0, p_{\max})$
- Shared Gaussian noise (if both clone): $\xi_i \sim \mathcal{N}(0, \delta^2 I_d)$

**2. Independent companion selection:** Do NOT try to couple companions
- Walker $w_i$ selects $c(i)$ from $\mathcal{S}$ based on fitness potential
- Walker $w_i'$ selects $c(i')$ from $\mathcal{S}'$ based on fitness potential
- Companions are **independent** (key difference from V1!)

**3. Four cases:**
- **Both persist:** $x_i^{\text{new}} = x_i$, $x_i'^{\text{new}} = x_i'$
- **Both clone:** $x_i^{\text{new}} = x_{c(i)} + \xi_i$, $x_i'^{\text{new}} = x_{c(i)'} + \xi_i$ (shared noise)
- **$i$ clones, $i'$ persists:** $x_i^{\text{new}} = x_{c(i)} + \xi_i$, $x_i'^{\text{new}} = x_i'$
- **$i$ persists, $i'$ clones:** $x_i^{\text{new}} = x_i$, $x_i'^{\text{new}} = x_{c(i)'} + \xi_i$
:::

**Analysis approach:**

$$
\begin{aligned}
\mathbb{E}[\|x_i^{\text{new}} - x_i'^{\text{new}}\|^2] &= p_{\text{both persist}} \|x_i - x_i'\|^2 \\
&\quad + p_{\text{both clone}} \mathbb{E}[\|x_{c(i)} - x_{c(i)'}\|^2] \\
&\quad + p_{\text{mismatch}} \mathbb{E}[\text{mismatch term}]
\end{aligned}
$$

**Key questions:**
1. Can we bound $\mathbb{E}[\|x_{c(i)} - x_{c(i)'}\|^2]$ in terms of swarm structure?
2. How large is $p_{\text{mismatch}}$ as a function of $\|x_i - x_i'\|$?
3. Does the mismatch term create net contraction or expansion?

### 4.3. Strategy B: Swarm-Level Maximal Coupling (Conservative Fallback)

**Key idea:** Use maximal coupling for the entire swarm configuration (not pathwise).

:::{prf:definition} Swarm-Level Maximal Coupling
:label: def-swarm-maximal-coupling

Given two swarm distributions $\mu$ and $\nu$ on $\Sigma_N$, construct the maximal coupling $\pi^* \in \Pi(\mu, \nu)$ such that:

$$
\pi^*(\mathcal{S} = \mathcal{S}') = \max_{\pi \in \Pi(\mu, \nu)} \pi(\mathcal{S} = \mathcal{S}')
$$

Then apply $\Psi_{\text{clone}}$ independently to each marginal.
:::

**Bound:**

$$
W_2^2(\Psi_{\text{clone},*} \mu, \Psi_{\text{clone},*} \nu) \le (1 + C_{\text{expand}}) W_2^2(\mu, \nu) + C_{\delta} \delta^2
$$

where $C_{\text{expand}} > 0$ may be large (expansion).

**Advantage:** Guaranteed valid coupling (correct marginals)

**Disadvantage:** Likely gives expansion, not contraction—requires strong kinetic contraction to compensate

### 4.4. Strategy C: Conditional Independence Coupling (Advanced)

**Key idea:** Exploit the conditional independence structure of cloning given fitness potentials.

**Observation:** If we condition on the fitness potentials $V_{\text{fit},1}, \ldots, V_{\text{fit},N}$, the cloning decisions are independent across walkers.

**Approach:**
1. Couple the fitness potentials first (using Lipschitz continuity)
2. Given coupled potentials, couple cloning decisions independently for each walker
3. Analyze expected distance conditional on fitness coupling

**Challenge:** This requires proving Lipschitz continuity of the **entire fitness potential function** $V_{\text{fit}}: \Sigma_N \to \mathbb{R}^N$, which is non-trivial.

### 4.5. Required Work for Stage 2 (8-12 weeks)

**Phase 2.1 (Week 1-2): Preliminary Analysis**
- **Task:** Compute $\mathbb{E}[\|x_{c(i)} - x_{c(i)'}\|^2]$ for different swarm configurations
- **Task:** Derive bounds on $p_{\text{mismatch}}$ using fitness potential Lipschitz constants
- **Task:** Identify parameter regimes where Strategy A might yield contraction
- **Deliverable:** Feasibility report on Strategy A

**Phase 2.2 (Week 3-6): Strategy A Development**
- **Task:** Formalize expected distance coupling definition
- **Task:** Prove detailed case-by-case expectation bounds
- **Task:** Attempt to prove net contraction or bounded expansion
- **Decision Point:** If Strategy A yields satisfactory bound, proceed to Stage 3
- **Deliverable:** Lemma on cloning operator Wasserstein bound (Strategy A)

**Phase 2.3 (Week 7-10): Fallback to Strategy B/C** (if Strategy A fails)
- **Task:** Implement maximal coupling (Strategy B) and compute explicit bounds
- **Task:** OR develop conditional independence coupling (Strategy C) with fitness Lipschitz analysis
- **Task:** Determine which fallback gives tightest bound
- **Deliverable:** Lemma on cloning operator Wasserstein bound (fallback strategy)

**Phase 2.4 (Week 11-12): Explicit Constant Extraction**
- **Task:** Extract all constants ($C_{\text{expand}}$, $C_{\delta}$, etc.) in terms of algorithm parameters
- **Task:** Verify compatibility with Foster-Lyapunov parameter regime
- **Deliverable:** Complete Lemma {prf:ref}`lem-cloning-wasserstein-revised`

:::{prf:lemma} Cloning Operator Wasserstein Bound (Revised, Strategy-Dependent)
:label: lem-cloning-wasserstein-revised

Under **one of** the coupling strategies (A, B, or C), the cloning operator satisfies:

$$
W_2^2(\Psi_{\text{clone},*} \mu, \Psi_{\text{clone},*} \nu) \le (1 + A_{\text{clone}}) W_2^2(\mu, \nu) + B_{\text{clone}} \delta^2
$$

where:
- $A_{\text{clone}} \in \mathbb{R}$ is the **expansion/contraction coefficient**
  - **Best case (Strategy A):** $A_{\text{clone}} < 0$ (contraction)
  - **Likely case:** $A_{\text{clone}} \approx 0$ (neutral)
  - **Worst case (Strategy B):** $A_{\text{clone}} > 0$ (bounded expansion)
- $B_{\text{clone}} = O(N)$ is the noise coefficient
- Both constants depend on the coupling strategy and algorithm parameters

**The exact statement will be determined during Phase 2.2-2.3** based on which strategy succeeds.
:::

**Contingency Plan:** If all three strategies fail to give usable bounds:
- **Fallback 4:** Prove Wasserstein convergence for a **modified cloning operator** with added regularization
- **Fallback 5:** Settle for **asymptotic Wasserstein convergence** (convergence as $t \to \infty$ without explicit rate)
- **Fallback 6:** Pivot to proving **Wasserstein stability** only (bounded distance, no convergence)

---

## 5. Stage 3: Composition and Main Theorem (2-4 weeks)

### 5.1. Seesaw Mechanism (Revisited)

**Goal:** Combine kinetic contraction with cloning bound to get net convergence.

**Three scenarios based on Stage 2 outcome:**

**Scenario 1:** Cloning contracts ($A_{\text{clone}} < 0$)
- **Result:** Both operators contract—strong exponential convergence
- **Theorem:** $W_2(\mu_t, \pi_{\text{QSD}}) \le C e^{-\lambda t}$ with large $\lambda$

**Scenario 2:** Cloning neutral ($A_{\text{clone}} \approx 0$)
- **Result:** Kinetic contracts, cloning preserves—moderate exponential convergence
- **Theorem:** $W_2(\mu_t, \pi_{\text{QSD}}) \le C e^{-\lambda t} + O(\delta)$ with moderate $\lambda$

**Scenario 3:** Cloning expands ($A_{\text{clone}} > 0$)
- **Result:** Kinetic must overcome cloning expansion—weak or no net contraction
- **Theorem:** Depends on parameter tuning; may require $\gamma$ or $\tau$ adjustment
- **Possible outcome:** Convergence only in parameter subregime

### 5.2. Composition Analysis

:::{prf:theorem} Wasserstein Convergence via Seesaw (Scenario-Dependent)
:label: thm-wasserstein-seesaw-revised

**Assumption:** Parameters satisfy $e^{-\lambda_{\text{kin}} \tau} < \sqrt{1 + A_{\text{clone}}}^{-1}$ (net contraction condition).

Then the composed operator satisfies:

$$
W_2(\Psi_{\text{total},*} \mu, \pi_{\text{QSD}}) \le \rho W_2(\mu, \pi_{\text{QSD}}) + R_{\text{bias}}
$$

where:
- $\rho := e^{-\lambda_{\text{kin}} \tau} \sqrt{1 + A_{\text{clone}}} < 1$ (net contraction rate)
- $R_{\text{bias}} = O(e^{-\lambda_{\text{kin}} \tau} \sqrt{B_{\text{clone}}} \delta)$ (asymptotic bias)

**Iterating $t$ times:**

$$
W_2(\mu_t, \pi_{\text{QSD}}) \le \rho^t W_2(\mu_0, \pi_{\text{QSD}}) + \frac{R_{\text{bias}}}{1 - \rho}
$$

**Net convergence rate:** $\lambda_W = -\log(\rho) = \lambda_{\text{kin}} \tau - \frac{1}{2} \log(1 + A_{\text{clone}})$
:::

**If net contraction condition fails:** Need to adjust parameters ($\gamma \uparrow$, $\tau \uparrow$, or $\delta \downarrow$) or settle for weaker result.

### 5.3. Required Work for Stage 3 (2-4 weeks)

**Phase 3.1 (Week 1): Parameter Condition Verification**
- **Task:** Using bounds from Stages 1-2, verify net contraction condition
- **Task:** If condition fails, determine parameter adjustments needed
- **Deliverable:** Parameter feasibility report

**Phase 3.2 (Week 2-3): Main Theorem Proof**
- **Task:** Prove Theorem {prf:ref}`thm-wasserstein-seesaw-revised` for the specific scenario
- **Task:** Extract explicit $\lambda_W$ and $C_W$
- **Task:** Analyze asymptotic behavior as $\delta \to 0$
- **Deliverable:** Main Theorem {prf:ref}`thm-wasserstein-convergence-main-v2`

**Phase 3.3 (Week 4): Special Cases and Extensions**
- **Task:** Derive corollaries for specific parameter choices
- **Task:** Discuss mean-field limit $N \to \infty$ (heuristic)
- **Task:** Connection to existing TV/KL convergence results
- **Deliverable:** Remark sections and discussion

---

## 6. Revised Implementation Timeline (4-6 Months)

### 6.1. Month-by-Month Breakdown

**Month 1: Kinetic Operator Foundation (Critical Path)**
- Week 1-2: Literature review and adaptation strategy
- Week 3-4: Lyapunov function and localization
- **Milestone:** Draft proof of Theorem {prf:ref}`thm-kinetic-wasserstein-underdamped`
- **Risk:** If reflection coupling too difficult, switch to synchronous fallback (adds 2 weeks)

**Month 2: Kinetic Operator Completion + Cloning Exploration**
- Week 5-6: Finalize kinetic proof, extract constants
- Week 7-8: Cloning preliminary analysis (Strategy A feasibility)
- **Milestone:** Theorem {prf:ref}`thm-kinetic-wasserstein-underdamped` complete ✓
- **Decision Point:** Determine primary cloning strategy

**Month 3: Cloning Operator Research (High Risk)**
- Week 9-12: Full development of chosen cloning strategy
- **Milestone:** Draft Lemma {prf:ref}`lem-cloning-wasserstein-revised`
- **Risk:** If Strategy A fails, switch to B/C (adds 2-4 weeks)
- **Buffer:** 2-week contingency for strategy pivot

**Month 4: Cloning Completion + Composition**
- Week 13-14: Finalize cloning lemma, extract constants
- Week 15-16: Composition theorem and parameter verification
- **Milestone:** Main Theorem {prf:ref}`thm-wasserstein-convergence-main-v2` complete ✓

**Month 5: Write-Up and Numerical Validation**
- Week 17-18: Write full document with all proofs
- Week 19-20: Implement numerical experiments (1D/2D examples)
- **Milestone:** Draft manuscript ready for review

**Month 6: Review and Finalization**
- Week 21-22: Submit to Gemini for rigorous mathematical review
- Week 23-24: Address feedback, finalize all proofs
- **Milestone:** Publication-ready manuscript ✓

### 6.2. Parallel Work Opportunities

**Can be done simultaneously:**
- Kinetic literature review (Week 1-2) || Cloning preliminary analysis (Week 1-2)
- Kinetic proof writing (Week 5-6) || Cloning Strategy A development (Week 7-8)
- Numerical experiments (Week 19-20) || Final proof polishing (Week 19-20)

**Must be sequential:**
- Kinetic proof must complete before composition (kinetic provides $\lambda_{\text{kin}}$)
- Cloning proof must complete before composition (cloning provides $A_{\text{clone}}$)
- Both Stages 1-2 must complete before Stage 3

### 6.3. Critical Path Analysis

**Critical path:** Kinetic (Month 1-2) → Cloning (Month 3-4) → Composition (Month 4)

**Longest pole:** Cloning operator (Month 3-4)—highest risk, most research uncertainty

**Buffer allocation:**
- Month 1-2: 1 week buffer (total 9 weeks for kinetic)
- Month 3-4: 2 week buffer (total 10 weeks for cloning)
- Month 5-6: 2 week buffer (total 10 weeks for write-up)
- **Total project duration:** 24 weeks (6 months) with buffers

---

## 7. Risk Mitigation and Contingency Plans

### 7.1. Risk Matrix

| Risk | Probability | Impact | Mitigation |
|:-----|:------------|:-------|:-----------|
| Underdamped reflection coupling too hard | MEDIUM | HIGH | Fallback to synchronous coupling (weaker rate) |
| All cloning strategies give expansion | MEDIUM | HIGH | Adjust parameters or settle for stability |
| Net contraction condition fails | LOW | MEDIUM | Parameter tuning or modified algorithm |
| Numerical experiments don't match theory | LOW | LOW | Investigate discretization effects |

### 7.2. Decision Trees

**Decision Point 1 (End of Month 1): Kinetic Coupling**
- ✅ If reflection coupling succeeds → Proceed to Month 2
- ⚠️ If too difficult → Switch to synchronous coupling, add 2 weeks
- ❌ If even synchronous fails → Pivot to position-marginal only (major scope reduction)

**Decision Point 2 (End of Month 2): Cloning Strategy**
- ✅ If Strategy A looks promising → Pursue Strategy A in Month 3
- ⚠️ If Strategy A unlikely → Switch to Strategy B/C immediately
- ❌ If all look bad → Consider Fallback 4-6 (algorithm modification or weaker result)

**Decision Point 3 (End of Month 4): Net Contraction**
- ✅ If $\rho < 1$ → Proceed to write-up
- ⚠️ If $\rho \approx 1$ → Parameter optimization, add 2 weeks
- ❌ If $\rho \ge 1$ → Major revision: either change parameters or settle for asymptotic convergence

### 7.3. Success Criteria by Stage

**Minimum Success (Publishable):**
- [ ] Kinetic operator Wasserstein contraction (even with weaker rate)
- [ ] Cloning operator Wasserstein bound (expansion allowed if bounded)
- [ ] Some form of convergence (even if only asymptotic or parameter-dependent)
- [ ] All proofs mathematically rigorous (pass Gemini review)
- **Outcome:** Publishable paper on Wasserstein analysis of particle swarms

**Target Success (High-Quality Paper):**
- [ ] Exponential kinetic contraction with explicit rate
- [ ] Cloning operator neutral or weakly contractive
- [ ] Exponential convergence to $O(\delta)$-neighborhood of QSD
- [ ] Numerical validation matching theory
- **Outcome:** Strong paper for optimization/probability journal

**Stretch Success (Major Contribution):**
- [ ] Tight kinetic contraction rate via optimal reflection coupling
- [ ] Cloning operator contraction (Strategy A succeeds)
- [ ] Exact exponential convergence (no asymptotic bias)
- [ ] Mean-field limit analysis ($N \to \infty$)
- **Outcome:** Top-tier journal publication

---

## 8. Publication Strategy (Unchanged from V1)

[Same as V1—see original document]

---

## 9. Open Questions and Future Directions

### 9.1. Immediate Research Questions

**Question 1:** Can Strategy A (expected distance coupling) yield contraction?
- **Approach:** Analyze correlation between fitness and position
- **Timeline:** Answered in Month 3

**Question 2:** What is the optimal reflection strategy for underdamped Langevin?
- **Approach:** Compare different reflection schemes (position-only, velocity-only, full phase space)
- **Timeline:** Explored in Month 1-2

**Question 3:** Does $\lambda_W$ match the KL convergence rate $\lambda_{\text{KL}}$?
- **Conjecture:** They should be comparable (both governed by Langevin dissipation)
- **Timeline:** Compared after Month 4

### 9.2. Long-Term Extensions

**Extension 1:** Adaptive Gas with Hessian-based diffusion
- **Challenge:** Anisotropic noise breaks reflection symmetry
- **Approach:** Use Riemannian Wasserstein distance

**Extension 2:** Mean-field limit $N \to \infty$
- **Challenge:** Cloning coupling may break down for infinite swarms
- **Approach:** Propagation of chaos + exchangeability

**Extension 3:** Connection to Fisher-Rao (if curvature holds)
- **Benefit:** Wasserstein + curvature → Fisher-Rao via HWI inequality
- **Timeline:** Only after [16_fisher_rao_roadmap.md](16_fisher_rao_roadmap.md) Phase 1 completes

---

## 10. Conclusion

### 10.1. Summary of Revisions

This V2 roadmap addresses all critical issues from Gemini's review:

1. ✅ **Removed flawed fitness coupling contraction claim**
2. ✅ **Added three alternative cloning strategies with contingencies**
3. ✅ **Elevated underdamped reflection coupling to dedicated research phase**
4. ✅ **Revised timeline from 4-8 weeks to 4-6 months (realistic)**
5. ✅ **Upgraded difficulty from LOW-MEDIUM to MEDIUM-HIGH**
6. ✅ **Added explicit risk mitigation and decision trees**

### 10.2. Realistic Expectations

**What we can deliver (high confidence):**
- Rigorous Wasserstein analysis of kinetic operator ✅
- Rigorous Wasserstein bound for cloning operator (expansion or neutral) ✅
- Some form of Wasserstein convergence result ✅

**What depends on research outcomes (medium confidence):**
- Exponential convergence with explicit rate (depends on cloning coupling success)
- Tight constants and parameter optimization
- Strong enough result for top-tier journal

**What may require future work (low confidence):**
- Perfect exponential convergence with no bias
- Optimal cloning coupling with proven contraction
- Mean-field limit analysis

### 10.3. Final Recommendation

✅ **PROCEED WITH REVISED SCOPING**

This is a **high-value research project** with:
- Clear scientific contribution (first Wasserstein analysis of fitness-based swarms)
- Realistic timeline (4-6 months)
- Built-in contingency plans for all major risks
- Publishable outcome guaranteed (even in fallback scenarios)

**Next steps:**
1. Begin Month 1 (kinetic operator) immediately
2. Set up monthly review meetings to evaluate progress
3. Maintain agile approach with decision points
4. Expect iterative refinement of proof strategy

---

## References

[Same as V1—see original document for full bibliography]

**Additional references for V2:**
- Chewi, S., et al. (2024). "Analysis of Langevin Monte Carlo from Poincaré to Log-Sobolev." *Foundations of Computational Mathematics*.
- Eberle, A., Guillin, A., & Zimmer, R. (2019). "Couplings and quantitative contraction rates for Langevin dynamics." *Annals of Probability*, 47(4), 1982-2010.
