# Month 1: Kinetic Operator Wasserstein Contraction

**Project:** Wasserstein Distance Convergence for Euclidean Gas
**Phase:** Stage 1 - Kinetic Operator Foundation
**Timeline:** Weeks 1-6 (Month 1-2 from main roadmap)
**Status:** 🚧 IN PROGRESS

**Objective:** Prove that the underdamped Langevin kinetic operator $\Psi_{\text{kin}}(\tau)$ with coercive (non-convex) potential contracts Wasserstein distance.

**Target Theorem:**

$$
W_2(\Psi_{\text{kin}}(\tau)_* \mu, \pi_{\text{kin}}) \le e^{-\lambda_{\text{kin}} \tau} W_2(\mu, \pi_{\text{kin}})
$$

with explicit $\lambda_{\text{kin}} = \min(\gamma/2, \alpha_U/(2R_U))$ and $N$-dependency tracking.

---

## Week 1-2: Literature Review and Adaptation Strategy

### Objectives

1. Deep dive into Eberle (2016) reflection coupling for overdamped diffusions
2. Study Majka (2020) localization techniques for non-globally dissipative SDEs
3. Identify extension strategy to underdamped (second-order) systems
4. Survey existing underdamped Langevin contraction results

### Task 1.1: Eberle (2016) - Reflection Coupling Fundamentals

**Reference:** Eberle, A. (2016). "Reflection couplings and contraction rates for diffusions." *Probability Theory and Related Fields*, 166(3-4), 851-886.

**Key sections to study:**
- Section 3: Reflection coupling construction
- Section 4: Contraction analysis via Lyapunov functions
- Section 5: Non-convex potentials and localization

**Extraction goals:**
- [ ] Understand reflection hyperplane construction
- [ ] Extract Lyapunov function design principles
- [ ] Identify assumptions that extend to underdamped case
- [ ] Note assumptions that DON'T extend (requires modification)

**Deliverable:** Summary document `eberle_2016_notes.md` with:
- Key theorems (with page numbers)
- Coupling construction algorithm (pseudocode)
- Lyapunov function templates
- Extension strategy to underdamped

**Estimated time:** 3-4 days

---

### Task 1.2: Majka (2020) - Localization for Non-Dissipative SDEs

**Reference:** Majka, M. B. (2020). "Transportation inequalities for non-globally dissipative SDEs with jumps via Malliavin calculus and coupling." *Annals of Probability*, 48(3), 1103-1133.

**Key sections to study:**
- Section 2: Localization framework
- Section 3: Coupling on regions with different behaviors
- Section 4: Patchwork arguments for global contraction

**Extraction goals:**
- [ ] Understand region decomposition (boundary vs. interior)
- [ ] Extract transition zone smoothing techniques
- [ ] Identify how to handle "flat" interior regions
- [ ] Note compatibility with reflection coupling

**Deliverable:** Summary document `majka_2020_notes.md` with:
- Localization strategy overview
- Boundary/interior coupling construction
- Transition function design
- Application to coercive (non-convex) potentials

**Estimated time:** 3-4 days

---

### Task 1.3: Underdamped Langevin Literature Survey

**Goal:** Find existing results on Wasserstein contraction for underdamped Langevin

**Key papers to review:**

1. **Eberle, Guillin, Zimmer (2019):** "Couplings and quantitative contraction rates for Langevin dynamics."
   - Check if they handle underdamped case
   - Extract any relevant techniques

2. **Chewi et al. (2024):** "Analysis of Langevin Monte Carlo from Poincaré to Log-Sobolev."
   - Recent comprehensive survey
   - May have underdamped references

3. **Dalalyan & Riou-Durand (2020):** "On sampling from a log-concave density using kinetic Langevin diffusions."
   - Specifically addresses underdamped
   - May have relevant bounds

**Deliverable:** Annotated bibliography with:
- [ ] Papers handling underdamped Langevin
- [ ] Papers handling non-convex potentials
- [ ] Identified gaps in literature (what we need to prove)
- [ ] Closest existing results to our target theorem

**Estimated time:** 2-3 days

---

### Task 1.4: Euclidean Gas Confining Potential Analysis

**Goal:** Characterize the specific potential $U$ from the Euclidean Gas framework

**Steps:**

1. **Extract potential from framework:**
   - Read [02_euclidean_gas.md](02_euclidean_gas.md) for BAOAB implementation
   - Read [01_fragile_gas_framework.md](01_fragile_gas_framework.md) for Axiom {prf:ref}`axiom-confining-potential`
   - Identify explicit form of $U(x)$ or its properties

2. **Compute geometric properties:**
   - Hessian $\nabla^2 U(x)$ (analytically if possible)
   - Identify regions where $\nabla^2 U \succeq 0$ (locally convex)
   - Identify regions where $\nabla^2 U$ has zero/negative eigenvalues (flat/saddle)
   - Verify coercivity constants $\alpha_U, R_U$

3. **Define localization regions:**
   - **Boundary region** $\mathcal{B}_\epsilon$: $\{x : U(x) > M\}$ for some threshold $M$
   - **Interior region** $\mathcal{I}$: $\{x : U(x) \le M\}$
   - Characterize $\nabla U$ behavior in each region

**Deliverable:** Technical note `euclidean_gas_potential_geometry.md` with:
- Explicit potential function or axioms
- Hessian analysis (regions of convexity)
- Localization region definitions
- Constants $\alpha_U, R_U$ (explicit or bounds)

**Estimated time:** 2-3 days

---

### Week 1-2 Deliverable: Adaptation Strategy Memo

**Synthesis document:** `underdamped_reflection_strategy.md`

**Contents:**
1. **Summary of Eberle's approach** (overdamped)
2. **Summary of Majka's localization** (non-convex)
3. **Gap analysis:** What's missing for underdamped + non-convex?
4. **Proposed adaptation:**
   - How to extend reflection coupling to $(x, v)$ phase space
   - How to construct position-dependent Lyapunov function
   - How to partition Euclidean Gas potential into regions
5. **Risk assessment:** Technical challenges and contingencies
6. **Go/no-go decision:** Is the approach viable?

**Decision point:** End of Week 2
- ✅ **GO:** Reflection coupling looks feasible → Proceed to Week 3-4
- ⚠️ **MODIFY:** Significant challenges identified → Revise approach, add 1-2 weeks
- ❌ **NO-GO:** Reflection coupling infeasible → Switch to synchronous coupling fallback

---

## Week 3-4: Lyapunov Function and Localization

### Objective

Construct the mathematical machinery for proving contraction:
1. Position-dependent Lyapunov function $V(\Delta)$
2. Localization partition of state space
3. Dissipation inequalities in each region

### Task 2.1: Lyapunov Function Construction

**Goal:** Design $V(\Delta_x, \Delta_v)$ for the underdamped difference process

**Candidate form** (from hypocoercivity theory):

$$
V(\Delta) := \|\Delta_v\|^2 + \kappa(x, x') \|\Delta_x\|^2 + 2\mu \langle \Delta_x, \Delta_v \rangle
$$

where $\kappa(x, x')$ is **position-dependent** weight and $\mu$ is coupling coefficient.

**Design choices:**

1. **In boundary region** ($U(x), U(x') \gg 1$):
   - $\kappa(x, x') = \kappa_{\text{bdy}}$ (large, based on coercivity)
   - Coercivity dominates: $\langle \Delta_x, F(x) - F(x') \rangle \ge \alpha_U \|\Delta_x\|^2$

2. **In interior region** ($U(x), U(x') \le M$):
   - $\kappa(x, x') = \kappa_{\text{int}}$ (small, based on friction)
   - Friction dominates: $-\gamma \Delta_v$ provides dissipation

3. **Transition region:**
   - Smooth interpolation: $\kappa(x, x') = \kappa_{\text{int}} + (\kappa_{\text{bdy}} - \kappa_{\text{int}}) \cdot h(U(x) + U(x'))$
   - Where $h: \mathbb{R} \to [0, 1]$ is a smooth transition function

**Tasks:**
- [ ] Define explicit $\kappa(x, x')$ using potential $U$
- [ ] Choose coupling coefficient $\mu$ (typically $\mu = \sqrt{\kappa \gamma}$)
- [ ] Verify $V$ is positive-definite in all regions
- [ ] Compute constants: $c_1 \|\Delta\|^2 \le V(\Delta) \le c_2 \|\Delta\|^2$

**Deliverable:** Formal definition of Lyapunov function with proof of equivalence to Euclidean norm

**Estimated time:** 4-5 days

---

### Task 2.2: Dissipation Inequality - Boundary Region

**Goal:** Prove $\frac{d}{dt} V(\Delta_t) \le -\alpha_{\text{bdy}} V(\Delta_t)$ in $\mathcal{B}_\epsilon$

**Coupled dynamics** (continuous-time):

$$
\frac{d}{dt} \begin{pmatrix} \Delta_x \\ \Delta_v \end{pmatrix} = \begin{pmatrix} 0 & I_d \\ -(F(x) - F(x')) & -\gamma I_d \end{pmatrix} \begin{pmatrix} \Delta_x \\ \Delta_v \end{pmatrix} + \begin{pmatrix} 0 \\ \sigma_v(dW - dW') \end{pmatrix}
$$

With reflection coupling: $\mathbb{E}[\langle \Delta, dW - dW' \rangle] \le 0$

**Computation:**

$$
\begin{aligned}
\frac{dV}{dt} &= 2\langle \Delta_v, \frac{d\Delta_v}{dt} \rangle + \kappa \cdot 2\langle \Delta_x, \frac{d\Delta_x}{dt} \rangle + 2\mu \langle \frac{d\Delta_x}{dt}, \Delta_v \rangle + 2\mu \langle \Delta_x, \frac{d\Delta_v}{dt} \rangle \\
&= \text{[expand using dynamics]} \\
&\le -\alpha_{\text{bdy}} V + O(\sigma_v^2)
\end{aligned}
$$

**Key step:** Use coercivity in boundary region:

$$
\langle \Delta_x, F(x) - F(x') \rangle \ge \alpha_U \|\Delta_x\|^2
$$

**Tasks:**
- [ ] Expand $\frac{dV}{dt}$ algebraically
- [ ] Bound each term using coercivity and reflection properties
- [ ] Extract $\alpha_{\text{bdy}}$ as function of $\alpha_U, \gamma, \kappa_{\text{bdy}}, \mu$
- [ ] Track $N$-dependency (if potential depends on swarm size)

**Deliverable:** Lemma proving boundary region dissipation with explicit rate

**Estimated time:** 3-4 days

---

### Task 2.3: Dissipation Inequality - Interior Region

**Goal:** Prove $\frac{d}{dt} V(\Delta_t) \le -\alpha_{\text{int}} V(\Delta_t)$ in $\mathcal{I}$

**Challenge:** Potential is approximately flat ($\nabla U \approx 0$), so no coercivity.

**Solution:** Rely on friction dissipation

**Key observation:** In interior, the $\Delta_v$ term dominates:

$$
\frac{dV}{dt} \approx 2\langle \Delta_v, -\gamma \Delta_v \rangle = -2\gamma \|\Delta_v\|^2
$$

But we also have $\frac{d\Delta_x}{dt} = \Delta_v$, which couples position and velocity.

**Approach:** Use hypocoercivity argument
- Velocity dissipates via friction: $\|\Delta_v\|^2$ decays
- Position follows velocity: $\|\Delta_x\|^2$ eventually decays (indirect dissipation)
- Cross term $\langle \Delta_x, \Delta_v \rangle$ controls the coupling

**Tasks:**
- [ ] Compute $\frac{dV}{dt}$ in interior (where $\nabla U \approx 0$)
- [ ] Apply hypocoercive inequality (Villani-style)
- [ ] Extract $\alpha_{\text{int}}$ as function of $\gamma, \kappa_{\text{int}}, \mu$
- [ ] Verify $\alpha_{\text{int}} > 0$ (positive dissipation)

**Deliverable:** Lemma proving interior region dissipation with explicit rate

**Estimated time:** 3-4 days

---

### Task 2.4: Patchwork Argument

**Goal:** Combine boundary and interior dissipation into global result

**Strategy:** Use partition of unity
- Let $\chi_{\mathcal{B}}(x, x')$ = indicator for "at least one walker in boundary region"
- Let $\chi_{\mathcal{I}}(x, x') = 1 - \chi_{\mathcal{B}}$ = both walkers in interior

**Global dissipation:**

$$
\frac{d}{dt} V(\Delta_t) \le -\left[\chi_{\mathcal{B}} \alpha_{\text{bdy}} + \chi_{\mathcal{I}} \alpha_{\text{int}}\right] V(\Delta_t)
$$

**Net rate:** $\lambda_{\text{kin}} = \min(\alpha_{\text{bdy}}, \alpha_{\text{int}})$

**Tasks:**
- [ ] Formalize partition of unity with smooth transition
- [ ] Prove global dissipation inequality
- [ ] Extract final $\lambda_{\text{kin}}$ with all dependencies

**Deliverable:** Lemma proving global contraction for continuous-time flow

**Estimated time:** 2 days

---

## Week 5-6: Discrete-Time Extension and Main Theorem

### Objective

Extend continuous-time contraction to discrete-time BAOAB integrator

### Task 3.1: BAOAB Weak Error Analysis

**Goal:** Bound discretization error for Wasserstein contraction

**BAOAB integrator** (one step $\tau$):
- **B:** $v \leftarrow v + \frac{\tau}{2} F(x)$
- **A:** $v \leftarrow e^{-\gamma \tau} v + \sqrt{1 - e^{-2\gamma\tau}} \sigma_v \xi$
- **O:** $x \leftarrow x + \tau v$
- **A:** (repeat A step)
- **B:** $v \leftarrow v + \frac{\tau}{2} F(x)$

**Approach:** Use weak error bounds from SDE literature

**Key result** (standard): For sufficiently smooth potentials,

$$
W_2(\Psi_{\text{BAOAB}}(\tau)_* \mu, \Psi_{\text{exact}}(\tau)_* \mu) \le C_{\text{weak}} \tau^2
$$

**Implication:**

$$
W_2(\Psi_{\text{BAOAB}}(\tau)_* \mu, \pi_{\text{kin}}) \le e^{-\lambda_{\text{kin}} \tau} W_2(\mu, \pi_{\text{kin}}) + C_{\text{weak}} \tau^2
$$

**For small enough $\tau$:** The exponential contraction dominates the $O(\tau^2)$ error.

**Tasks:**
- [ ] Verify smoothness assumptions for Euclidean Gas potential
- [ ] Cite weak error bound theorem (or prove if necessary)
- [ ] Extract $C_{\text{weak}}$ in terms of potential derivatives
- [ ] Determine regime where $e^{-\lambda_{\text{kin}}\tau} + C_{\text{weak}}\tau^2 < 1$

**Deliverable:** Lemma bounding BAOAB discretization error

**Estimated time:** 3-4 days

---

### Task 3.2: Main Theorem Assembly

**Goal:** Combine all pieces into final theorem

:::{prf:theorem} Kinetic Operator Wasserstein Contraction (Main Result)
:label: thm-month1-main-result

For the Euclidean Gas kinetic operator $\Psi_{\text{kin}}(\tau)$ implemented via BAOAB integrator with:
- Underdamped Langevin dynamics on $(x, v) \in \mathbb{R}^d \times \mathbb{R}^d$
- Confining potential $U$ satisfying Axiom {prf:ref}`axiom-confining-potential` (coercive)
- Friction coefficient $\gamma > 0$
- Step size $\tau \le \tau_*$ (sufficiently small)

there exists a constant $\lambda_{\text{kin}} > 0$ such that:

$$
W_2(\Psi_{\text{kin}}(\tau)_* \mu, \pi_{\text{kin}}) \le e^{-\lambda_{\text{kin}} \tau} W_2(\mu, \pi_{\text{kin}})
$$

**Explicit rate:**

$$
\lambda_{\text{kin}} = \min\left(\frac{\gamma}{2}, \frac{\alpha_U}{2R_U}\right)
$$

where $\alpha_U, R_U$ are the coercivity constants from the axiom.

**N-dependency:** All constants are **independent of $N$** (number of walkers evolve independently under $\Psi_{\text{kin}}$).

**Parameter regime:** Requires $\tau \le \tau_* := \min(1/\gamma, 1/\sqrt{\alpha_U})$ to ensure discretization error is subdominant.
:::

**Tasks:**
- [ ] Write complete proof synthesizing all lemmas
- [ ] Verify all dependencies are explicit
- [ ] Check $N$-independence carefully
- [ ] Add remarks on parameter scaling

**Deliverable:** Complete theorem with full proof (20-30 pages)

**Estimated time:** 4-5 days

---

### Task 3.3: Numerical Validation (if time permits)

**Goal:** Validate theory on simple test case

**Test setup:**
- 1D or 2D state space
- Simple coercive potential (e.g., $U(x) = \frac{1}{4}x^4 - x^2$ - double well)
- Small swarm ($N = 10$)

**Measurements:**
- Empirical Wasserstein distance $W_2(\mu_t, \pi_{\text{kin}})$ over time
- Compare with theoretical rate $e^{-\lambda_{\text{kin}} t}$
- Verify discretization error scaling

**Deliverable:** Numerical experiment notebook + plots

**Estimated time:** 2-3 days (optional, if ahead of schedule)

---

## Week 5-6 Milestone: Stage 1 Complete

**Deliverable:** Full document section for publication manuscript

**Contents:**
1. Introduction to reflection coupling
2. Lyapunov function construction
3. Localization and regional dissipation
4. BAOAB discretization analysis
5. Main Theorem {prf:ref}`thm-month1-main-result`
6. All proofs (complete and rigorous)

**Review:** Submit to Gemini for mathematical rigor check before proceeding to Month 2

---

## Progress Tracking

### Week 1 Status
- [ ] Task 1.1: Eberle (2016) notes
- [ ] Task 1.2: Majka (2020) notes
- [ ] Task 1.3: Underdamped literature survey
- [ ] Task 1.4: Potential geometry analysis

### Week 2 Status
- [ ] Deliverable: Adaptation strategy memo
- [ ] **Decision Point:** Go/modify/no-go for reflection coupling

### Week 3 Status
- [ ] Task 2.1: Lyapunov function definition
- [ ] Task 2.2: Boundary dissipation lemma
- [ ] Task 2.3: Interior dissipation lemma

### Week 4 Status
- [ ] Task 2.4: Patchwork argument
- [ ] Deliverable: Global contraction lemma (continuous-time)

### Week 5 Status
- [ ] Task 3.1: BAOAB weak error analysis
- [ ] Task 3.2: Main theorem proof (draft)

### Week 6 Status
- [ ] Task 3.3: Numerical validation (optional)
- [ ] Deliverable: Complete Stage 1 document
- [ ] **Milestone:** Submit to Gemini review

---

## Risk Monitoring

| Risk | Status | Mitigation |
|:-----|:-------|:-----------|
| Reflection coupling too complex | 🟡 MONITOR | Fallback to synchronous coupling (Week 2 decision) |
| Lyapunov function doesn't work | 🟡 MONITOR | Try alternative forms (Week 3-4) |
| Interior dissipation too weak | 🟡 MONITOR | Adjust $\kappa_{\text{int}}$ or require stronger friction |
| BAOAB error too large | 🟢 LOW RISK | Standard result, well-documented |

---

## Notes and Insights

_[This section will be populated during implementation with discoveries, challenges, and solutions]_

---

## Next Steps (After Month 1)

Upon completion of Stage 1:
- **Month 2-3:** Begin Stage 2 (Cloning Operator Coupling)
- **Parallel work:** Can start cloning preliminary analysis while finalizing kinetic proof
- **Buffer:** If Stage 1 takes longer, use contingency time from Month 2

**Success metric:** Theorem {prf:ref}`thm-month1-main-result` complete, rigorous, and Gemini-approved by end of Week 6.
