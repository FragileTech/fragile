# Comprehensive Input/Output Extraction Report
## Document: 06_convergence.md (Chapter 6: Convergence Theory)

**Generated**: 2025-10-26
**Total Objects**: 26 (propositions, theorems, algorithms)
**Critical Context**: This chapter synthesizes cloning (Ch 03) and kinetic (Ch 05) operator results to prove geometric ergodicity of the full composed operator.

---

## PART I: SYNERGISTIC COMPOSITION (Objects 1-2)

### 1. prop-complete-drift-summary (Line 249)

**TYPE**: Proposition (Catalog/Summary)
**OUTPUT_TYPE**: Relation

**INPUT_OBJECTS**:
- `obj-swarm-state` (Ch 01/02)
- `obj-lyapunov-components` (Ch 03): $V_W$, $V_{\text{Var},x}$, $V_{\text{Var},v}$, $W_b$
- `obj-cloning-operator` (Ch 03): $\Psi_{\text{clone}}$
- `obj-kinetic-operator` (Ch 05): $\Psi_{\text{kin}}$

**INPUT_AXIOMS**:
- None directly (references theorems from Ch 03, 05)

**INPUT_PARAMETERS**:
- Cloning: $\kappa_x$, $\kappa_b$, $C_x$, $C_v$, $C_b$ (from Ch 03)
- Kinetic: $\kappa_W$, $\gamma$, $\sigma_v$, $\tau$, $\kappa_{\text{pot}}$, $C_W'$, $C_{\text{kin},x}$, $d\sigma_{\max}^2$ (from Ch 05)

**PROPERTIES_REQUIRED**:
- Cloning operator satisfies drift bounds from `thm-operator-drift-bounds-cloning` (Ch 03, Theorem 12.3.1)
- Kinetic operator satisfies drift bounds from:
  - `thm-inter-swarm-contraction-kinetic` (Ch 05, Section 2.3)
  - `thm-velocity-variance-contraction-kinetic` (Ch 05, Section 3.3)
  - `thm-positional-variance-bounded-expansion` (Ch 05, Section 4.3)
  - `thm-boundary-potential-contraction-kinetic` (Ch 05, Section 5.3)

**PROPERTIES_ADDED**:
- **Complementary dissipation structure**: Each operator contracts what the other expands
- **Component-wise drift inequalities** for composed operator

**RELATIONS_ESTABLISHED**:
- Cloning: contracts $V_{\text{Var},x}$, $W_b$; boundedly expands $V_W$, $V_{\text{Var},v}$
- Kinetic: contracts $V_W$, $V_{\text{Var},v}$, $W_b$; boundedly expands $V_{\text{Var},x}$
- **Key relation**: Synergistic composition enables net contraction

**INTERNAL_LEMMAS**: None (summary proposition)

**DEPENDENCIES FORWARD**:
- Used in `thm-foster-lyapunov-main` to prove net contraction
- Critical for Ch 08 (propagation of chaos) analysis

---

### 2. thm-foster-lyapunov-main (Line 266)

**TYPE**: Theorem (Main Result)
**OUTPUT_TYPE**: Property (Drift Inequality)

**INPUT_OBJECTS**:
- All objects from `prop-complete-drift-summary`
- `obj-synergistic-lyapunov` (Ch 03, Def 3.3.1): $V_{\text{total}} = V_W + c_V V_{\text{Var}} + c_B W_b$
- `obj-composed-operator`: $\Psi_{\text{total}} = \Psi_{\text{kin}} \circ \Psi_{\text{clone}}$

**INPUT_AXIOMS**:
- Foundational axioms from Ch 03 (Cloning), Ch 04 (Framework)
- Langevin dynamics axioms (Ch 05)

**INPUT_PARAMETERS**:
- All parameters from `prop-complete-drift-summary`
- **NEW coupling constants**: $c_V^*$, $c_B^*$ (to be determined)
- Time step: $\tau$

**PROPERTIES_REQUIRED**:
- Component drift bounds from `prop-complete-drift-summary`
- $\kappa_x, \kappa_b > 0$ (N-uniform, from Ch 03)
- $\kappa_W, \gamma > 0$ (from Ch 05)

**PROPERTIES_ADDED**:
- **Foster-Lyapunov drift condition**: $\mathbb{E}[V_{\text{total}}(S')|S] \leq (1-\kappa_{\text{total}}\tau)V_{\text{total}}(S) + C_{\text{total}}$
- **Contraction rate**: $\kappa_{\text{total}} = \min(\kappa_W/2, c_V^*\kappa_x/2, c_V^*\gamma/2, c_B^*(\kappa_b+\kappa_{\text{pot}}\tau)/2) > 0$
- **Drift constant**: $C_{\text{total}} = C_W + C_W'\tau + c_V^*(C_x + C_v + C_{\text{kin},x}\tau) + c_B^*(C_b + C_{\text{pot}}\tau) < \infty$
- **N-uniformity**: Both constants independent of $N$

**RELATIONS_ESTABLISHED**:
- **Optimal coupling weights**:
  - $c_V^* = \frac{\kappa_W\tau}{2\kappa_x}$
  - $c_B^* = \frac{\kappa_W\tau}{2(\kappa_b + \kappa_{\text{pot}}\tau)}$
- **Balanced contraction**: All components contract at minimum rate $\kappa_{\text{total}}$

**INTERNAL_LEMMAS**:
- Tower property decomposition: $\mathbb{E}_{\text{total}}[\Delta V] = \mathbb{E}_{\text{clone}}[\Delta V] + \mathbb{E}_{\text{clone}}[\mathbb{E}_{\text{kin}}[\Delta V|S^{\text{clone}}]]$
- Component drift aggregation (Parts III-IV of proof)
- Weight selection inequality system (Part V)

**DEPENDENCIES FORWARD**:
- **Direct dependency**: `thm-main-convergence` (proves geometric ergodicity using this drift)
- **Critical for**: Ch 08 propagation of chaos (uses $\kappa_{\text{total}}$ for mean-field limit)
- **Used in**: All parameter optimization results (Ch 6, Sections 5-6)

---

## PART II: MAIN CONVERGENCE THEOREM (Objects 3-7)

### 3. thm-phi-irreducibility (Line 635)

**TYPE**: Theorem
**OUTPUT_TYPE**: Property (Topological)

**INPUT_OBJECTS**:
- `obj-euclidean-gas-markov-chain`: On alive state space $\Sigma_N^{\text{alive}}$
- `obj-core-set` $\mathcal{C}$ (defined in proof): Interior, all alive, low velocities
- Cloning operator, Kinetic operator (from Ch 03, 05)

**INPUT_AXIOMS**:
- Axiom 4.2.1 (Ch 03): Reward structure for cloning companion selection
- Gaussian noise axioms (Ch 05): Langevin noise $\sigma_v$, perturbation noise $\sigma_{\text{pert}}$

**INPUT_PARAMETERS**:
- Barrier potential: $\varphi_{\text{barrier}}(x)$
- Reward structure: $r_i = f(x_i, v_i, s_i)$
- Langevin noise: $\sigma_v$
- Cloning noise: $\delta_{\text{clone}}$ (inelastic collision spread)
- Minimum eigenvalue of Hessian: $\lambda_{\min}$ (for hypocoercivity)

**PROPERTIES_REQUIRED**:
- **Cloning**: Positive probability of selecting any alive walker as companion
- **Gaussian noise**: Positive density everywhere in $\mathbb{R}^{2d}$
- **Hörmander condition**: Lie algebra spans tangent space (for hypoellipticity)

**PROPERTIES_ADDED**:
- **φ-irreducibility** w.r.t. Lebesgue measure: $\forall S_A, O_B \subset \Sigma_N^{\text{alive}}, \exists M: P^M(S_A, O_B) > 0$
- **Two-stage accessibility**:
  - Stage 1 (Gathering): Any state can reach core $\mathcal{C}$ in one step via lucky cloning
  - Stage 2 (Spreading): From $\mathcal{C}$, can reach any open set via hypoelliptic diffusion

**RELATIONS_ESTABLISHED**:
- **Cloning as global teleportation**: Breaks ergodic barriers, enables arbitrarily large jumps
- **Kinetics as local steering**: Hörmander hypoellipticity provides controllability
- **Synergy**: Neither alone is irreducible; composition is

**INTERNAL_LEMMAS**:
- **Lem 1 (Alpha walker)**: Minimum barrier walker has positive selection probability $p_\alpha > 0$
- **Lem 2 (Lucky cloning)**: All walkers selecting alpha has probability $p_\alpha^{N-1} > 0$
- **Lem 3 (Hörmander controllability)**: Single particle reaches any neighborhood with positive probability
- **Lem 4 (N-particle independence)**: Independent noise $\Rightarrow$ product probability $\prod_{i=1}^N p_i > 0$
- **Lem 5 (Safe interior)**: Survival probability during $M$ steps bounded below

**DEPENDENCIES FORWARD**:
- **Direct**: `thm-main-convergence` (uniqueness of QSD requires irreducibility)
- **Meyn-Tweedie theory**: Irreducibility + aperiodicity + drift $\Rightarrow$ geometric ergodicity
- **Ch 08**: Mean-field limit requires unique QSD (no isolated islands in state space)

---

### 4. thm-aperiodicity (Line 836)

**TYPE**: Theorem
**OUTPUT_TYPE**: Property (Topological)

**INPUT_OBJECTS**:
- `obj-euclidean-gas-markov-chain` on $\Sigma_N^{\text{alive}}$
- Gaussian noise (continuous, non-degenerate)

**INPUT_AXIOMS**:
- Gaussian perturbation at every step: $\eta \sim \mathcal{N}(0, \sigma_{\text{pert}}^2 I)$

**INPUT_PARAMETERS**:
- Perturbation noise: $\sigma_{\text{pert}} > 0$

**PROPERTIES_REQUIRED**:
- **Continuous noise**: Positive density w.r.t. Lebesgue measure
- **φ-irreducibility**: From `thm-phi-irreducibility`

**PROPERTIES_ADDED**:
- **Aperiodicity**: $\forall S \in \Sigma_N^{\text{alive}}, \exists m,n: \gcd(m,n)=1, P^m(S,U)>0, P^n(S,U)>0$
- **No periodic structure**: Chain cannot cycle with period $d > 1$

**RELATIONS_ESTABLISHED**:
- Continuous noise $\Rightarrow$ zero probability of exact return
- Core set $\mathcal{C}$ accessible in one step from any periodic class $\Rightarrow$ contradiction

**INTERNAL_LEMMAS**:
- **Lem 1 (Continuous noise)**: $P(S_1 = S_0 | S_0) = 0$ (density w.r.t. Lebesgue)
- **Lem 2 (Contradiction)**: Suppose period $d > 1 \Rightarrow$ disjoint classes $\mathcal{S}_0, \ldots, \mathcal{S}_{d-1}$ $\Rightarrow$ core $\mathcal{C}$ in multiple classes $\Rightarrow$ contradiction

**DEPENDENCIES FORWARD**:
- **Direct**: `thm-main-convergence` (Meyn-Tweedie requires aperiodicity)
- **Convergence without oscillations**: Exponential approach to QSD

---

### 5. thm-main-convergence (Line 906)

**TYPE**: Theorem (Main Result - Geometric Ergodicity)
**OUTPUT_TYPE**: Existence + Convergence Property

**INPUT_OBJECTS**:
- `obj-euclidean-gas-markov-chain`
- `obj-qsd`: Quasi-stationary distribution $\nu_{\text{QSD}}$
- `obj-synergistic-lyapunov`: $V_{\text{total}}$

**INPUT_AXIOMS**:
- Foundational axioms (Ch 03, Ch 04)
- All axioms required by `thm-foster-lyapunov-main`, `thm-phi-irreducibility`, `thm-aperiodicity`

**INPUT_PARAMETERS**:
- All parameters from `thm-foster-lyapunov-main`
- Swarm size: $N$

**PROPERTIES_REQUIRED**:
- **Foster-Lyapunov drift**: From `thm-foster-lyapunov-main`
- **φ-irreducibility**: From `thm-phi-irreducibility`
- **Aperiodicity**: From `thm-aperiodicity`
- **Compact level sets**: Boundary potential $W_b$ prevents escape

**PROPERTIES_ADDED**:
1. **Existence and uniqueness of QSD**: $\exists! \nu_{\text{QSD}}$ on $\Sigma_N^{\text{alive}}$
2. **Exponential convergence**: $\|\mu_t - \nu_{\text{QSD}}\|_{\text{TV}} \leq C_{\text{conv}} e^{-\kappa_{\text{QSD}} t}$
   - Rate: $\kappa_{\text{QSD}} = \Theta(\kappa_{\text{total}}\tau) > 0$
   - Constant: $C_{\text{conv}}$ depends on $\mu_0$, $V_{\text{total}}(S_0)$
3. **Exponentially long survival**: $\mathbb{E}_{\nu_{\text{QSD}}}[\tau_\dagger] = e^{\Theta(N)}$
4. **Concentration around QSD**: $P(V_{\text{total}} > (1+\epsilon)V^{\text{QSD}}|\text{survived}) \leq e^{-\Theta(N)}$

**RELATIONS_ESTABLISHED**:
- **Meyn-Tweedie framework**: Drift + irreducibility + aperiodicity $\Rightarrow$ geometric ergodicity
- **Survival time scaling**: Collective energy barrier $\propto N \Rightarrow$ extinction time exponential in $N$
- **QSD structure** (from `prop-qsd-properties`): Gibbs-like positions, Gaussian velocities

**INTERNAL_LEMMAS** (Proof sketch):
- **Lem 1 (MT existence)**: Meyn-Tweedie Theorem 14.0.1 $\Rightarrow$ unique invariant measure
- **Lem 2 (Champagnat-Villemonais)**: Absorbing case $\Rightarrow$ invariant measure is QSD
- **Lem 3 (Lyapunov decay)**: $\mathbb{E}[V_{\text{total}}(S_t)] \leq (1-\kappa\tau)^t V_0 + C/(\kappa\tau)$
- **Lem 4 (Markov coupling)**: Lyapunov decay $\Rightarrow$ TV convergence (standard)
- **Lem 5 (Survival bound)**: $P(\text{survive 1 step}|S_t) \geq 1 - e^{-\Theta(N)}$ from boundary concentration
- **Lem 6 (McDiarmid)**: Concentration via bounded differences

**DEPENDENCIES FORWARD**:
- **Ch 08 (propagation of chaos)**: Uses unique QSD and $\kappa_{\text{QSD}}$ for mean-field PDE convergence
- **Ch 07 (parameter optimization)**: Explicit $\kappa_{\text{total}}$ formulas enable tuning
- **All downstream applications**: Optimization, sampling, rare events

---

### 6. prop-qsd-properties (Line 1013)

**TYPE**: Proposition
**OUTPUT_TYPE**: Property (Distribution Structure)

**INPUT_OBJECTS**:
- `obj-qsd`: $\nu_{\text{QSD}}$ (from `thm-main-convergence`)
- Potential: $U(x)$, $\varphi_{\text{barrier}}(x)$
- Noise: $\sigma_v$

**INPUT_AXIOMS**:
- Langevin dynamics (Ch 05)
- Cloning fitness structure (Ch 03)

**INPUT_PARAMETERS**:
- Friction: $\gamma$
- Noise: $\sigma_v$
- Contraction rates: $\kappa_x$, $\gamma$
- Equilibrium constants: $C_x$, $C_v$

**PROPERTIES_REQUIRED**:
- Existence of QSD from `thm-main-convergence`
- Foster-Lyapunov drift bounds

**PROPERTIES_ADDED**:
1. **Position marginal**: $\rho_{\text{pos}}(x) \propto e^{-U(x) - \varphi_{\text{barrier}}(x)}$ (Gibbs-like)
2. **Velocity marginal**: $\rho_{\text{vel}}(v) \propto e^{-\|v\|^2/(2\sigma_v^2/\gamma)}$ (Gaussian at effective temperature $\sigma_v^2/\gamma$)
3. **Correlations decay**: $\mathbb{E}_{\nu_{\text{QSD}}}[\langle x-\bar{x}, v-\bar{v}\rangle] = O(e^{-\gamma\Delta t})$
4. **Internal variance**: $V_{\text{Var},x}^{\text{QSD}} = O(C_x/\kappa_x)$, $V_{\text{Var},v}^{\text{QSD}} = O(\sigma_v^2/\gamma)$ (both N-independent)

**RELATIONS_ESTABLISHED**:
- Positional distribution concentrates in low-potential regions (fitness-driven)
- Velocity distribution equilibrates at thermal temperature (Langevin-driven)
- Phase-space factorization: $\nu_{\text{QSD}}(x,v) \approx \rho_{\text{pos}}(x) \cdot \rho_{\text{vel}}(v)$ (weakly correlated)

**INTERNAL_LEMMAS**: None (direct from equilibrium analysis)

**DEPENDENCIES FORWARD**:
- **Optimization algorithms**: Position concentration ensures convergence to low-potential regions
- **Sampling applications**: QSD provides target distribution for rare event sampling
- **Mean-field limit** (Ch 08): QSD structure inherited by limiting PDE solution

---

### 7. thm-equilibrium-variance-bounds (Line 1055)

**TYPE**: Theorem
**OUTPUT_TYPE**: Bound (Equilibrium)

**INPUT_OBJECTS**:
- `obj-qsd`: $\nu_{\text{QSD}}$
- Lyapunov components: $V_{\text{Var},x}$, $V_{\text{Var},v}$, $W_b$

**INPUT_AXIOMS**:
- Drift inequalities from Ch 03, 05

**INPUT_PARAMETERS**:
- Cloning: $\kappa_x$, $C_x$ (from Ch 03, Theorem 10.3.1)
- Kinetic: $\gamma$, $\sigma_{\max}$, $C_v$ (from Ch 05, Theorem 3.3.1)
- Boundary: $\kappa_b$, $C_b$ (from Ch 03, Theorem 11.3.1)
- Dimension: $d$
- Time step: $\tau$

**PROPERTIES_REQUIRED**:
- Component drift inequalities:
  - $\mathbb{E}_{\text{clone}}[\Delta V_{\text{Var},x}] \leq -\kappa_x V_{\text{Var},x} + C_x$
  - $\mathbb{E}_{\text{total}}[\Delta V_{\text{Var},v}] \leq -2\gamma V_{\text{Var},v}\tau + (C_v + \sigma_{\max}^2 d\tau)$
  - $\mathbb{E}_{\text{total}}[\Delta W_b] \leq -(\kappa_b + \kappa_{\text{pot}}\tau)W_b + (C_b + C_{\text{pot}}\tau)$

**PROPERTIES_ADDED**:
1. **Positional variance bound**: $V_{\text{Var},x}^{\text{QSD}} \leq C_x/\kappa_x$
2. **Velocity variance bound**: $V_{\text{Var},v}^{\text{QSD}} \leq (C_v + \sigma_{\max}^2 d\tau)/(2\gamma\tau)$
   - Simplified (dominant noise): $V_{\text{Var},v}^{\text{QSD}} \approx d\sigma_{\max}^2/(2\gamma)$ (equipartition)
3. **Boundary potential bound**: $W_b^{\text{QSD}} \leq C_b/\kappa_b$
4. **N-uniformity**: All bounds independent of $N$ (scalable to mean-field limit)

**RELATIONS_ESTABLISHED**:
- **Equilibrium = Drift balance**: At QSD, $\mathbb{E}[\Delta V_i] = 0 \Rightarrow V_i^{\text{QSD}} = C_i/\kappa_i$
- **Contraction-expansion trade-off**:
  - Position: Cloning contraction vs. kinetic diffusion
  - Velocity: Friction dissipation vs. cloning collisions + Langevin noise
  - Boundary: Safe Harbor + confining potential vs. thermal escapes

**INTERNAL_LEMMAS**:
- **Lem 1 (Equilibrium condition)**: QSD $\Rightarrow$ $\mathbb{E}_{\nu_{\text{QSD}}}[\Delta V_i] = 0$
- **Lem 2 (Solving for equilibrium)**: $-\kappa_i V_i^{\text{eq}} + C_i = 0 \Rightarrow V_i^{\text{eq}} = C_i/\kappa_i$

**DEPENDENCIES FORWARD**:
- **Parameter optimization** (Ch 6, Section 5): Equilibrium bounds determine target variances
- **Mixing time estimates** (Prop 5.6): $T_{\text{mix}} \sim 1/\kappa_{\text{total}} \cdot \ln(V_0/V^{\text{eq}})$
- **Mean-field limit** (Ch 08): N-uniformity essential for $N \to \infty$

---

## PART III: EXPLICIT PARAMETER DEPENDENCE (Objects 8-17)

### 8. prop-velocity-rate-explicit (Line 1233)

**TYPE**: Proposition
**OUTPUT_TYPE**: Property (Explicit Formula)

**INPUT_OBJECTS**:
- Velocity variance: $V_{\text{Var},v}$
- BAOAB integrator (Ch 05)

**INPUT_AXIOMS**:
- Langevin dynamics (Ch 05)

**INPUT_PARAMETERS**:
- Friction: $\gamma$
- Noise: $\sigma_v$
- Time step: $\tau$
- Dimension: $d$

**PROPERTIES_REQUIRED**:
- BAOAB O-step: $v_{n+1/2} = e^{-\gamma\tau}v_n + \sqrt{\frac{\sigma_v^2}{\gamma}(1-e^{-2\gamma\tau})}\xi_n$

**PROPERTIES_ADDED**:
- **Dissipation rate**: $\kappa_v = 2\gamma - O(\tau)$
- **Equilibrium constant**: $C_v' = d\sigma_v^2/\gamma + O(\tau\sigma_v^2)$
- **Equilibrium variance**: $V_{\text{Var},v}^{\text{eq}} = d\sigma_v^2/\gamma(1+O(\tau))$ (Gibbs thermal variance)

**RELATIONS_ESTABLISHED**:
- $\gamma \uparrow \Rightarrow \kappa_v \uparrow$ (faster), $C_v' \downarrow$ (tighter equilibrium)
- $\sigma_v \uparrow \Rightarrow$ rate unchanged, equilibrium wider (exploration knob)
- $\tau \uparrow \Rightarrow \kappa_v \downarrow$ (discretization penalty)

**INTERNAL_LEMMAS**:
- **Lem 1 (Variance propagation)**: $\mathbb{E}[\|v_{n+1/2}\|^2] = e^{-2\gamma\tau}\mathbb{E}[\|v_n\|^2] + d\sigma_v^2(1-e^{-2\gamma\tau})/\gamma$
- **Lem 2 (Taylor expansion)**: $e^{-2\gamma\tau} = 1 - 2\gamma\tau + 2\gamma^2\tau^2 + O(\tau^3)$

**DEPENDENCIES FORWARD**:
- `thm-total-rate-explicit`: Component of $\kappa_{\text{total}}$
- `prop-parameter-classification`: Class A (direct rate controller)

---

### 9. prop-position-rate-explicit (Line 1321)

**TYPE**: Proposition
**OUTPUT_TYPE**: Property (Explicit Formula)

**INPUT_OBJECTS**:
- Positional variance: $V_{\text{Var},x}$
- Cloning operator (Ch 03)

**INPUT_AXIOMS**:
- Keystone Principle (Ch 03, Theorem 5.1)

**INPUT_PARAMETERS**:
- Cloning rate: $\lambda$
- Fitness-variance correlation: $c_{\text{fit}}(\lambda_{\text{alg}}, \epsilon_c, \epsilon_d)$
- Kinetic diffusion: $\sigma_v$, $\gamma$, $\tau$
- Jitter: $\sigma_x$

**PROPERTIES_REQUIRED**:
- Keystone Principle: $\mathbb{E}_{\text{clone}}[\Delta V_{\text{Var},x}] = -\lambda \cdot \text{Cov}(f_i, \|x_i-\bar{x}\|^2) + O(1/N)$
- Kinetic diffusion: $\mathbb{E}_{\text{kin}}[\Delta V_{\text{Var},x}] \sim \sqrt{V_{\text{Var},x}V_{\text{Var},v}}\tau$

**PROPERTIES_ADDED**:
- **Contraction rate**: $\kappa_x = \lambda \cdot c_{\text{fit}} + O(\tau)$
- **Equilibrium constant**: $C_x = O(\sigma_v^2\tau^3/\gamma) + O(\tau\sigma_x^2)$

**RELATIONS_ESTABLISHED**:
- $\lambda \uparrow \Rightarrow \kappa_x \uparrow \propto \lambda$ (proportional)
- $c_{\text{fit}} \uparrow \Rightarrow \kappa_x \uparrow$ (better pairing quality)
- $\sigma_v, \tau \uparrow \Rightarrow C_x \uparrow$ (diffusion expands)
- $N \uparrow \Rightarrow \kappa_x$ tighter estimate ($+O(1/N)$ correction)

**INTERNAL_LEMMAS**:
- **Lem 1 (Fitness-variance anti-correlation)**: Cloning preferentially removes high-variance walkers
- **Lem 2 (Kinetic expansion)**: $\Delta x = v\tau \Rightarrow \Delta V_{\text{Var},x} \sim 2\langle v, x-\bar{x}\rangle\tau$

**DEPENDENCIES FORWARD**:
- `thm-total-rate-explicit`: Component of $\kappa_{\text{total}}$
- `prop-parameter-classification`: Class A + C (rate controller + geometric)

---

### 10. prop-wasserstein-rate-explicit (Line 1419)

**TYPE**: Proposition
**OUTPUT_TYPE**: Property (Explicit Formula)

**INPUT_OBJECTS**:
- Inter-swarm Wasserstein: $V_W$
- Hypocoercive Langevin (Ch 05)

**INPUT_AXIOMS**:
- Hypocoercivity theory (Ch 05, Section 2.3)

**INPUT_PARAMETERS**:
- Friction: $\gamma$
- Noise: $\sigma_v$
- Hessian eigenvalue: $\lambda_{\min}$
- Hypocoercivity constant: $c_{\text{hypo}} \sim 0.1-1$
- Time step: $\tau$
- Swarm size: $N$
- Dimension: $d$

**PROPERTIES_REQUIRED**:
- Hypocoercive contraction (Ch 05): Velocity equilibration (rate $\sim \gamma$) + positional mixing (rate $\sim \lambda_{\min}$)

**PROPERTIES_ADDED**:
- **Contraction rate**: $\kappa_W = \frac{c_{\text{hypo}}^2\gamma}{1+\gamma/\lambda_{\min}}$
  - Underdamped ($\gamma \ll \lambda_{\min}$): $\kappa_W \sim \gamma$
  - Overdamped ($\gamma \gg \lambda_{\min}$): $\kappa_W \sim \lambda_{\min}$
- **Equilibrium constant**: $C_W' = O(\sigma_v^2\tau/N^{1/d}) + O(\tau^2)$

**RELATIONS_ESTABLISHED**:
- $\gamma \uparrow \Rightarrow \kappa_W \uparrow$ (up to $\sim \lambda_{\min}$, then saturates)
- **Optimal friction**: $\gamma^* = \lambda_{\min}$ (balanced underdamped/overdamped)
- $N \uparrow \Rightarrow C_W' \downarrow \propto N^{-1/d}$ (law of large numbers)
- $\lambda_{\min} \uparrow \Rightarrow \kappa_W \uparrow$ (smoother landscape faster)

**INTERNAL_LEMMAS**:
- **Lem 1 (Hörmander theorem)**: Hypoelliptic SDE $\Rightarrow$ positive transition density
- **Lem 2 (Hypocoercivity formula)**: $\kappa_W = c^2 \gamma\lambda_{\min}/(\gamma+\lambda_{\min})$
- **Lem 3 (Empirical measure fluctuations)**: $\Delta W_2 \sim N^{-1/d}\sigma_v\sqrt{\tau}$ (CLT)

**DEPENDENCIES FORWARD**:
- `thm-total-rate-explicit`: Component of $\kappa_{\text{total}}$ (often bottleneck for ill-conditioned landscapes)
- Parameter optimization: Key constraint for rough potentials

---

### 11. prop-boundary-rate-explicit (Line 1518)

**TYPE**: Proposition
**OUTPUT_TYPE**: Property (Explicit Formula)

**INPUT_OBJECTS**:
- Boundary potential: $W_b$
- Safe Harbor (Ch 03, Ch 7)
- Confining potential (Ch 05)

**INPUT_AXIOMS**:
- Safe Harbor mechanism (Ch 03, Section 7)
- Confining potential contraction (Ch 05, Section 5.3)

**INPUT_PARAMETERS**:
- Cloning rate: $\lambda$
- Fitness gap: $\Delta f_{\text{boundary}} = f(\text{interior}) - f(\text{boundary})$
- Boundary stiffness: $\kappa_{\text{wall}}$
- Friction: $\gamma$
- Noise: $\sigma_v$, $\tau$
- Safe Harbor distance: $d_{\text{safe}}$

**PROPERTIES_REQUIRED**:
- Safe Harbor: Cloning removes boundary walkers at rate $\sim \lambda \Delta f/f_{\text{typical}}$
- Confining potential: Kinetic contraction rate $\kappa_{\text{pot}} + \gamma$

**PROPERTIES_ADDED**:
- **Contraction rate**: $\kappa_b = \min(\lambda \cdot \frac{\Delta f_{\text{boundary}}}{f_{\text{typical}}}, \kappa_{\text{wall}} + \gamma)$
  - Cloning-limited: $\kappa_b \propto \lambda$
  - Kinetic-limited: $\kappa_b \propto \kappa_{\text{wall}} + \gamma$
- **Equilibrium constant**: $C_b = O(\sigma_v^2\tau/d_{\text{safe}}^2) + O(\tau^2)$

**RELATIONS_ESTABLISHED**:
- $\lambda \uparrow \Rightarrow \kappa_b \uparrow$ (if cloning-limited)
- $\kappa_{\text{wall}} \uparrow \Rightarrow \kappa_b \uparrow$ (if kinetic-limited)
- $d_{\text{safe}} \uparrow \Rightarrow C_b \downarrow \propto 1/d_{\text{safe}}^2$ (thermal escape harder)
- $\sigma_v \uparrow \Rightarrow C_b \uparrow$ (more thermal kicks toward boundary)

**INTERNAL_LEMMAS**:
- **Lem 1 (Fitness deficit)**: Near boundary, $\Delta f \sim \kappa_{\text{wall}}(x-\bar{x})^2$
- **Lem 2 (Thermal escape probability)**: $P(\text{reach boundary}) \sim \sigma_v\tau^{3/2}/(\sqrt{\gamma}d_{\text{safe}})$

**DEPENDENCIES FORWARD**:
- `thm-total-rate-explicit`: Component of $\kappa_{\text{total}}$
- Safety analysis: Ensures $P(\text{extinction}) \ll 1$

---

### 12. thm-synergistic-rate-derivation (Line 1621)

**TYPE**: Theorem
**OUTPUT_TYPE**: Property (Synergistic Drift)

**INPUT_OBJECTS**:
- All Lyapunov components: $V_{\text{Var},x}$, $V_{\text{Var},v}$, $V_W$, $W_b$
- Weighted Lyapunov: $V_{\text{total}} = V_{\text{Var},x} + \alpha_v V_{\text{Var},v} + \alpha_W V_W + \alpha_b W_b$

**INPUT_AXIOMS**:
- Component drift inequalities from Ch 03, 05

**INPUT_PARAMETERS**:
- Component rates: $\kappa_x$, $\kappa_v$, $\kappa_W$, $\kappa_b$
- Component constants: $C_x$, $C_v$, $C_W$, $C_b$
- **Cross-coupling**: $C_{xv}$, $C_{xW}$, $C_{vx}$ (expansion from complementary operator)
- Weights: $\alpha_v$, $\alpha_W$, $\alpha_b$ (to be chosen)

**PROPERTIES_REQUIRED**:
- Component drifts with cross-terms:
  - $\mathbb{E}_{\text{clone}}[\Delta V_{\text{Var},x}] \leq -\kappa_x V_{\text{Var},x} + C_x + C_{xv}V_{\text{Var},v} + C_{xW}V_W$
  - $\mathbb{E}_{\text{kin}}[\Delta V_{\text{Var},v}] \leq -\kappa_v V_{\text{Var},v} + C_v + C_{vx}V_{\text{Var},x}$

**PROPERTIES_ADDED**:
- **Synergistic rate**: $\kappa_{\text{total}} = \min(\kappa_x, \alpha_v\kappa_v, \alpha_W\kappa_W, \alpha_b\kappa_b) \cdot (1-\epsilon_{\text{coupling}})$
- **Coupling penalty**: $\epsilon_{\text{coupling}} = \max(\frac{\alpha_v C_{xv}}{\kappa_v V_{\text{Var},v}}, \frac{\alpha_W C_{xW}}{\kappa_W V_W}, \ldots) \ll 1$
- **Total constant**: $C_{\text{total}} = C_x + \alpha_v C_v + \alpha_W C_W + \alpha_b C_b$
- **Weight selection**: $\alpha_v \geq C_{xv}/(\kappa_v V_{\text{Var},v}^{\text{eq}})$ (ensure coupling dominated by contraction)

**RELATIONS_ESTABLISHED**:
- **Bottleneck principle**: Total rate = minimum component rate
- **Coupling domination**: Weights chosen so cross-terms $< $ contraction
- **Synergy**: Each operator corrects other's expansion $\Rightarrow$ net contraction

**INTERNAL_LEMMAS**:
- **Lem 1 (Weighted combination)**: $\mathbb{E}[\Delta V_{\text{total}}] = \sum_i w_i \mathbb{E}[\Delta V_i]$
- **Lem 2 (Coupling balance)**: Choose $\alpha_i$ so $C_{ij}V_j - \alpha_j\kappa_j V_j \leq -\epsilon_j\alpha_j\kappa_j V_j$

**DEPENDENCIES FORWARD**:
- `thm-total-rate-explicit`: Provides explicit formulas for $\kappa_{\text{total}}$
- Parameter optimization: Guides weight selection

---

### 13. thm-total-rate-explicit (Line 1715)

**TYPE**: Theorem
**OUTPUT_TYPE**: Property (Explicit Formula)

**INPUT_OBJECTS**:
- All component rates from Props 8-11
- Synergistic framework from `thm-synergistic-rate-derivation`

**INPUT_AXIOMS**:
- All prerequisite axioms from component rate propositions

**INPUT_PARAMETERS**:
- **ALL primitive parameters**:
  - Cloning: $\lambda$, $\sigma_x$, $\alpha_{\text{rest}}$, $\lambda_{\text{alg}}$, $\epsilon_c$, $\epsilon_d$
  - Kinetic: $\gamma$, $\sigma_v$, $\tau$
  - System: $N$, $\kappa_{\text{wall}}$, $d_{\text{safe}}$
  - Landscape: $\lambda_{\min}$, $\lambda_{\max}$, $d$

**PROPERTIES_REQUIRED**:
- Component rates: $\kappa_x, \kappa_v, \kappa_W, \kappa_b$ (from Props 8-11)
- Coupling penalty: $\epsilon_{\text{coupling}} \ll 1$ (from `thm-synergistic-rate-derivation`)

**PROPERTIES_ADDED**:
- **Total rate formula**:
  $$\kappa_{\text{total}} = \min(\kappa_x, \kappa_v, \kappa_W, \kappa_b) \cdot (1-\epsilon_{\text{coupling}})$$
  $$\sim \min(\lambda c_{\text{fit}}, 2\gamma, \frac{c_{\text{hypo}}^2\gamma}{1+\gamma/\lambda_{\min}}, \lambda\frac{\Delta f_{\text{boundary}}}{f_{\text{typical}}}) \cdot (1-O(\tau))$$
- **Total constant**:
  $$C_{\text{total}} \sim \frac{1}{\kappa_{\text{total}}}\left(\frac{\sigma_v^2\tau^2}{\gamma\lambda} + \frac{d\sigma_v^2}{\gamma} + \frac{\sigma_v^2\tau}{N^{1/d}} + \frac{\sigma_v^2\tau}{d_{\text{safe}}^2}\right)$$

**RELATIONS_ESTABLISHED**:
- **Parameter effects on total rate**:

| Parameter | Effect on $\kappa_{\text{total}}$ | Bottleneck component |
|-----------|-----------------------------------|---------------------|
| $\gamma \uparrow$ | ✅ Faster (if $\gamma < \lambda, \lambda_{\min}$) | Velocity or Wasserstein |
| $\lambda \uparrow$ | ✅ Faster (if $\lambda < \gamma$) | Position or Boundary |
| $\sigma_v \uparrow$ | ➖ No direct effect on rate | (Affects equilibrium only) |
| $\tau \uparrow$ | ❌ Slower ($-O(\tau)$ correction) | All components |
| $N \uparrow$ | ➖ No direct effect | (Tightens Wasserstein equilibrium) |

- **Optimal balanced scaling**: $\lambda \sim \gamma \sim \lambda_{\min}$ (no single bottleneck)

**INTERNAL_LEMMAS**:
- **Lem 1 (Minimum dominates)**: $\kappa_{\text{total}} = \min_i(\kappa_i)$ when coupling dominated
- **Lem 2 (Equilibrium balance)**: $V_{\text{total}}^{\text{eq}} = C_{\text{total}}/\kappa_{\text{total}}$

**DEPENDENCIES FORWARD**:
- `prop-mixing-time-explicit`: Uses $\kappa_{\text{total}}$ for convergence time
- **ALL parameter optimization** (Section 6): Requires explicit formulas
- **Ch 08**: Mean-field limit uses $\kappa_{\text{total}}$ for PDE convergence rate

---

### 14. prop-mixing-time-explicit (Line 1821)

**TYPE**: Proposition
**OUTPUT_TYPE**: Bound (Convergence Time)

**INPUT_OBJECTS**:
- Total Lyapunov: $V_{\text{total}}$
- Total rate: $\kappa_{\text{total}}$ (from `thm-total-rate-explicit`)
- Total constant: $C_{\text{total}}$

**INPUT_AXIOMS**:
- Foster-Lyapunov drift (from `thm-foster-lyapunov-main`)

**INPUT_PARAMETERS**:
- $\kappa_{\text{total}}$ (from all component parameters)
- Target accuracy: $\epsilon$ (typically 0.01 for 99% convergence)
- Initial Lyapunov value: $V_{\text{total}}^{\text{init}}$

**PROPERTIES_REQUIRED**:
- Drift inequality: $\mathbb{E}[V_{\text{total}}(t)] \leq e^{-\kappa_{\text{total}}t}V^{\text{init}} + \frac{C_{\text{total}}}{\kappa_{\text{total}}}(1-e^{-\kappa_{\text{total}}t})$

**PROPERTIES_ADDED**:
- **Mixing time formula**:
  $$T_{\text{mix}}(\epsilon) = \frac{1}{\kappa_{\text{total}}}\ln\left(\frac{V_{\text{total}}^{\text{init}}}{\epsilon C_{\text{total}}}\right)$$
- **Typical case** ($V^{\text{init}} \sim O(1)$, $\epsilon = 0.01$):
  $$T_{\text{mix}} \sim \frac{5}{\kappa_{\text{total}}} = \frac{5}{\min(\lambda c_{\text{fit}}, 2\gamma, \kappa_W, \kappa_b)}$$

**RELATIONS_ESTABLISHED**:
- $\kappa_{\text{total}} \uparrow \Rightarrow T_{\text{mix}} \downarrow \propto 1/\kappa_{\text{total}}$ (inversely proportional)
- **Numerical examples**:

| Setup | $\kappa_{\text{total}}$ | $T_{\text{mix}}$ (steps) | $T_{\text{mix}}$ (time) |
|-------|------------------------|-------------------------|------------------------|
| Fast smooth | 1.0 | 500 | 5.0 |
| Slow smooth | 0.2 | 2500 | 25.0 |
| Fast rough | 0.5 | 1000 | 10.0 |
| Underdamped | 0.1 | 5000 | 50.0 |

**INTERNAL_LEMMAS**:
- **Lem 1 (Lyapunov decay)**: Iterate drift inequality $t$ times
- **Lem 2 (Error bound)**: $|V(t) - V^{\text{eq}}| \leq e^{-\kappa t}V^{\text{init}}$
- **Lem 3 (Solve for $\epsilon$-accuracy)**: $e^{-\kappa T}V^{\text{init}} = \epsilon V^{\text{eq}}$

**DEPENDENCIES FORWARD**:
- **Computational cost estimation**: $\text{Cost} = T_{\text{mix}} \cdot O(N(d+\log N))$
- **Algorithm comparison**: Provides rigorous convergence guarantees vs. heuristics
- **Practical tuning**: Choose parameters to meet target $T_{\text{mix}}$

---

### 15. prop-parameter-classification (Line 2200)

**TYPE**: Proposition (Taxonomy)
**OUTPUT_TYPE**: Relation (Classification)

**INPUT_OBJECTS**:
- **Complete parameter vector** (12 parameters):
  - Cloning: $\lambda$, $\sigma_x$, $\alpha_{\text{rest}}$, $\lambda_{\text{alg}}$, $\epsilon_c$, $\epsilon_d$
  - Kinetic: $\gamma$, $\sigma_v$, $\tau$
  - System: $N$, $\kappa_{\text{wall}}$, $d_{\text{safe}}$

**INPUT_AXIOMS**:
- All component rate formulas (Props 8-11)

**INPUT_PARAMETERS**: (Same as input objects)

**PROPERTIES_REQUIRED**:
- Explicit rate formulas from `thm-total-rate-explicit`
- Equilibrium formulas from `thm-equilibrium-variance-bounds`

**PROPERTIES_ADDED**:
- **Class A: Direct Rate Controllers** (first-order effects on rates)
  - $\lambda \to \kappa_x, \kappa_b$ (proportional)
  - $\gamma \to \kappa_v, \kappa_W, \kappa_b$ (proportional)
  - $\kappa_{\text{wall}} \to \kappa_b$ (additive)
- **Class B: Indirect Rate Modifiers** (second-order, via equilibrium)
  - $\alpha_{\text{rest}} \to C_v$ (collision energy retention)
  - $\sigma_x \to C_x, C_b$ (jitter noise)
  - $\tau \to \kappa_i$ (penalty $-O(\tau)$), $C_i$ (noise accumulation $+O(\tau)$)
- **Class C: Geometric Structure Parameters** (fitness-variance correlation)
  - $\lambda_{\text{alg}} \to c_{\text{fit}} \to \kappa_x$ (pairing quality)
  - $\epsilon_c, \epsilon_d \to c_{\text{fit}} \to \kappa_x$ (selectivity)
- **Class D: Pure Equilibrium Parameters** (no rate effect)
  - $\sigma_v \to C_i$ for all $i$ (thermal noise)
  - $N \to C_W$ (law of large numbers: $\propto N^{-1/d}$)
- **Class E: Safety/Feasibility Constraints**
  - $d_{\text{safe}} \to C_b$ (thermal escape probability)

**RELATIONS_ESTABLISHED**:
- **Effective control dimensions**: Classes A + C control rates (4 effective dimensions)
- **Null space dimension**: Classes B, D, E provide 8 degrees of freedom for secondary objectives (cost, robustness, exploration)
- **Parameter coupling**: Cannot tune independently (e.g., $\alpha_{\text{rest}}$ requires adjusting $\gamma$ to maintain $V_{\text{Var},v}^{\text{eq}}$)

**INTERNAL_LEMMAS**: None (classification based on explicit formulas)

**DEPENDENCIES FORWARD**:
- `thm-svd-rate-matrix`: Sensitivity matrix structure reflects classification
- **Optimization algorithms**: Exploit null space for multi-objective optimization
- **Pareto frontier**: Classes D, E traded off along Pareto curve

---

### 16. thm-explicit-rate-sensitivity (Line 2384)

**TYPE**: Theorem
**OUTPUT_TYPE**: Property (Sensitivity Matrix)

**INPUT_OBJECTS**:
- Parameter vector $\mathbf{P} \in \mathbb{R}^{12}$
- Rate vector $\boldsymbol{\kappa} = (\kappa_x, \kappa_v, \kappa_W, \kappa_b)$

**INPUT_AXIOMS**:
- Component rate formulas (Props 8-11)

**INPUT_PARAMETERS**:
- Balanced operating point: $\gamma \approx \lambda \approx \sqrt{\lambda_{\min}}$, $\lambda_{\text{alg}} = 0.1$, $\tau = 0.01$

**PROPERTIES_REQUIRED**:
- Differentiability of rate functions $\kappa_i(\mathbf{P})$

**PROPERTIES_ADDED**:
- **Rate sensitivity matrix** $M_\kappa \in \mathbb{R}^{4 \times 12}$:
  $$(M_\kappa)_{ij} = \frac{\partial \log \kappa_i}{\partial \log P_j}\bigg|_{P_0}$$
  (log-elasticity: 1% change in $P_j$ → $(M_\kappa)_{ij}$% change in $\kappa_i$)

- **Explicit matrix** (at balanced point):
  $$M_\kappa = \begin{bmatrix}
  1.0 & 0 & 0 & 0.3 & -0.3 & 0 & 0 & 0 & -0.1 & 0 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0 & 0 & 1.0 & 0 & -0.1 & 0 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0 & 0 & 0.5 & 0 & 0 & 0 & 0 & 0 \\
  0.5 & 0 & 0 & 0 & 0 & 0 & 0.3 & 0 & 0 & 0 & 0.4 & 0
  \end{bmatrix}$$
  Rows: $(\kappa_x, \kappa_v, \kappa_W, \kappa_b)$
  Columns: $(\lambda, \sigma_x, \alpha_{\text{rest}}, \lambda_{\text{alg}}, \epsilon_c, \epsilon_d, \gamma, \sigma_v, \tau, N, \kappa_{\text{wall}}, d_{\text{safe}})$

**RELATIONS_ESTABLISHED**:
- **Column 1 ($\lambda$)**: Strong effect on $\kappa_x$ (1.0), moderate on $\kappa_b$ (0.5)
- **Column 7 ($\gamma$)**: Strong on $\kappa_v$ (1.0), moderate on $\kappa_W$ (0.5), $\kappa_b$ (0.3)
- **Column 4 ($\lambda_{\text{alg}}$), Column 5 ($\epsilon_c$)**: Moderate on $\kappa_x$ via pairing quality
- **Columns 2, 3, 6, 8, 10, 12**: Zero entries (Classes B, D, E don't affect rates)

**INTERNAL_LEMMAS**:
- **Lem 1 (Log-derivative)**: $\frac{\partial \log \kappa}{\partial \log P} = \frac{P}{\kappa}\frac{\partial \kappa}{\partial P}$
- **Lem 2 (Chain rule)**: $\kappa_x = \lambda \cdot c_{\text{fit}}(\lambda_{\text{alg}}) \Rightarrow \frac{\partial \log \kappa_x}{\partial \log \lambda_{\text{alg}}} = \frac{\partial \log c_{\text{fit}}}{\partial \log \lambda_{\text{alg}}}$
- **Lem 3 (Hypocoercivity sensitivity)**: $\frac{\partial \log \kappa_W}{\partial \log \gamma} = \frac{\lambda_{\min}}{\gamma + \lambda_{\min}}$

**DEPENDENCIES FORWARD**:
- `thm-svd-rate-matrix`: SVD decomposition of $M_\kappa$
- `thm-error-propagation`: Parameter error bounds using $\|M_\kappa\|$
- **Optimization**: Gradient ascent direction = row of $M_\kappa$ for bottleneck

---

### 17. thm-svd-rate-matrix (Line 2483)

**TYPE**: Theorem
**OUTPUT_TYPE**: Property (Spectral Decomposition)

**INPUT_OBJECTS**:
- Rate sensitivity matrix: $M_\kappa \in \mathbb{R}^{4 \times 12}$ (from `thm-explicit-rate-sensitivity`)

**INPUT_AXIOMS**:
- Spectral theorem for rectangular matrices

**INPUT_PARAMETERS**: (Embedded in $M_\kappa$)

**PROPERTIES_REQUIRED**:
- $M_\kappa$ is real, rank 4

**PROPERTIES_ADDED**:
- **SVD decomposition**: $M_\kappa = U\Sigma V^T$
  - $U \in \mathbb{R}^{4 \times 4}$ (orthonormal, rate space)
  - $\Sigma \in \mathbb{R}^{4 \times 12}$ (diagonal: $\sigma_1 \geq \sigma_2 \geq \sigma_3 \geq \sigma_4 > 0$)
  - $V \in \mathbb{R}^{12 \times 12}$ (orthonormal, parameter space)

- **Singular values**: $\sigma_1 \approx 1.58$, $\sigma_2 \approx 1.12$, $\sigma_3 \approx 0.76$, $\sigma_4 \approx 0.29$

- **Principal right singular vectors** (parameter space modes):
  - **Mode 1** ($v_1$, $\sigma_1 = 1.58$): **Balanced kinetic control**
    - $v_1 \approx (0.52\lambda, 0, 0, 0.12\lambda_{\text{alg}}, -0.12\epsilon_c, 0, 0.61\gamma, 0, -0.05\tau, 0, 0, 0)$
    - Simultaneously increase $\lambda$ and $\gamma$ in balanced proportion
    - **Most powerful control mode** (largest $\sigma$)

  - **Mode 2** ($v_2$, $\sigma_2 = 1.12$): **Boundary safety control**
    - $v_2 \approx (0.42\lambda, 0, 0, 0, 0, 0, 0.22\gamma, 0, 0, 0, 0.85\kappa_{\text{wall}}, 0)$
    - Increase boundary protection ($\lambda, \gamma, \kappa_{\text{wall}}$)

  - **Mode 3** ($v_3$, $\sigma_3 = 0.76$): **Geometric fine-tuning**
    - $v_3 \approx (0.15\lambda, 0, 0, 0.81\lambda_{\text{alg}}, -0.56\epsilon_c, 0, 0.05\gamma, 0, 0, 0, 0, 0)$
    - Optimize companion selection quality

  - **Mode 4** ($v_4$, $\sigma_4 = 0.29$): **Timestep penalty**
    - $v_4 \approx (0, 0, 0, 0, 0, 0, 0, 0, -1.0\tau, 0, 0, 0)$
    - Pure degradation mode (minimize $\tau$ subject to stability)

- **Null space** ($v_5, \ldots, v_{12}$, dimension 8):
  - $v_5 \approx (0, 1\sigma_x, 0, \ldots)$ (position jitter)
  - $v_6 \approx (0, 0, 1\alpha_{\text{rest}}, 0, \ldots)$ (restitution)
  - $v_7 \approx (0, 0, 0, 0, 0, 0, 0, 1\sigma_v, 0, \ldots)$ (exploration noise)
  - $v_8 \approx (0, 0, 0, 0, 0, 0, 0, 0, 0, 1N, 0, 0)$ (swarm size)
  - $v_9 \approx (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1d_{\text{safe}})$ (safety buffer)
  - **Zero effect on convergence rates** (only affect equilibrium, cost, safety)

**RELATIONS_ESTABLISHED**:
- **Control hierarchy**: Mode 1 ($\lambda, \gamma$ balance) > Mode 2 (boundary) > Mode 3 (geometric) > Mode 4 (timestep)
- **Null space interpretation**: 8-dimensional family of parameters achieving identical rates but different secondary objectives
- **Optimization strategy**: Move primarily along $v_1$ for maximum rate improvement

**INTERNAL_LEMMAS**:
- **Lem 1 (SVD existence)**: Real matrix $\Rightarrow$ SVD exists
- **Lem 2 (Rank)**: $M_\kappa$ has rank 4 (4 independent rows, 12 columns)
- **Lem 3 (Null space dimension)**: $\dim(\text{Null}(M_\kappa)) = 12 - 4 = 8$

**DEPENDENCIES FORWARD**:
- `prop-condition-number-rate`: Uses $\sigma_1/\sigma_4$ for robustness
- **Optimization algorithms**: Gradient ascent along principal modes
- **Parameter tuning**: Exploit null space for multi-objective optimization

---

## PART IV: SPECTRAL OPTIMIZATION (Objects 18-23)

### 18. prop-condition-number-rate (Line 2580)

**TYPE**: Proposition
**OUTPUT_TYPE**: Bound (Robustness)

**INPUT_OBJECTS**:
- $M_\kappa$ and its singular values (from `thm-svd-rate-matrix`)

**INPUT_AXIOMS**:
- Definition of condition number: $\kappa(A) = \sigma_{\max}/\sigma_{\min}$

**INPUT_PARAMETERS**: (Embedded in $M_\kappa$)

**PROPERTIES_REQUIRED**:
- Singular values: $\sigma_1 = 1.58$, $\sigma_4 = 0.29$

**PROPERTIES_ADDED**:
- **Condition number**: $\kappa(M_\kappa) = \sigma_1/\sigma_4 = 1.58/0.29 \approx 5.4$
- **Classification**: **Moderately well-conditioned**
  - Not too sensitive ($\kappa > 100$ would be ill-conditioned)
  - Not too insensitive ($\kappa < 2$ would be trivial)

**RELATIONS_ESTABLISHED**:
- **Numerical stability**: Parameter optimization is stable
- **Error amplification**: Small parameter errors → proportionally small rate errors (factor $\sim 5.4$)
- **Practical implication**: ±10% parameter errors → ≤25% rate degradation (acceptable)

**INTERNAL_LEMMAS**:
- **Lem 1 (Condition number definition)**: $\kappa(M) = \|M\| \cdot \|M^{-1}\|$ (for square matrices; generalized for rectangular)

**DEPENDENCIES FORWARD**:
- `thm-error-propagation`: Uses $\kappa(M_\kappa)$ for error bounds
- **Robustness design guidelines**: Target ±10% parameter precision

---

### 19. thm-subgradient-min (Line 2634)

**TYPE**: Theorem
**OUTPUT_TYPE**: Property (Convex Analysis)

**INPUT_OBJECTS**:
- Objective function: $\kappa_{\text{total}}(\mathbf{P}) = \min(\kappa_1(\mathbf{P}), \ldots, \kappa_4(\mathbf{P}))$

**INPUT_AXIOMS**:
- Convex analysis: Subgradient calculus for $\min()$ function

**INPUT_PARAMETERS**:
- Parameter vector $\mathbf{P}$

**PROPERTIES_REQUIRED**:
- $\min()$ function is concave
- Component functions $\kappa_i(\mathbf{P})$ are differentiable

**PROPERTIES_ADDED**:
- **Subgradient set**:
  $$\partial \kappa_{\text{total}}(\mathbf{P}) = \text{conv}\{\nabla \kappa_i : \kappa_i(\mathbf{P}) = \kappa_{\text{total}}(\mathbf{P})\}$$
  (convex hull of gradients of active constraints)

- **Cases**:
  1. **Unique minimum** (e.g., $\kappa_x < \kappa_v, \kappa_W, \kappa_b$):
     $\partial \kappa_{\text{total}} = \{\nabla \kappa_x\}$ (unique gradient)

  2. **Two-way tie** (e.g., $\kappa_x = \kappa_v < \kappa_W, \kappa_b$):
     $\partial \kappa_{\text{total}} = \{\alpha\nabla\kappa_x + (1-\alpha)\nabla\kappa_v : \alpha \in [0,1]\}$

  3. **Four-way tie** ($\kappa_x = \kappa_v = \kappa_W = \kappa_b$):
     $\partial \kappa_{\text{total}} = \{\sum_{i=1}^4 \alpha_i\nabla\kappa_i : \alpha_i \geq 0, \sum\alpha_i = 1\}$

**RELATIONS_ESTABLISHED**:
- **Optimality condition**: $0 \in \partial \kappa_{\text{total}}(\mathbf{P}^*) \Rightarrow \mathbf{P}^*$ is local maximum
- **Non-smooth optimization**: Use subgradient methods at corners

**INTERNAL_LEMMAS**:
- **Lem 1 (Concavity of min)**: $\min_i f_i$ is concave if all $f_i$ are concave
- **Lem 2 (Subgradient definition)**: $\partial f(x) = \{g : f(y) \leq f(x) + \langle g, y-x\rangle \, \forall y\}$

**DEPENDENCIES FORWARD**:
- `thm-balanced-optimality`: Uses subgradient to prove balanced rates at optimum
- **Optimization algorithms**: Gradient ascent with subgradient when tie occurs

---

### 20. thm-balanced-optimality (Line 2667)

**TYPE**: Theorem
**OUTPUT_TYPE**: Property (Necessary Condition)

**INPUT_OBJECTS**:
- Objective: $\kappa_{\text{total}}(\mathbf{P}) = \min(\kappa_x, \kappa_v, \kappa_W, \kappa_b)$
- Subgradient (from `thm-subgradient-min`)

**INPUT_AXIOMS**:
- First-order optimality conditions
- Component rate formulas (Props 8-11)

**INPUT_PARAMETERS**:
- Parameter vector $\mathbf{P}$

**PROPERTIES_REQUIRED**:
- $\kappa_i(\mathbf{P})$ are smooth functions (differentiable)
- Interior point $\mathbf{P}^*$ (not on boundary of feasible region)

**PROPERTIES_ADDED**:
- **Balanced optimality condition**: If $\mathbf{P}^*$ is a local maximum in the interior, then **at least two rates must be equal**:
  $$\exists i \neq j: \kappa_i(\mathbf{P}^*) = \kappa_j(\mathbf{P}^*) = \kappa_{\text{total}}(\mathbf{P}^*)$$

**RELATIONS_ESTABLISHED**:
- **Proof by contradiction**: Suppose all rates strictly distinct $\Rightarrow$ unique minimum $\kappa_1 < \kappa_2, \kappa_3, \kappa_4$
  - $\Rightarrow$ Subgradient is $\{\nabla\kappa_1\}$ (unique)
  - $\Rightarrow$ For optimum, need $\nabla\kappa_1 = 0$
  - $\Rightarrow$ But $\partial\kappa_1/\partial\lambda = c_{\text{fit}} > 0$ (can increase $\kappa_1$ by increasing $\lambda$)
  - $\Rightarrow$ Contradicts local maximum assumption
  - $\Rightarrow$ **At least two rates must be equal**

- **Geometric interpretation**: Optimal point lies on **corner** or **edge** where rate surfaces intersect

- **Typical optimal configurations**:
  1. **Two-way balanced**: $\kappa_x = \kappa_v < \kappa_W, \kappa_b$ (friction-cloning balance)
  2. **Three-way balanced**: $\kappa_x = \kappa_v = \kappa_W < \kappa_b$ (kinetic balance)
  3. **Four-way balanced**: $\kappa_x = \kappa_v = \kappa_W = \kappa_b$ (fully balanced, rare)

**INTERNAL_LEMMAS**:
- **Lem 1 (First-order condition)**: Local max $\Rightarrow$ $0 \in \partial f(\mathbf{P}^*)$
- **Lem 2 (Gradient nonzero)**: $\nabla\kappa_i \neq 0$ for Class A parameters
- **Lem 3 (Contradiction)**: Unique minimum + gradient nonzero $\Rightarrow$ can improve $\Rightarrow$ not optimal

**DEPENDENCIES FORWARD**:
- **Parameter optimization**: Guides search toward balanced manifolds
- `thm-closed-form-optimum`: Uses two-way balance ($\kappa_x = \kappa_v$) for closed-form solution
- **Hessian analysis**: Compute curvature on balanced manifold

---

### 21. prop-restitution-friction-coupling (Line 2774)

**TYPE**: Proposition
**OUTPUT_TYPE**: Relation (Coupling)

**INPUT_OBJECTS**:
- Velocity equilibrium: $V_{\text{Var},v}^{\text{eq}}$
- Restitution coefficient: $\alpha_{\text{rest}}$
- Friction: $\gamma$

**INPUT_AXIOMS**:
- Velocity equilibrium formula (from `thm-equilibrium-variance-bounds`)

**INPUT_PARAMETERS**:
- $\alpha_{\text{rest}} \in [0,1]$
- $\gamma > 0$
- $\sigma_v > 0$
- $d$ (dimension)
- Target: $V_{\text{eq}}^{\text{target}}$

**PROPERTIES_REQUIRED**:
- $V_{\text{Var},v}^{\text{eq}} = \frac{C_v(\alpha_{\text{rest}})}{\kappa_v(\gamma)} = \frac{d\sigma_v^2}{\gamma}(1 + f(\alpha_{\text{rest}}))$
- Collision energy retention: $f(\alpha) \approx \frac{\alpha^2}{2-\alpha^2}$

**PROPERTIES_ADDED**:
- **Optimal friction formula**:
  $$\gamma^*(\alpha_{\text{rest}}) = \frac{d\sigma_v^2}{V_{\text{eq}}^{\text{target}}} \cdot (1 + f(\alpha_{\text{rest}})) = \frac{d\sigma_v^2}{V_{\text{eq}}^{\text{target}}} \cdot \frac{2}{2-\alpha_{\text{rest}}^2}$$

- **Extreme cases**:
  - Perfectly inelastic ($\alpha = 0$): $\gamma^* = d\sigma_v^2/V_{\text{eq}}^{\text{target}}$ (minimum friction)
  - Perfectly elastic ($\alpha = 1$): $\gamma^* = 2d\sigma_v^2/V_{\text{eq}}^{\text{target}}$ (need double friction)

- **Trade-off curve** ($V_{\text{eq}} = 0.1$, $\sigma_v = 0.2$, $d = 10$):

| $\alpha_{\text{rest}}$ | $f(\alpha)$ | $\gamma^*$ | Cost | Exploration |
|------------------------|-------------|------------|------|-------------|
| 0.0 (inelastic)        | 0.0         | 0.40       | Low  | Low         |
| 0.3                    | 0.047       | 0.42       | Low  | Moderate    |
| 0.5                    | 0.143       | 0.46       | Mod  | Moderate    |
| 0.7                    | 0.326       | 0.53       | High | High        |
| 1.0 (elastic)          | 1.0         | 0.80       | High | Very High   |

**RELATIONS_ESTABLISHED**:
- **Low $\alpha$**: Cheap (low friction) but poor exploration (kinetic energy dissipates)
- **High $\alpha$**: Expensive (high friction) but rich exploration (kinetic energy preserved)
- **Optimal for most problems**: $\alpha \approx 0.3-0.5$ (moderate dissipation)

**INTERNAL_LEMMAS**:
- **Lem 1 (Collision energy)**: $f(\alpha) = \mathbb{E}[\text{kinetic energy retained}/\text{kinetic energy before collision}]$
- **Lem 2 (Random rotations)**: Average over isotropic velocities → $f(\alpha) \sim \alpha^2/(2-\alpha^2)$

**DEPENDENCIES FORWARD**:
- **Parameter tuning**: Cannot set $\alpha_{\text{rest}}$ and $\gamma$ independently
- **Multi-objective optimization**: Trade exploration (high $\alpha$) vs. cost (low $\gamma$)

---

### 22. prop-jitter-cloning-coupling (Line 2829)

**TYPE**: Proposition
**OUTPUT_TYPE**: Relation (Coupling)

**INPUT_OBJECTS**:
- Positional equilibrium: $V_{\text{Var},x}^{\text{eq}}$
- Jitter: $\sigma_x$
- Cloning rate: $\lambda$

**INPUT_AXIOMS**:
- Positional equilibrium formula (from `thm-equilibrium-variance-bounds`)

**INPUT_PARAMETERS**:
- $\sigma_x > 0$
- $\lambda > 0$
- $\sigma_v$, $\gamma$, $\tau$
- Target: $V_{\text{Var},x}^{\text{target}}$

**PROPERTIES_REQUIRED**:
- $V_{\text{Var},x}^{\text{eq}} = \frac{C_x(\sigma_x)}{\kappa_x(\lambda)} \sim \frac{\sigma_x^2}{\lambda} + \frac{\sigma_v^2\tau^2}{\gamma\lambda}$

**PROPERTIES_ADDED**:
- **Iso-variance curve**:
  $$\lambda^*(\sigma_x) = \frac{\sigma_x^2 + \sigma_v^2\tau^2/\gamma}{V_{\text{Var},x}^{\text{target}}}$$

- **Limiting behaviors**:
  $$\lambda^*(\sigma_x) \approx \begin{cases}
  \frac{\sigma_v^2\tau^2}{\gamma V_{\text{Var},x}^{\text{target}}} & \text{if } \sigma_x \ll \sigma_v\tau/\sqrt{\gamma} \quad \text{(clean)} \\
  \frac{\sigma_x^2}{V_{\text{Var},x}^{\text{target}}} & \text{if } \sigma_x \gg \sigma_v\tau/\sqrt{\gamma} \quad \text{(noisy)}
  \end{cases}$$

- **Crossover point**: $\sigma_x^* = \sigma_v\tau/\sqrt{\gamma}$

- **Trade-off table** ($\sigma_v = 0.2$, $\tau = 0.01$, $\gamma = 0.3$, $V_{\text{Var},x}^{\text{target}} = 0.05$):

| $\sigma_x$ | Regime | $\lambda^*$ | Comments |
|-----------|--------|-------------|----------|
| 0.001 | Clean | 0.027 | Minimal cloning, low communication |
| 0.004 (crossover) | Transition | 0.031 | Jitter starts mattering |
| 0.01 | Noisy | 0.20 | High cloning to compensate |
| 0.02 | Noisy | 0.80 | Very frequent cloning |

**RELATIONS_ESTABLISHED**:
- **Clean cloning** ($\sigma_x$ small):
  - ✅ Low $\lambda$ → less communication overhead
  - ❌ Walkers cluster tightly → premature convergence risk
  - **Best for**: Exploitation, local refinement

- **Noisy cloning** ($\sigma_x$ large):
  - ✅ Maintains diversity automatically
  - ❌ High $\lambda$ → more communication
  - **Best for**: Exploration, multimodal landscapes

**INTERNAL_LEMMAS**:
- **Lem 1 (Equilibrium)**: $C_x/\kappa_x = V_{\text{Var},x}^{\text{target}} \Rightarrow$ solve for $\lambda$
- **Lem 2 (Crossover)**: $\sigma_x^2 \approx \sigma_v^2\tau^2/\gamma$ when both terms comparable

**DEPENDENCIES FORWARD**:
- **Adaptive tuning**: Anneal $\sigma_x$ from noisy (exploration) to clean (exploitation)
- **Communication budget**: Constrain $\lambda \leq \lambda_{\max} \Rightarrow$ minimum $\sigma_x$

---

### 23. prop-phase-space-pairing (Line 2878)

**TYPE**: Proposition
**OUTPUT_TYPE**: Relation (Coupling)

**INPUT_OBJECTS**:
- Fitness-variance correlation: $c_{\text{fit}}$
- Phase-space weight: $\lambda_{\text{alg}}$
- Pairing range: $\epsilon_c$

**INPUT_AXIOMS**:
- Companion selection metric (Ch 03): $d_{\text{alg}}(i,j)^2 = \|x_i-x_j\|^2 + \lambda_{\text{alg}}\|v_i-v_j\|^2$

**INPUT_PARAMETERS**:
- $\lambda_{\text{alg}} \geq 0$
- $\epsilon_c > 0$
- $\sigma_x$, $\sigma_v$ (typical separations)
- Target: $c_{\text{target}}$

**PROPERTIES_REQUIRED**:
- Pairing quality depends on signal-to-noise ratio

**PROPERTIES_ADDED**:
- **Correlation formula**:
  $$c_{\text{fit}}(\lambda_{\text{alg}}, \epsilon_c) \approx c_0 \cdot \left(1 + \frac{\lambda_{\text{alg}}\sigma_v^2}{\sigma_x^2}\right)^{-1/2} \cdot \left(1 + \frac{\epsilon_c^2}{\sigma_x^2}\right)^{-1}$$
  - $c_0 \approx 0.5-0.8$ (baseline, position-only tight pairing)

- **Term 1**: Velocity weighting degradation
  - $\left(1 + \frac{\lambda_{\text{alg}}\sigma_v^2}{\sigma_x^2}\right)^{-1/2}$
  - Velocity differences contaminate positional signal
  - For good performance: $\lambda_{\text{alg}}\sigma_v^2/\sigma_x^2 < 1$

- **Term 2**: Pairing range degradation
  - $\left(1 + \frac{\epsilon_c^2}{\sigma_x^2}\right)^{-1}$
  - Large $\epsilon_c$ allows mismatched pairs
  - For good performance: $\epsilon_c < \sigma_x$

- **Optimal curve** (fixed $c_{\text{target}}$):
  $$\epsilon_c^*(\lambda_{\text{alg}}) = \sigma_x\sqrt{\frac{c_0}{c_{\text{target}}}\left(1 + \frac{\lambda_{\text{alg}}\sigma_v^2}{\sigma_x^2}\right)^{1/2} - 1}$$

- **Numerical example** ($\sigma_x = 0.01$, $\sigma_v = 0.2$, $c_{\text{target}} = 0.6$):

| $\lambda_{\text{alg}}$ | Noise ratio | $\epsilon_c^*$ | Comments |
|------------------------|-------------|----------------|----------|
| 0 (position-only)      | 0           | 0.0024         | Tightest pairing |
| 0.001                  | 0.04        | 0.0025         | Minimal velocity effect |
| 0.01                   | 0.4         | 0.0034         | Moderate coupling |
| 0.1                    | 4.0         | 0.0092         | Strong velocity coupling |
| 1.0                    | 40.0        | 0.031          | Dominant velocity |

**RELATIONS_ESTABLISHED**:
- **Design rule**: $\lambda_{\text{alg}} \sim \sigma_x^2/\sigma_v^2$ (balance position and velocity contributions), then $\epsilon_c \sim \sigma_x$
- **Velocity noise reduces correlation**: $\sqrt{1 + \text{SNR}}$ penalty
- **Loose pairing reduces selectivity**: $1 + (\epsilon_c/\sigma_x)^2$ penalty
- **Deep interdependence**: Cannot tune $\lambda_{\text{alg}}$ and $\epsilon_c$ independently

**INTERNAL_LEMMAS**:
- **Lem 1 (Signal-to-noise)**: Velocity component adds independent noise → correlation reduced by $1/\sqrt{1 + \text{SNR}}$
- **Lem 2 (Selectivity dilution)**: Wide pairing range → fraction of correct pairs $\sim \sigma_x^2/(\sigma_x^2 + \epsilon_c^2)$

**DEPENDENCIES FORWARD**:
- **Geometric parameter tuning**: $\lambda_{\text{alg}}, \epsilon_c$ coupled
- **Sensitivity matrix** (`thm-explicit-rate-sensitivity`): Columns 4, 5 reflect this coupling

---

## PART V: ROBUSTNESS AND CLOSED-FORM SOLUTIONS (Objects 24-25)

### 24. thm-error-propagation (Line 2928)

**TYPE**: Theorem
**OUTPUT_TYPE**: Bound (Error Propagation)

**INPUT_OBJECTS**:
- Rate sensitivity matrix: $M_\kappa$ (from `thm-explicit-rate-sensitivity`)
- Singular values: $\sigma_1, \ldots, \sigma_4$ (from `thm-svd-rate-matrix`)
- Condition number: $\kappa(M_\kappa) = 5.4$ (from `prop-condition-number-rate`)

**INPUT_AXIOMS**:
- Matrix norms and perturbation theory

**INPUT_PARAMETERS**:
- Parameter errors: $\delta\mathbf{P}/\mathbf{P}_0 = \boldsymbol{\epsilon}$ with $\|\boldsymbol{\epsilon}\|_\infty = \epsilon_{\max}$

**PROPERTIES_REQUIRED**:
- Taylor expansion: $\delta\boldsymbol{\kappa} \approx M_\kappa \cdot (\mathbf{P}_0 \circ \delta\mathbf{P}/\mathbf{P}_0)$ ($\circ$ = element-wise product)

**PROPERTIES_ADDED**:
- **Error bound**:
  $$\frac{|\delta\kappa_{\text{total}}|}{\kappa_{\text{total}}} \leq \kappa(M_\kappa) \cdot \|M_\kappa\|_2 \cdot \epsilon_{\max}/\sqrt{12} + O(\epsilon_{\max}^2)$$

- **Using** $\kappa(M_\kappa) = 5.4$ and $\|M_\kappa\|_2 = \sigma_1 = 1.58$:
  $$\frac{|\delta\kappa_{\text{total}}|}{\kappa_{\text{total}}} \leq 5.4 \times 1.58 \times \epsilon_{\max}/\sqrt{12} \approx 2.5\epsilon_{\max}$$

- **Numerical examples**:
  - **10% parameter errors** ($\epsilon_{\max} = 0.1$): **≤25% rate slowdown**
  - **5% parameter errors** ($\epsilon_{\max} = 0.05$): **≤12% rate slowdown**
  - **20% parameter errors** ($\epsilon_{\max} = 0.2$): **≤50% rate slowdown**

- **Robustness table**:

| Parameter precision | $\epsilon_{\max}$ | Max rate degradation | Convergence slowdown |
|---------------------|-------------------|----------------------|---------------------|
| Tight (±5%)         | 0.05              | 12%                  | Negligible          |
| Moderate (±10%)     | 0.10              | 25%                  | Acceptable          |
| Loose (±20%)        | 0.20              | 50%                  | Significant         |
| Very loose (±50%)   | 0.50              | >100%                | System may fail     |

**RELATIONS_ESTABLISHED**:
- **Design guideline**: Aim for **±10% parameter precision** for robust performance
- **Error amplification factor**: $\sim 2.5$ (moderately well-conditioned)
- **Practical implication**: Parameter tuning doesn't need extreme precision

**INTERNAL_LEMMAS**:
- **Lem 1 (Taylor expansion)**: $\kappa(\mathbf{P} + \delta\mathbf{P}) \approx \kappa(\mathbf{P}_0) + M_\kappa \cdot \delta\mathbf{P} + O(\|\delta\mathbf{P}\|^2)$
- **Lem 2 (Norm bound)**: $\|\delta\boldsymbol{\kappa}\| \leq \|M_\kappa\|_2 \cdot \|\delta\mathbf{P}\|_2$
- **Lem 3 (Infinity to 2-norm)**: $\|\mathbf{x}\|_2 \leq \sqrt{n}\|\mathbf{x}\|_\infty$ for $\mathbf{x} \in \mathbb{R}^n$

**DEPENDENCIES FORWARD**:
- **Robustness design**: Tolerance specifications for parameter control
- **Adaptive tuning**: Error bars on estimated $\kappa_{\text{total}}$

---

### 25. thm-closed-form-optimum (Line 3092)

**TYPE**: Theorem
**OUTPUT_TYPE**: Property (Explicit Solution)

**INPUT_OBJECTS**:
- Landscape: $\lambda_{\min}$, $\lambda_{\max}$, $d$
- Target exploration: $V_{\text{target}}$

**INPUT_AXIOMS**:
- Balanced optimality (from `thm-balanced-optimality`)
- Component rate formulas (Props 8-11)

**INPUT_PARAMETERS**: (To be computed)

**PROPERTIES_REQUIRED**:
- Two-way balanced optimum: $\kappa_x = \kappa_v$ at optimal point

**PROPERTIES_ADDED**:
- **Closed-form optimal parameters** (unconstrained):

**Step 1: Friction from landscape**
$$\gamma^* = \lambda_{\min}$$
Maximizes $\kappa_W = c^2\gamma/(1+\gamma/\lambda_{\min})$

**Step 2: Cloning rate from balance**
$$\lambda^* = \frac{2\gamma^*}{c_{\text{fit}}} \approx \frac{2\lambda_{\min}}{0.65} \approx 3\lambda_{\min}$$
Achieves $\kappa_x = 2\gamma = \kappa_v$ (balanced two-way tie)

**Step 3: Timestep from stability**
$$\tau^* = \min\left(\frac{0.5}{\gamma^*}, \frac{1}{\sqrt{\lambda_{\max}}}, 0.01\right)$$
Ensures $\gamma\tau < 0.5$ and $\sqrt{\lambda_{\max}}\tau < 1$ (BAOAB stability)

**Step 4: Exploration noise from target**
$$\sigma_v^* = \sqrt{\gamma^* \cdot V_{\text{target}}}$$
Equilibrium variance $V_{\text{eq}} \sim \sigma_v^2/\gamma$

**Step 5: Position jitter from crossover**
$$\sigma_x^* = \frac{\sigma_v^*\tau^*}{\sqrt{\gamma^*}}$$
Crossover point where jitter = kinetic diffusion

**Step 6: Geometric parameters**
$$\lambda_{\text{alg}}^* = \frac{(\sigma_x^*)^2}{(\sigma_v^*)^2}, \quad \epsilon_c^* = \sigma_x^*$$
Balances position and velocity in pairing

**Step 7: Restitution coefficient**
$$\alpha_{\text{rest}}^* = \sqrt{2 - \frac{2\gamma_{\text{budget}}}{\gamma^*}}$$
($\gamma_{\text{budget}} \approx 1.5\gamma^*$ for moderate dissipation)

**Step 8: Boundary parameters**
$$d_{\text{safe}}^* = 3\sqrt{V_{\text{target}}}, \quad \kappa_{\text{wall}}^* = 10\lambda_{\min}$$
Three-sigma safety buffer, moderate boundary stiffness

- **Expected performance**:
  $$\kappa_{\text{total}}^* = \min(3\lambda_{\min}, 2\lambda_{\min}, c^2\lambda_{\min}/2) = c^2\lambda_{\min}/2 \approx 0.125\lambda_{\min}$$
  $$T_{\text{mix}} = \frac{5}{\kappa_{\text{total}}^*} = \frac{40}{\lambda_{\min}}$$

- **Numerical example** ($\lambda_{\min} = 0.1$, $\lambda_{\max} = 10$, $d = 10$, $V_{\text{target}} = 0.1$):
  $$\gamma^* = 0.1, \quad \lambda^* = 0.3, \quad \tau^* = 0.01, \quad \sigma_v^* = 0.1, \quad \sigma_x^* = 0.00316$$
  $$\lambda_{\text{alg}}^* = 0.001, \quad \epsilon_c^* = 0.00316, \quad \alpha_{\text{rest}}^* = 0, \quad d_{\text{safe}}^* = 0.95$$
  **Predicted**: $\kappa_{\text{total}}^* \approx 0.0125$, $T_{\text{mix}} \approx 400$ time units

**RELATIONS_ESTABLISHED**:
- **Complete automation**: Given landscape, compute all 12 parameters
- **No iteration needed**: Closed-form solution (no optimization loop)
- **Hypocoercivity bottleneck**: For ill-conditioned landscapes, $\kappa_W$ limits performance

**INTERNAL_LEMMAS**:
- **Lem 1 (Hypocoercivity optimum)**: $\kappa_W$ maximized when $\gamma = \lambda_{\min}$
- **Lem 2 (Balance condition)**: $\lambda c_{\text{fit}} = 2\gamma \Rightarrow \kappa_x = \kappa_v$
- **Lem 3 (Crossover)**: $\sigma_x = \sigma_v\tau/\sqrt{\gamma} \Rightarrow$ clean/noisy transition

**DEPENDENCIES FORWARD**:
- **Automated tuning**: Implement as default parameter selection
- **Benchmark baseline**: Compare constrained optimization vs. closed-form
- **Constrained problems**: Use as initial guess for iterative optimization

---

## SUMMARY STATISTICS

### By Output Type:
- **Properties**: 18 (drift conditions, rates, bounds, distributions)
- **Relations**: 5 (classifications, couplings, sensitivities)
- **Existence**: 1 (`thm-main-convergence`: QSD)
- **Bounds**: 4 (equilibrium, error, mixing time)

### By Chapter Section:
- **Section 3 (Synergistic Composition)**: 2 objects
- **Section 4 (Main Convergence)**: 5 objects (QSD existence, irreducibility, aperiodicity, properties, equilibrium bounds)
- **Section 5 (Explicit Parameters)**: 7 objects (component rates, total rate, mixing time, classification)
- **Section 6 (Spectral Optimization)**: 12 objects (sensitivity, SVD, optimization, coupling, robustness)

### Critical Forward Dependencies (Ch 08):
- **`thm-foster-lyapunov-main`**: Drift condition → propagation of chaos
- **`thm-main-convergence`**: Unique QSD → mean-field PDE convergence
- **`thm-total-rate-explicit`**: Explicit $\kappa_{\text{total}}$ → PDE contraction rate

### Key Mathematical Frameworks Used:
- **Meyn-Tweedie theory**: Geometric ergodicity from drift + irreducibility + aperiodicity
- **Hörmander hypoellipticity**: Irreducibility via Langevin controllability
- **Foster-Lyapunov drift**: Synergistic composition of complementary dissipation
- **Spectral analysis**: SVD of sensitivity matrices for optimization
- **Convex analysis**: Subgradients for non-smooth $\min()$ optimization

### N-Uniformity (Mean-Field Validity):
All 26 objects establish or use **N-uniform constants** (independent of swarm size $N$), critical for:
- Scalability to large swarms ($N \to \infty$)
- Mean-field limit (Ch 08)
- Thermodynamic limit rigor

---

## NOTES FOR DOWNSTREAM PROCESSING

1. **Object 2 (`thm-foster-lyapunov-main`)** is the keystone - used by 15+ downstream objects
2. **Object 5 (`thm-main-convergence`)** is the main convergence guarantee - most cited in Ch 08
3. **Object 13 (`thm-total-rate-explicit`)** provides all explicit formulas - essential for applications
4. **Object 25 (`thm-closed-form-optimum`)** enables automated parameter selection

5. **Cross-chapter dependencies**:
   - Ch 03 (cloning) → Objects 1, 2, 9, 11
   - Ch 05 (kinetic) → Objects 1, 2, 8, 10
   - Ch 08 (propagation chaos) ← Objects 2, 5, 13

6. **Parameter coupling network**:
   - Objects 21, 22, 23 show deep coupling (cannot tune parameters independently)
   - Sensitivity matrix (Objects 16, 17) quantifies all couplings
   - Null space (Object 17) provides 8-dimensional degeneracy for multi-objective optimization

---

**END OF REPORT**
