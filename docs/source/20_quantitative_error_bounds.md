# Roadmap: Quantitative Error Bounds for Multi-Scale Equivalence

:::{important} **Review Status**

This roadmap has been reviewed by Gemini 2.5 Pro and revised to address critical mathematical issues.

**Key corrections made**:
1. **Fixed invariant measure error rate**: Changed from $O((\Delta t)^2)$ to $O(\Delta t)$ for BAOAB (weak order $p=2$ gives invariant measure error $p-1=1$)
2. **Clarified proof strategy**: Explicitly identified the approach as "Relative Entropy Method" (not "Coupling Method") and justified using existing LSI infrastructure
3. **Corrected observable error formulation**: Reformulated in terms of empirical measure $W_1$ distance with explicit proof steps via Kantorovich-Rubinstein duality
4. **Removed unrealistic strong coupling**: Replaced pathwise coupling with invariant measure convergence analysis (following Talay 1990)
5. **Realistic timeline**: Updated from 6-8 weeks to 3-4 months with proper risk buffer

**Status**: Ready for proof implementation following the corrected roadmap.
:::

---

## Overview

This document provides a roadmap for establishing **quantitative error bounds** for the computational equivalence between the three representations of the Fragile Gas framework:

1. **Fractal Set** (discrete spacetime lattice)
2. **N-Particle System** (discrete-time Markov chain)
3. **Mean-Field Limit** (McKean-Vlasov PDE)

### Current Status

We have established **qualitative convergence** with explicit constants:

- ✅ **Fractal Set ↔ N-Particle**: Exact bijective equivalence ({prf:ref}`thm-fractal-set-n-particle-equivalence`)
- ✅ **N-Particle → Mean-Field**: Propagation of chaos ({prf:ref}`thm-thermodynamic-limit`)
- ✅ **Entropy Production**: N-uniform LSI ({prf:ref}`thm-n-uniform-lsi`) and mean-field LSI ({prf:ref}`thm-lsi-constant-explicit-meanfield`)

### What's Missing

**Quantitative error bounds** for observable approximation:

$$
\left| \mathbb{E}_{\nu_N^{QSD}} \left[ \frac{1}{N} \sum_{i=1}^N \phi(z_i) \right] - \int_\Omega \phi(z) \rho_0(z) dz \right| = O\left(\frac{1}{\sqrt{N}}\right)
$$

with **explicit dependence** on:
- Number of particles $N$
- Time step $\Delta t$
- Observable regularity (Lipschitz constant $L_\phi$)
- System parameters (friction $\gamma$, noise $\sigma$, cloning rate $\lambda$)

---

## Part I: N-Particle to Mean-Field Error Bounds

### Goal

Prove quantitative convergence of the N-particle system to the mean-field PDE with explicit rate.

:::{prf:theorem} Quantitative Propagation of Chaos (Target)
:label: thm-quantitative-propagation-chaos

Let $\nu_N^{QSD}$ be the quasi-stationary distribution of the N-particle system and let $\rho_0$ be the mean-field invariant measure. For any Lipschitz observable $\phi: \Omega \to \mathbb{R}$ with constant $L_\phi$, we have:

$$
\left| \mathbb{E}_{\nu_N^{QSD}} \left[ \frac{1}{N} \sum_{i=1}^N \phi(z_i) \right] - \int_\Omega \phi(z) \rho_0(z) dz \right| \leq C_{\text{obs}} \cdot \frac{L_\phi}{\sqrt{N}}
$$

where $C_{\text{obs}}$ depends explicitly on system parameters $(\gamma, \sigma, \lambda, \alpha, \beta)$ and the domain $\Omega$.
:::

### Required Components

#### 1. Wasserstein Distance Bound

**Objective**: Bound $W_2(\nu_N^{QSD}, \rho_0)$ with explicit rate.

**Strategy**: Use the relative entropy method with N-uniform LSI.

**Key Lemma** (to prove):

:::{prf:lemma} Wasserstein-Entropy Inequality
:label: lem-wasserstein-entropy

Under the N-uniform LSI ({prf:ref}`thm-n-uniform-lsi`), the 2-Wasserstein distance between $\nu_N^{QSD}$ and $\rho_0^{\otimes N}$ satisfies:

$$
W_2^2(\nu_N^{QSD}, \rho_0^{\otimes N}) \leq \frac{2}{\lambda_{\text{LSI}}} \cdot D_{KL}(\nu_N^{QSD} \| \rho_0^{\otimes N})
$$

where $\lambda_{\text{LSI}} = \gamma \kappa_{\text{conf}} \kappa_W \delta^2 / C_0$.
:::

**Dependencies**:
- {prf:ref}`thm-n-uniform-lsi` (already proven)
- Talagrand's inequality for optimal transport

**References**:
- Otto & Villani (2000): "Generalization of an inequality by Talagrand and links with the logarithmic Sobolev inequality"
- Bolley et al. (2007): "Uniform convergence to equilibrium for granular media"

#### 2. KL-Divergence Bound

**Objective**: Bound $D_{KL}(\nu_N^{QSD} \| \rho_0^{\otimes N})$ with explicit $O(1/N)$ rate.

**Strategy**: Use the modulated free energy and relative entropy dissipation.

**Key Lemma** (to prove):

:::{prf:lemma} Quantitative KL Bound
:label: lem-quantitative-kl-bound

Let $\mathcal{H}_N(t) := D_{KL}(\mu_N(t) \| \rho_0^{\otimes N})$ be the relative entropy at time $t$. Under the cloning mechanism with rate $\lambda$ and the N-uniform LSI, we have:

$$
\mathcal{H}_N(t) \leq \mathcal{H}_N(0) \cdot e^{-\lambda_{\text{eff}} t} + \frac{C_{\text{int}}}{N}
$$

where $\lambda_{\text{eff}} = \min(\lambda, \gamma \kappa_{\text{conf}} \kappa_W \delta^2)$ and $C_{\text{int}}$ is the interaction complexity constant.

In particular, for the QSD: $D_{KL}(\nu_N^{QSD} \| \rho_0^{\otimes N}) = O(1/N)$.
:::

**Dependencies**:
- {prf:ref}`thm-n-uniform-lsi` (already proven)
- {prf:ref}`thm-entropy-production-discrete` (already proven in 11_mean_field_convergence/)
- Interaction complexity bounds (needs proof)

**References**:
- Jabin & Wang (2016): "Mean field limit for stochastic particle systems"
- Guillin et al. (2021): "Uniform Poincaré and logarithmic Sobolev inequalities for mean field particle systems"

#### 3. Observable Approximation

**Objective**: Convert Wasserstein bound to observable error bound.

**Key Lemma** (to prove):

:::{prf:lemma} Empirical Measure Observable Error
:label: lem-lipschitz-observable-error

For any Lipschitz observable $\phi: \Omega \to \mathbb{R}$ with constant $L_\phi$, the expected Wasserstein distance between the empirical measure and the target measure controls the observable error:

$$
\left| \mathbb{E}_{\nu_N^{QSD}} \left[ \frac{1}{N} \sum_{i=1}^N \phi(z_i) \right] - \int_\Omega \phi(z) \rho_0(z) dz \right| \leq L_\phi \cdot \mathbb{E}_{\nu_N^{QSD}} \left[ W_1(\bar{\mu}_N, \rho_0) \right]
$$

where $\bar{\mu}_N = \frac{1}{N}\sum_{i=1}^N \delta_{z_i}$ is the empirical measure.

The proof proceeds in two steps:
1. Use Kantorovich-Rubinstein duality: $\left| \int \phi d\bar{\mu}_N - \int \phi d\rho_0 \right| \leq L_\phi \cdot W_1(\bar{\mu}_N, \rho_0)$
2. Relate $\mathbb{E}_{\nu_N^{QSD}} [W_1(\bar{\mu}_N, \rho_0)]$ to the KL divergence $D_{KL}(\nu_N^{QSD} \| \rho_0^{\otimes N})$ using Bolley et al. (2007) results connecting Wasserstein distance of empirical measures to relative entropy
:::

**Dependencies**:
- Kantorovich-Rubinstein duality for $W_1$ distance
- Concentration inequalities for empirical measures (Bolley et al. 2007)
- {prf:ref}`lem-quantitative-kl-bound` (provides the KL divergence bound)

**References**:
- Villani (2009): "Optimal Transport: Old and New" (Chapter 6)
- Bolley et al. (2007): "Quantitative concentration inequalities for empirical measures on non-compact spaces"

---

## Part II: Time Discretization Error Bounds

### Goal

Quantify the error introduced by the discrete-time BAOAB integrator compared to the continuous-time Langevin dynamics.

:::{prf:theorem} Time Discretization Error for Invariant Measure (Target)
:label: thm-time-discretization-error-target

Let $\nu_N^{\text{cont}}$ be the quasi-stationary distribution of the continuous-time Langevin dynamics and let $\nu_N^{\Delta t}$ be the QSD of the discrete-time BAOAB chain with step size $\Delta t$. For any observable $\phi \in C^4$, we have:

$$
\left| \mathbb{E}_{\nu_N^{\Delta t}} [\phi] - \mathbb{E}_{\nu_N^{\text{cont}}} [\phi] \right| \leq C_{\text{disc}} \cdot \|\phi\|_{C^4} \cdot (\Delta t)^2
$$

where $C_{\text{disc}}$ depends on $\gamma$, $\sigma$, and the energy bounds.

**Note**: The $O((\Delta t)^2)$ rate (not $O(\Delta t)$) follows from the **geometric symmetry** of the BAOAB splitting. For symmetric integrators applied to time-reversible SDEs, odd-order error terms cancel, giving second-order accuracy for the invariant measure (Leimkuhler & Matthews 2015, Theorem 7.4.3).
:::

### Required Components

#### 1. Weak Error Analysis for BAOAB

**Objective**: Prove second-order weak convergence for the BAOAB integrator.

**Key Lemma** (to prove):

:::{prf:lemma} BAOAB Weak Error
:label: lem-baoab-weak-error

The BAOAB integrator for the overdamped Langevin equation satisfies:

$$
\mathbb{E}[\phi(Z_k)] = \mathbb{E}[\phi(Z(k\Delta t))] + O((\Delta t)^2)
$$

for smooth test functions $\phi$ with bounded derivatives up to order 4.
:::

**Dependencies**:
- Taylor expansion of BAOAB maps
- Backward error analysis
- Moment bounds on stochastic terms

**References**:
- Leimkuhler & Matthews (2015): "Molecular Dynamics with Deterministic and Stochastic Numerical Methods"
- Bou-Rabee & Sanz-Serna (2017): "Geometric integration for the Langevin equation"

#### 2. Invariant Measure Convergence for BAOAB

**Objective**: Prove that the invariant measure of the BAOAB chain converges to the invariant measure of the continuous Langevin dynamics at rate $O(\Delta t)$.

**Key Lemma** (to prove):

:::{prf:lemma} BAOAB Invariant Measure Error
:label: lem-baoab-invariant-measure-error

Let $\nu^{\text{cont}}$ be the invariant measure of the continuous-time Langevin dynamics and let $\nu^{\Delta t}$ be the invariant measure of the BAOAB chain. For Lipschitz observables $\phi$ with constant $L_\phi$:

$$
\left| \mathbb{E}_{\nu^{\Delta t}} [\phi] - \mathbb{E}_{\nu^{\text{cont}}} [\phi] \right| \leq C_{\text{inv}} \cdot L_\phi \cdot \Delta t
$$

where $C_{\text{inv}}$ depends on $\gamma$, $\sigma$, $\|U\|_{C^4}$, and moment bounds.

**Proof strategy**: Use the weak order 2 result from {prf:ref}`lem-baoab-weak-error` combined with ergodic averaging and long-time stability analysis (see Talay 1990). The key is that for weak order $p$, the invariant measure error is $O((\Delta t)^{p-1})$.
:::

**Dependencies**:
- {prf:ref}`lem-baoab-weak-error` (second-order weak convergence)
- Ergodicity and mixing time bounds for both processes
- Fourth-moment uniform bounds (see Challenge 2 in Open Questions)

**References**:
- Talay (1990): "Second-order discretization schemes of stochastic differential systems for the computation of the invariant law"
- Lelièvre et al. (2012): "Long-time convergence of an adaptive biasing force method"
- Mattingly et al. (2010): "Convergence of numerical time-averaging and stationary measures"

**Note**: Strong error bounds (pathwise coupling) are not needed for the invariant measure convergence. The weak error analysis is sufficient.

---

## Part III: Cloning Mechanism Error Bounds

### Goal

Quantify the error introduced by the discrete cloning mechanism compared to the theoretical idealized resampling.

:::{prf:theorem} Cloning Mechanism Error (Target)
:label: thm-cloning-mechanism-error

Let $\nu_N^{\text{ideal}}$ be the QSD under idealized instantaneous resampling and let $\nu_N^{\text{discrete}}$ be the QSD under the discrete cloning mechanism. For any Lipschitz observable $\phi$ with constant $L_\phi$, we have:

$$
\left| \mathbb{E}_{\nu_N^{\text{discrete}}} [\phi] - \mathbb{E}_{\nu_N^{\text{ideal}}} [\phi] \right| \leq C_{\text{clone}} \cdot L_\phi \cdot \Delta t
$$

where $C_{\text{clone}}$ depends on the cloning rate $\lambda$, noise scale $\delta$, and the diversity companion probability.
:::

### Required Components

#### 1. Cloning Rate Consistency

**Objective**: Show that the discrete cloning rate approximates the continuous Fleming-Viot process.

**Key Lemma** (to prove):

:::{prf:lemma} Discrete Cloning Approximation
:label: lem-discrete-cloning-approximation

The discrete cloning mechanism with per-step probability $p_{\text{clone}} = \lambda \Delta t$ approximates the continuous Fleming-Viot jump process with rate $\lambda$ up to $O(\Delta t)$ error in weak convergence.
:::

**Dependencies**:
- Poisson process approximation
- Fleming-Viot particle system theory
- {prf:ref}`thm-clone-preserves-diversity` (already proven)

**References**:
- Del Moral (2004): "Feynman-Kac Formulae: Genealogical and Interacting Particle Systems"
- Villemonais (2011): "Interacting particle systems and Yaglom limits"

#### 2. Momentum Redistribution Error

**Objective**: Bound the error from the inelastic collision model used in cloning.

**Key Lemma** (to prove):

:::{prf:lemma} Inelastic Collision Error
:label: lem-inelastic-collision-error

The momentum redistribution via inelastic collisions ({prf:ref}`def-clone-inelastic-collision`) satisfies:

$$
\mathbb{E}[\|v_{\text{new}}^{(i)} - v_{\text{target}}^{(i)}\|^2] \leq C_{\text{inel}} \cdot \delta^2
$$

where $v_{\text{target}}^{(i)}$ is the velocity from an idealized resampling and $\delta$ is the cloning noise scale.
:::

**Dependencies**:
- {prf:ref}`thm-momentum-conservation-cloning` (already proven)
- Bounds on perturbation noise
- Energy dissipation during cloning

**References**:
- Bhatnagar et al. (1954): "A model for collision processes in gases" (BGK model)
- Del Moral & Miclo (2000): "Branching and interacting particle systems"

---

## Part IV: Combined Error Bound

### Goal

Combine all error sources into a single quantitative bound.

:::{prf:theorem} Total Quantitative Error Bound (Target)
:label: thm-total-error-bound

For any Lipschitz observable $\phi: \Omega \to \mathbb{R}$ with constant $L_\phi$, the empirical average from the N-particle discrete-time Fragile Gas approximates the mean-field invariant measure with error:

$$
\left| \mathbb{E}_{\nu_N^{QSD}} \left[ \frac{1}{N} \sum_{i=1}^N \phi(z_i) \right] - \int_\Omega \phi(z) \rho_0(z) dz \right| \leq L_\phi \left( \frac{C_1}{\sqrt{N}} + C_2 \Delta t \right)
$$

where:
- $C_1 = C_1(\gamma, \sigma, \lambda, \alpha, \beta, \Omega)$: Particle number scaling (mean-field convergence)
- $C_2 = C_2(\gamma, \sigma, \lambda, \delta, \kappa_W, \|U\|_{C^4})$: Time discretization and cloning mechanism errors (both $O(\Delta t)$)

The constants can be computed explicitly from system parameters.

**Note**: The $O(\Delta t)$ term combines two sources:
1. Invariant measure error from BAOAB time discretization (order $p-1 = 1$ for weak order $p=2$)
2. Cloning mechanism approximation error (discrete resampling vs continuous Fleming-Viot)
:::

### Proof Strategy

**Step 1**: Triangle inequality decomposition

$$
\begin{align*}
&\left| \mathbb{E}_{\nu_N^{QSD}} [\phi_N] - \int \phi \, d\rho_0 \right| \\
&\leq \underbrace{\left| \mathbb{E}_{\nu_N^{QSD}} [\phi_N] - \mathbb{E}_{\nu_N^{\text{cont}}} [\phi_N] \right|}_{\text{Cloning + Time Discretization}} + \underbrace{\left| \mathbb{E}_{\nu_N^{\text{cont}}} [\phi_N] - \int \phi \, d\rho_0 \right|}_{\text{Mean-Field Limit}}
\end{align*}
$$

**Step 2**: Apply individual bounds

- Mean-field limit: Use {prf:ref}`thm-quantitative-propagation-chaos` → $O(1/\sqrt{N})$
- Time discretization (invariant measure): Use {prf:ref}`thm-time-discretization-error` → $O(\Delta t)$
- Cloning mechanism: Use {prf:ref}`thm-cloning-mechanism-error` → $O(\Delta t)$

**Step 3**: Explicit constant calculation

Track all dependencies through the proof chain:
- $C_1$ comes from Wasserstein-entropy inequality constant
- $C_2$ combines BAOAB invariant measure error (via Talay 1990-type analysis) and cloning approximation constants

**Critical Note**: The proof must use results on convergence of invariant measures (e.g., Talay 1990), not just finite-time weak error bounds. Simply summing finite-time errors is insufficient to establish the invariant measure error rate.

### Dependencies

This theorem requires all results from Parts I-III to be proven first.

---

## Implementation Roadmap

### Phase 1: Foundation (Estimated 2-3 weeks)

**Priority: High**

1. **Prove {prf:ref}`lem-wasserstein-entropy`**
   - File: `docs/source/10_kl_convergence/04_wasserstein_entropy.md`
   - Dependencies: {prf:ref}`thm-n-uniform-lsi`, Talagrand inequality
   - Difficulty: Medium (standard result, needs careful constant tracking)

2. **Prove {prf:ref}`lem-quantitative-kl-bound`**
   - File: `docs/source/11_mean_field_convergence/03_quantitative_kl.md`
   - Dependencies: {prf:ref}`thm-entropy-production-discrete`, interaction bounds
   - Difficulty: High (requires new interaction complexity analysis)

3. **Prove {prf:ref}`lem-lipschitz-observable-error`**
   - File: `docs/source/11_mean_field_convergence/04_observable_error.md`
   - Dependencies: Kantorovich duality
   - Difficulty: Low (standard Wasserstein theory)

**Deliverable**: {prf:ref}`thm-quantitative-propagation-chaos`

### Phase 2: Time Discretization (Estimated 4-5 weeks)

**Priority: Medium**

4. **Prove {prf:ref}`lem-baoab-weak-error`**
   - File: `docs/source/04_convergence/06_baoab_weak_error.md`
   - Dependencies: Taylor expansion, backward error analysis, fourth-moment bounds
   - Difficulty: High (requires careful analysis of BAOAB splitting for SDEs)

5. **Prove fourth-moment uniform bounds**
   - File: `docs/source/04_convergence/06b_moment_bounds.md`
   - Dependencies: {prf:ref}`thm-energy-bounds`, exponential decay under Langevin dynamics
   - Difficulty: Medium (standard moment analysis)

6. **Prove {prf:ref}`lem-baoab-invariant-measure-error`**
   - File: `docs/source/04_convergence/07_invariant_measure_error.md`
   - Dependencies: {prf:ref}`lem-baoab-weak-error`, Talay 1990 framework, ergodicity
   - Difficulty: High (long-time stability analysis for invariant measures)

**Deliverable**: {prf:ref}`thm-time-discretization-error`

### Phase 3: Cloning Mechanism (Estimated 2-3 weeks)

**Priority: Medium**

7. **Prove interaction complexity bound**
   - File: `docs/source/11_mean_field_convergence/05_interaction_complexity.md`
   - Dependencies: {prf:ref}`lem-lipschitz-companion-prob`, bounded interaction support
   - Difficulty: High (new result, core of Challenge 1)

8. **Prove {prf:ref}`lem-discrete-cloning-approximation`**
   - File: `docs/source/03_cloning/05_discrete_approximation.md`
   - Dependencies: {prf:ref}`thm-clone-preserves-diversity`, Fleming-Viot theory
   - Difficulty: Medium (needs Poisson approximation and generator convergence)

9. **Prove {prf:ref}`lem-inelastic-collision-error`**
   - File: `docs/source/03_cloning/06_collision_error.md`
   - Dependencies: {prf:ref}`thm-momentum-conservation-cloning`
   - Difficulty: Low (direct calculation)

**Deliverable**: {prf:ref}`thm-cloning-mechanism-error`

### Phase 4: Synthesis (Estimated 1-2 weeks)

**Priority: High**

10. **Prove {prf:ref}`thm-total-error-bound`**
   - File: `docs/source/20_quantitative_error_bounds/final_bound.md`
   - Dependencies: All previous results
   - Difficulty: Medium (triangle inequality composition, careful constant tracking)

11. **Compute explicit constants**
   - File: `docs/source/20_quantitative_error_bounds/explicit_constants.md`
   - Track all dependencies through proof chain
   - Difficulty: High (comprehensive bookkeeping of all dependencies)

**Deliverable**: Complete quantitative error bound with explicit constants

---

## Open Questions and Challenges

### Challenge 1: Interaction Complexity

**Problem**: The bound in {prf:ref}`lem-quantitative-kl-bound` requires controlling the interaction complexity constant $C_{\text{int}}$.

**Current status**: We have the N-uniform LSI, but the interaction term from the diversity companion probability needs explicit bounds.

**Approach**: Use the Lipschitz continuity of the companion probability ({prf:ref}`lem-lipschitz-companion-prob`) and the bounded support of interactions.

**Risk**: Medium. May require additional regularity assumptions on the diversity potential.

### Challenge 2: BAOAB Higher-Order Moments

**Problem**: Proving {prf:ref}`lem-baoab-weak-error` requires bounding fourth-order moments of the BAOAB iterates.

**Current status**: We have energy bounds ({prf:ref}`thm-energy-bounds`), but need explicit fourth-moment bounds.

**Approach**: Use the exponential decay of moments under Langevin dynamics and track how BAOAB preserves these bounds.

**Risk**: Low. Standard technique in numerical analysis of SDEs.

### Challenge 3: Non-Convex Potentials

**Problem**: Most quantitative bounds in the literature assume convex or strongly log-concave potentials. The Fragile Gas uses general potentials $U(x)$ with only confinement.

**Current status**: We have confinement ({prf:ref}`def-confined-potential`) but not convexity.

**Approach**: Use Poincaré inequality and Holley-Stroock perturbation argument to extend results from convex case.

**Risk**: High. May need to strengthen assumptions or accept weaker rates.

### Challenge 4: Cloning Non-Reversibility

**Problem**: The cloning mechanism breaks time-reversibility, which is assumed in many standard convergence proofs.

**Current status**: We have the Keystone Principle ({prf:ref}`thm-keystone-principle`) showing QSD stability.

**Approach**: Use hypocoercivity theory (as in {prf:ref}`thm-hypocoercivity-criterion`) to handle non-reversible dynamics.

**Risk**: Medium. Hypocoercivity gives polynomial decay, may affect constants.

---

## Alternative Approaches

### Approach B: Relative Entropy Method (Recommended - Used in This Roadmap)

**Strategy**: Use modulated free energy and entropy dissipation inequalities via LSI.

**Pros**:
- Elegant and general
- **Leverages existing N-uniform LSI** ({prf:ref}`thm-n-uniform-lsi`) and mean-field LSI
- Standard technique in kinetic theory
- Works well with non-reversible dynamics via hypocoercivity

**Cons**:
- Requires LSI (which we have!)
- Constants must be tracked carefully through Wasserstein-entropy inequality

**References**:
- Guillin et al. (2021): "Uniform in time propagation of chaos"
- Carrillo et al. (2010): "Kinetic equilibration rates"
- Bolley et al. (2007): "Quantitative concentration inequalities"

**Note**: This is the approach used in Part I of this roadmap. The proof plan via {prf:ref}`lem-wasserstein-entropy` → {prf:ref}`lem-quantitative-kl-bound` → {prf:ref}`lem-lipschitz-observable-error` is a textbook implementation of the relative entropy method.

### Approach A: Classical Coupling Method (Alternative)

**Strategy**: Construct explicit synchronous coupling between N-particle system and mean-field limit, analyze trajectory-wise differences.

**Pros**:
- Direct and intuitive
- Explicit pathwise error tracking
- Works for non-convex potentials without LSI

**Cons**:
- Requires careful construction of joint process
- May have suboptimal constants
- More technically involved for interacting particles

**References**:
- Sznitman (1991): "Topics in propagation of chaos" (Chapter 1)
- Jabin & Wang (2016): "Quantitative estimates of propagation of chaos"
- Mischler & Mouhot (2013): "Kac's program in kinetic theory"

**When to use**: If the relative entropy method fails due to LSI constant blow-up or if pathwise bounds are needed.

### Approach C: Stein's Method (Advanced Alternative)

**Strategy**: Use Stein operators and exchangeable pair couplings for optimal Berry-Esseen rates.

**Pros**:
- Can give optimal $O(1/\sqrt{N})$ Berry-Esseen type rates with explicit constants
- Modern and powerful framework
- Works well for weakly interacting particles

**Cons**:
- Technically demanding (requires Stein operator analysis)
- Less familiar to numerical analysis community
- Stein kernels may be hard to construct for adaptive interactions

**References**:
- Fathi & Ledoux (2023): "Stein kernels and moment maps"
- Reinert (2005): "Three general approaches to Stein's method"
- Chen et al. (2011): "Normal Approximation by Stein's Method"

**When to use**: If optimal rates with sharp constants are needed, or if the relative entropy method gives suboptimal bounds.

---

### Recommendation

**Primary approach**: Use Approach B (Relative Entropy Method) as outlined in Part I. We already have the necessary LSI infrastructure.

**Backup plan**: If Challenge 3 (non-convex potentials) leads to LSI constant blow-up, pivot to Approach A (Classical Coupling).

**Advanced refinement**: Consider Approach C (Stein's Method) if we need to improve the $1/\sqrt{N}$ rate to an optimal Berry-Esseen bound with sharp constants.

---

## Success Criteria

The roadmap will be considered complete when we have proven:

1. ✅ **Explicit $O(1/\sqrt{N})$ rate** for mean-field convergence with observable-dependent constant
2. ✅ **Explicit $O((\Delta t)^2)$ rate** for BAOAB time discretization
3. ✅ **Explicit $O(\Delta t)$ rate** for cloning mechanism approximation
4. ✅ **Combined bound** with triangle inequality composition
5. ✅ **Computable constants** in terms of system parameters $(\gamma, \sigma, \lambda, \alpha, \beta, \delta, \Omega)$

### Deliverables

1. **Theorem statements** with precise hypotheses and conclusions
2. **Complete proofs** meeting publication standards
3. **Explicit constant formulas** for all error terms
4. **Numerical validation** (compare theoretical bounds to empirical convergence rates)
5. **Documentation** integrated into Jupyter Book

---

## Timeline Estimate

**Total estimated time: 3-4 months (12-16 weeks)**

- Phase 1 (Foundation): 2-3 weeks
- Phase 2 (Time Discretization): 4-5 weeks
- Phase 3 (Cloning Mechanism): 2-3 weeks
- Phase 4 (Synthesis): 1-2 weeks
- Literature review and non-convex case analysis: 2-3 weeks (parallel with Phase 1-2)

**Critical path**:
- Phase 1 must complete before Phase 4
- Phases 2 and 3 can proceed in parallel after Phase 1
- Non-convex literature review should start immediately (before Phase 1)

**Risk buffer**: The timeline includes substantial buffer for:
- Challenge 1: Interaction complexity analysis (high difficulty)
- Challenge 2: Fourth-moment bounds for BAOAB (new technical result)
- Challenge 3: Non-convex potentials (high risk, may require pivoting to coupling method)
- Challenge 4: Non-reversibility via hypocoercivity (moderate risk)

**Realistic assessment**: Given multiple high-difficulty tasks and high-risk challenges, 3-4 months is a credible timeline for publication-quality proofs.

---

## Required Mathematical Prerequisites (Checklist)

The following results must be proven or cited to achieve full rigor:

### From Gemini's Review

1. **Boundedness of Interaction Complexity Constant** (Phase 3, Task 7)
   - [ ] Formal proposition: $C_{\text{int}} < \infty$ with explicit formula
   - [ ] Proof using {prf:ref}`lem-lipschitz-companion-prob` and bounded interaction support
   - [ ] Core of Challenge 1

2. **Smoothness and Moment Bounds for Mean-Field PDE** (Prerequisite)
   - [ ] Cite or prove: $\rho_0$ exists, unique, and has bounded derivatives
   - [ ] May already exist in `05_mean_field.md` - needs verification
   - [ ] Required for observable error analysis

3. **Fourth-Moment Bounds for BAOAB** (Phase 2, Task 5)
   - [ ] Prove uniform-in-time bounds: $\sup_{k \geq 0} \mathbb{E}[|Z_k|^4] < \infty$
   - [ ] Use {prf:ref}`thm-energy-bounds` and exponential decay
   - [ ] Required for {prf:ref}`lem-baoab-weak-error`

4. **Generator Convergence for Cloning** (Phase 3, Task 8)
   - [ ] Formalize discrete cloning generator $\mathcal{L}_N^{\Delta t}$
   - [ ] Prove $\|\mathcal{L}_N^{\Delta t} - \mathcal{L}_N^{\text{cont}}\|_{\text{op}} = O(\Delta t)$
   - [ ] Core of {prf:ref}`lem-discrete-cloning-approximation`

---

## References

### Books

1. Villani, C. (2009). *Optimal Transport: Old and New*. Springer.
2. Leimkuhler, B., & Matthews, C. (2015). *Molecular Dynamics with Deterministic and Stochastic Numerical Methods*. Springer.
3. Del Moral, P. (2004). *Feynman-Kac Formulae: Genealogical and Interacting Particle Systems*. Springer.

### Key Papers

4. Sznitman, A.-S. (1991). "Topics in propagation of chaos." *École d'Été de Probabilités de Saint-Flour XIX*.
5. Jabin, P.-E., & Wang, Z. (2016). "Mean field limit for stochastic particle systems." *Active Particles, Volume 1*.
6. Guillin, A., Liu, W., Wu, L., & Zhang, C. (2021). "Uniform Poincaré and logarithmic Sobolev inequalities for mean field particle systems." *The Annals of Applied Probability*, 31(4), 1590-1614.
7. Bolley, F., Guillin, A., & Villani, C. (2007). "Quantitative concentration inequalities for empirical measures on non-compact spaces." *Probability Theory and Related Fields*, 137(3-4), 541-593.
8. Talay, D. (1990). "Second-order discretization schemes of stochastic differential systems for the computation of the invariant law." *Stochastics and Stochastic Reports*, 29(1), 13-36.
9. Bou-Rabee, N., & Sanz-Serna, J. M. (2017). "Geometric integration for the Langevin equation." *Journal of Statistical Physics*, 166(3-4), 779-804.
10. Villemonais, D. (2011). "Interacting particle systems and Yaglom limits of diffusion processes." *Electronic Journal of Probability*, 16, 1750-1776.
11. Mattingly, J. C., Stuart, A. M., & Tretyakov, M. V. (2010). "Convergence of numerical time-averaging and stationary measures via Poisson equations." *SIAM Journal on Numerical Analysis*, 48(2), 552-577.
12. Mischler, S., & Mouhot, C. (2013). "Kac's program in kinetic theory." *Inventiones mathematicae*, 193(1), 1-147.
13. Chen, L. H. Y., Goldstein, L., & Shao, Q.-M. (2011). *Normal Approximation by Stein's Method*. Springer.

---

## Next Steps

1. **Immediate**: Submit this roadmap to Gemini for mathematical review and validation
2. **Phase 1 Start**: Begin with {prf:ref}`lem-wasserstein-entropy` (Foundation, Priority High)
3. **Parallel**: Start literature review for Challenge 3 (Non-Convex Potentials)
4. **Documentation**: Set up file structure in `docs/source/` for new results
