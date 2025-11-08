# Discrete-Time Finite-N Hypocoercivity Strategy for KL-Convergence

## Executive Summary

This document provides the strategy for adapting the mean-field continuous-time hypocoercivity proof from `16_convergence_mean_field.md` to the **discrete-time finite-N Euclidean Gas** in `09_kl_convergence.md`.

**Key Achievement**: This approach **avoids the reference measure mismatch** that doomed previous proofs by analyzing the **full composite operator** $\Psi_{\text{total}} = \Psi_{\text{kin}}(\tau) \circ \Psi_{\text{clone}}$ as a **single** discrete-time Markov operator relative to $\pi_{\text{QSD}}$.

## Problem Statement

**Target Theorem**: {prf:ref}`thm-main-kl-final` in `09_kl_convergence.md:1902`

Prove that the discrete-time N-particle Euclidean Gas satisfies:

$$
D_{\text{KL}}(\mu_{t+1} \| \pi_{\text{QSD}}) \le (1 - \alpha \tau) D_{\text{KL}}(\mu_t \| \pi_{\text{QSD}}) + O(\tau^2)
$$

with explicit rate $\alpha > 0$ independent of N.

## Why Previous Proofs Failed

**Critical Error**: Reference measure mismatch
- Stage 1-3 proof: Analyzed $\Psi_{\text{kin}}$ relative to $\pi_{\text{kin}}$ (Maxwell-Boltzmann), then $\Psi_{\text{clone}}$ relative to $\pi_{\text{QSD}}$
- Part III alternative: Same fundamental error
- **Both compositions are mathematically invalid**: cannot compose operators with different reference measures

**Root Cause**: Trying to prove LSI for each operator separately, then compose.

## The Correct Approach (Gemini + Codex Consensus)

**Approach C (Mean-Field Generator Method)**: Analyze the **full generator** $\mathcal{L} = \mathcal{L}_{\text{kin}} + \mathcal{L}_{\text{jump}}$ as a **single operator** relative to $\rho_\infty$ (the continuum QSD).

**16_convergence_mean_field.md implements this correctly for continuous-time mean-field limit.**

**Our task**: Adapt this to **discrete-time finite-N**.

## Conceptual Bridge: Continuous → Discrete

### Mean-Field Continuous-Time (16_convergence_mean_field.md)

**Generator**: $\mathcal{L}[\rho] = \mathcal{L}_{\text{kin}}[\rho] + \mathcal{L}_{\text{jump}}[\rho]$

**Entropy production**:
$$
\frac{d}{dt} D_{\text{KL}}(\rho_t \| \rho_\infty) = -\frac{\sigma^2}{2} \mathcal{I}_v(\rho_t \| \rho_\infty) + R_{\text{coupling}}[\rho_t] + \mathcal{I}_{\text{jump}}[\rho_t]
$$

**Key structure**:
- **Dissipation**: Velocity Fisher information $\mathcal{I}_v$ from kinetic operator
- **Expansion**: Coupling terms $R_{\text{coupling}}$ and jump terms $\mathcal{I}_{\text{jump}}$
- **Balance**: Kinetic dominance condition $\delta = \lambda_{\text{LSI}} \sigma^2 - (\text{expansions}) > 0$

### Discrete-Time Finite-N (Our Target)

**Composite operator**: $\Psi_{\text{total}} = \Psi_{\text{kin}}(\tau) \circ \Psi_{\text{clone}}$

**Entropy change** (one time step):
$$
D_{\text{KL}}(\mu_{t+1} \| \pi_{\text{QSD}}) - D_{\text{KL}}(\mu_t \| \pi_{\text{QSD}}) = ?
$$

**Required structure** (discrete analog):
- **Dissipation**: From BAOAB integrator (hypocoercive friction + diffusion)
- **Expansion**: From cloning operator (killing + revival mechanism)
- **Balance**: Discrete kinetic dominance condition

## Required Tools for Discrete-Time Analysis

### Tool 1: Discrete Relative Entropy Decomposition

For discrete-time Markov operators $\Psi: \mathcal{P}(\mathcal{Y}) \to \mathcal{P}(\mathcal{Y})$ with invariant measure $\pi$:

$$
D_{\text{KL}}(\Psi_*\mu \| \pi) = \int \log \frac{d\Psi_*\mu}{d\pi}(y) \, d\Psi_*\mu(y)
$$

**Change of variables** (pull back through $\Psi$):
$$
D_{\text{KL}}(\Psi_*\mu \| \pi) = \int \log \frac{d\Psi_*\mu}{d\pi}(\Psi(S)) \, d\mu(S)
$$

**Key formula** (discrete Chapman-Kolmogorov for entropy):
$$
D_{\text{KL}}(\mu_{t+1} \| \pi) = \int D_{\text{KL}}(\Psi(S, \cdot) \| \pi) \, d\mu_t(S) + D_{\text{KL}}(\mu_t \| \Psi^*\pi)
$$

where $\Psi(S, \cdot)$ is the transition kernel from state $S$.

**This is the discrete analog of the entropy production equation.**

### Tool 2: BAOAB Integrator Entropy Analysis

The BAOAB integrator for the kinetic operator has explicit structure:
- **B**: Momentum kick by potential gradient
- **A**: Ornstein-Uhlenbeck process (friction + noise)
- **O**: Position update (free flight)
- **A**: Second friction + noise step
- **B**: Second momentum kick

**Key property**: The **A** (Ornstein-Uhlenbeck) steps are **exactly solvable** and dissipate entropy relative to the Gaussian velocity marginal.

**Hypocoercivity mechanism**:
1. **O** step couples position and velocity
2. **A** steps dissipate velocity entropy
3. Coupling + dissipation → Full phase space entropy dissipation

**Discrete Fisher information**: The **A** steps produce measurable Fisher information dissipation:
$$
D_{\text{KL}}(\Psi_{\text{A}}^*\mu \| \pi_{\text{Gaussian}}) \le (1 - c\gamma\tau) D_{\text{KL}}(\mu \| \pi_{\text{Gaussian}}) + O(\tau^2)
$$

### Tool 3: Discrete Cloning Operator Entropy Bounds

The cloning operator $\Psi_{\text{clone}}$ has two stages:
1. **Killing**: Remove walkers with probability $p_{\text{kill}} = \exp(-\beta \tau V_{\text{fit}})$
2. **Revival**: Sample companions and apply inelastic collision

**Entropy change** (from killing):
$$
D_{\text{KL}}(\mu_{\text{alive}} \| \pi_{\text{QSD}}) = D_{\text{KL}}(\mu \| \pi_{\text{QSD}}) + \int \log p_{\text{alive}}(S) \, d\mu(S)
$$

**Entropy change** (from revival):
- Cloning with inelastic collision is a **contraction in Wasserstein distance**
- Use **HWI inequality** to bound entropy change in terms of Wasserstein contraction

**Key formula** (discrete analog of jump operator):
$$
D_{\text{KL}}(\Psi_{\text{clone}}^*\mu \| \pi_{\text{QSD}}) \le D_{\text{KL}}(\mu \| \pi_{\text{QSD}}) + C_{\text{clone}} \cdot W_2(\mu, \pi_{\text{QSD}}) + O(\tau^2)
$$

### Tool 4: Discrete Hypocoercivity Coupling Argument

**Key insight**: The kinetic operator dissipates in velocity, cloning operator contracts in position. We need a **joint metric** that captures both effects.

**Discrete entropy-transport Lyapunov function**:
$$
\mathcal{L}(\mu) := D_{\text{KL}}(\mu \| \pi_{\text{QSD}}) + \alpha \tau \cdot W_2^2(\mu, \pi_{\text{QSD}})
$$

**Contraction under composition**:
$$
\mathcal{L}(\Psi_{\text{total}}^*\mu) \le (1 - \beta\tau) \mathcal{L}(\mu) + O(\tau^2)
$$

for appropriate choice of $\alpha$ and explicit rate $\beta > 0$.

**This is the discrete analog of the mean-field hypocoercivity proof.**

## Proof Architecture (Discrete Version)

### Stage 0: Preliminaries and Framework Setup

**What we inherit from existing framework:**
1. **QSD existence and uniqueness**: From `06_convergence.md` (Foster-Lyapunov theory)
2. **Propagation of chaos**: From `08_propagation_chaos.md` (finite-N → mean-field limit)
3. **Mean-field PDE**: From `07_mean_field.md` (McKean-Vlasov-Fokker-Planck)
4. **Kinetic operator hypocoercivity**: From `04_convergence.md` (Villani's hypocoercivity framework)
5. **Wasserstein contraction**: From `03_cloning.md` (Keystone Principle)

**What we need to establish:**
- QSD regularity properties for discrete-time finite-N (R1-R6 from mean-field document)
- Discrete-time tensorization of LSI constant (N-uniformity)

### Stage 1: Discrete-Time Kinetic Operator Analysis

**Lemma 1.1: BAOAB Hypocoercive Dissipation**

For the kinetic operator $\Psi_{\text{kin}}(\tau)$ implemented via BAOAB integrator:

$$
D_{\text{KL}}(\Psi_{\text{kin}}^*\mu \| \pi_{\text{QSD}}) \le D_{\text{KL}}(\mu \| \pi_{\text{QSD}}) - c_{\text{kin}} \gamma \tau \cdot \mathcal{I}_v(\mu \| \pi_{\text{QSD}}) + O(\tau^2)
$$

where:
- $c_{\text{kin}} > 0$ is the hypocoercivity constant (from Villani's theory)
- $\mathcal{I}_v$ is the relative Fisher information in velocity
- $O(\tau^2)$ includes integrator error and higher-order terms

**Proof strategy**:
1. Decompose BAOAB: $\Psi_{\text{kin}} = \mathbf{B} \circ \mathbf{A} \circ \mathbf{O} \circ \mathbf{A} \circ \mathbf{B}$
2. Track entropy change through each substep
3. Use exact OU solvability of **A** steps to quantify Fisher information dissipation
4. Use **O** step to couple position and velocity entropy
5. Compose and balance to obtain net dissipation

**Key technical point**: The $O(\tau^2)$ term must be **controllable** and not dominate the $O(\tau)$ dissipation.

### Stage 2: Discrete-Time Cloning Operator Analysis

**Lemma 2.1: Killing Operator Entropy Change**

For the killing operator (first stage of cloning):

$$
D_{\text{KL}}(\mu_{\text{alive}} \| \pi_{\text{QSD}}) = D_{\text{KL}}(\mu \| \pi_{\text{QSD}}) + \mathbb{E}_\mu[\log p_{\text{alive}}(S)]
$$

where $p_{\text{alive}}(S) = \frac{1}{N} \sum_{i=1}^N \exp(-\beta \tau V_{\text{fit}}(x_i, v_i))$.

**Lemma 2.2: Revival Operator Wasserstein Contraction**

For the revival operator (second stage of cloning):

$$
W_2(\Psi_{\text{revival}}^*\mu, \pi_{\text{QSD}}) \le (1 - \kappa_x \tau) W_2(\mu, \pi_{\text{QSD}}) + O(\tau^2)
$$

where $\kappa_x > 0$ is the Wasserstein contraction rate (from Keystone Principle).

**Lemma 2.3: HWI Inequality for Discrete Cloning**

Combining killing + revival via HWI inequality:

$$
D_{\text{KL}}(\Psi_{\text{clone}}^*\mu \| \pi_{\text{QSD}}) \le D_{\text{KL}}(\mu \| \pi_{\text{QSD}}) + C_{\text{kill}} \tau + C_{\text{HWI}} W_2(\mu, \pi_{\text{QSD}})
$$

where:
- $C_{\text{kill}}$ bounds the killing entropy expansion
- $C_{\text{HWI}}$ comes from the HWI inequality applied to revival

### Stage 3: Discrete Entropy-Transport Lyapunov Function

**Theorem 3.1: Coupled Contraction**

Define the discrete Lyapunov function:

$$
\mathcal{L}(\mu) := D_{\text{KL}}(\mu \| \pi_{\text{QSD}}) + \alpha \tau \cdot W_2^2(\mu, \pi_{\text{QSD}})
$$

with $\alpha$ chosen to balance kinetic dissipation and cloning expansion.

**Then**:

$$
\mathcal{L}(\Psi_{\text{total}}^*\mu) \le (1 - \beta\tau) \mathcal{L}(\mu) + C_{\text{offset}} \tau^2
$$

where:
- $\beta = c_{\text{kin}} \gamma - C_{\text{clone}}$ (net dissipation rate)
- $C_{\text{offset}}$ collects $O(\tau^2)$ terms

**Kinetic Dominance Condition**: $\beta > 0$ requires friction strong enough to overcome cloning expansion.

### Stage 4: Main Theorem

**Theorem 4.1: Discrete-Time Exponential KL-Convergence**

Under the discrete kinetic dominance condition $\beta > 0$:

$$
D_{\text{KL}}(\mu_{t+1} \| \pi_{\text{QSD}}) \le (1 - \beta\tau) D_{\text{KL}}(\mu_t \| \pi_{\text{QSD}}) + O(\tau^2)
$$

**Exponential convergence**:

$$
D_{\text{KL}}(\mu_t \| \pi_{\text{QSD}}) \le e^{-\beta t} D_{\text{KL}}(\mu_0 \| \pi_{\text{QSD}}) + \frac{C_{\text{offset}} \tau}{\beta}
$$

**Convergence to residual neighborhood**: As $t \to \infty$, $D_{\text{KL}} \to C_{\text{offset}} \tau / \beta$.

**Small time step regime**: For $\tau \to 0$, the residual vanishes and we recover exact QSD convergence.

## Technical Challenges and Solutions

### Challenge 1: Tensorization for Finite-N

**Problem**: Need to show LSI constant is **N-uniform**.

**Solution**:
- Use exchangeability of swarm state under permutations
- Apply Ledoux's tensorization theorem for symmetric functions
- Key: QSD has product structure in mean-field limit

### Challenge 2: $O(\tau^2)$ Error Control

**Problem**: BAOAB integrator has $O(\tau^2)$ error. Must show it doesn't dominate $O(\tau)$ dissipation.

**Solution**:
- Use backward error analysis for BAOAB
- Show modified energy is conserved to $O(\tau^3)$
- Entropy dissipation at $O(\tau)$ with controlled $O(\tau^2)$ perturbation

### Challenge 3: Discrete HWI Inequality

**Problem**: HWI inequality is typically stated for continuous-time gradient flows.

**Solution**:
- Use Otto calculus formulation of HWI
- Apply to discrete-time cloning map
- Key: Cloning is a contraction map, HWI applies to contractions

### Challenge 4: Finite-N vs Mean-Field Gap

**Problem**: Mean-field document analyzes continuum limit; we need finite-N.

**Solution**:
- Use propagation of chaos (08_propagation_chaos.md) to control finite-N error
- All constants have explicit $1/N$ corrections
- Show corrections are $O(1/N)$ and don't affect leading-order rate

## Connection to Mean-Field Result

**Mean-field limit** (16_convergence_mean_field.md):
$$
\frac{d}{dt} D_{\text{KL}}(\rho_t \| \rho_\infty) \le -\delta \, D_{\text{KL}}(\rho_t \| \rho_\infty) + C_{\text{offset}}
$$

where $\delta = \lambda_{\text{LSI}} \sigma^2 - 2\lambda_{\text{LSI}}C_{\text{Fisher}}^{\text{coup}} - C_{\text{KL}}^{\text{coup}} - A_{\text{jump}}$.

**Discrete-time finite-N** (our target):
$$
D_{\text{KL}}(\mu_{t+1} \| \pi_{\text{QSD}}) \le (1 - \beta\tau) D_{\text{KL}}(\mu_t \| \pi_{\text{QSD}}) + C_{\text{offset}} \tau^2
$$

**Consistency**: Taking $N \to \infty$ and $\tau \to 0$ with $t = n\tau$ fixed:
$$
\frac{D_{\text{KL}}(\mu_{t+\tau}) - D_{\text{KL}}(\mu_t)}{\tau} \approx -\beta D_{\text{KL}}(\mu_t) + O(\tau)
$$

**Matching rates**: $\beta \leftrightarrow \delta$ in the continuum limit.

## Expected LSI Constant

From mean-field analysis and dimensional analysis:

$$
C_{\text{LSI}} = \frac{1}{\beta} = O\left(\frac{1}{\gamma \kappa_{\text{conf}} \kappa_x \tau}\right)
$$

where:
- $\gamma$ is friction coefficient
- $\kappa_{\text{conf}}$ is confinement strength (from potential convexity)
- $\kappa_x$ is Wasserstein contraction rate (from cloning)
- $\tau$ is time step size

**Physical interpretation**:
- Stronger friction → faster convergence
- Tighter confinement → faster convergence
- Stronger cloning selection → faster convergence
- Smaller time steps → slower convergence (more discrete steps needed)

## Deliverable

**Target**: Complete, publication-ready proof of {prf:ref}`thm-main-kl-final` at line 1902 of `09_kl_convergence.md`, using the discrete-time hypocoercivity framework described above.

**Structure**:
1. Section introducing discrete hypocoercivity method
2. Lemmas 1.1-2.3 (kinetic dissipation, cloning bounds)
3. Theorem 3.1 (entropy-transport Lyapunov function)
4. Main Theorem 4.1 (exponential KL-convergence)
5. Complete rigorous proofs for all results
6. Connection to mean-field limit (consistency check)

**Rigor standard**: Annals of Mathematics (all epsilon-delta complete, all constants explicit).

**This proof will be the FIRST correct, complete proof of the main KL-convergence theorem for the Euclidean Gas.**
