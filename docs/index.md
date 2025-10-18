# The Fragile Framework: A Mathematical Theory of Adaptive Swarm Intelligence

**A rigorous foundation for stochastic optimization through physics-inspired collective dynamics**

---

## TLDR

The **Fragile Framework** is a complete mathematical theory establishing convergence guarantees for swarm-based stochastic optimization algorithms. It combines three core mechanisms—Langevin dynamics (physics), fitness-based cloning (evolution), and adaptive forces (intelligence)—into provably convergent algorithms for exploring complex state spaces without gradient information.

**Two-Part Structure**:
- **Part I (Euclidean Gas)**: Establishes the stable backbone with exponential convergence to quasi-stationary distribution
- **Part II (Geometric Gas)**: Extends to adaptive mechanisms via perturbation theory while maintaining convergence guarantees

**Main Achievement**: Proof that adaptive intelligence can coexist with mathematical rigor, resolving the fundamental tension between algorithmic power and theoretical tractability.

---

## What is Fragile?

Imagine a swarm of intelligent particles exploring a complex landscape. Each particle:
- **Moves** according to physical dynamics (Langevin diffusion with momentum)
- **Measures** its fitness relative to the swarm
- **Multiplies** when successful, **vanishes** when unsuccessful
- **Adapts** by sensing local geometric structure

What emerges from these simple rules is sophisticated collective intelligence. The Fragile framework proves this intelligence is **mathematically rigorous**: the swarm converges exponentially fast to an optimal exploration distribution with explicit convergence rates.

### The Core Insight

Success spreads through cloning. Information flows through viscous coupling. Intelligence emerges from local adaptation. From these three principles, we build:

1. **Provably stable backbone** (Euclidean Gas) via hypocoercivity theory
2. **Adaptive perturbations** (Geometric Gas) as bounded modifications
3. **Exponential convergence** to quasi-stationary distribution in finite-N and mean-field regimes
4. **N-uniform functional inequalities** (Log-Sobolev, Poincaré, Wasserstein contraction)
5. **Explicit convergence rates** computable from algorithm parameters

---

## The Complete Framework: A Visual Map

The following diagram shows how all chapters connect, from foundational axioms through adaptive extensions to mean-field limits:

```{mermaid}
graph TB
    subgraph "PART I: EUCLIDEAN GAS - The Stable Backbone"
        direction TB

        subgraph "Foundation (Ch 01-03)"
            C01["<b>01: Axiomatic Framework</b><br>State space, axioms<br>Measurement pipeline<br>Valid/Dead domains"]:::foundationStyle
            C02["<b>02: Euclidean Gas</b><br>Langevin + Cloning<br>BAOAB integrator<br>QSD definition"]:::foundationStyle
            C03["<b>03: Cloning Operator</b><br>Keystone Principle<br>KL-contraction<br>Fitness measurement"]:::foundationStyle
        end

        subgraph "Convergence Proofs (Ch 04-06)"
            C04["<b>04: Wasserstein Contraction</b><br>Coupling argument<br>d_W contraction rate<br>Markov chain convergence"]:::convergenceStyle
            C05["<b>05: Kinetic Contraction</b><br>Hypocoercivity for QSD<br>Phase-space distance W_h²<br>Velocity dissipation"]:::convergenceStyle
            C06["<b>06: Foster-Lyapunov</b><br>Drift condition<br>TV-convergence<br>Geometric ergodicity"]:::convergenceStyle
        end

        subgraph "Mean-Field Theory (Ch 07-08)"
            C07["<b>07: Mean-Field Limit</b><br>McKean-Vlasov PDE<br>N → ∞ scaling<br>Continuum dynamics"]:::meanfieldStyle
            C08["<b>08: Propagation of Chaos</b><br>Weak convergence<br>Empirical measure<br>Chaoticity preservation"]:::meanfieldStyle
        end

        subgraph "Advanced Theory (Ch 09-10)"
            C09["<b>09: KL-Convergence & LSI</b><br>N-uniform LSI<br>Exponential KL-decay<br>Entropy production"]:::advancedStyle
            C10["<b>10: QSD Exchangeability</b><br>Permutation symmetry<br>de Finetti theorem<br>Empirical measure limit"]:::advancedStyle
        end
    end

    subgraph "PART II: GEOMETRIC GAS - Adaptive Intelligence"
        direction TB

        subgraph "Adaptive Framework (Ch 11-12)"
            C11["<b>11: Geometric Gas</b><br>ρ-localization pipeline<br>Adaptive force ε_F ∇V_fit<br>Viscous coupling ν F_visc<br>Hessian diffusion Σ_reg<br>Uniform ellipticity (UEPH)<br>Perturbation bounds<br>Critical threshold ε_F*(ρ)<br>Foster-Lyapunov extension"]:::geometricStyle
            C12["<b>12: Symmetries</b><br>Permutation invariance<br>Euclidean equivariance<br>Scaling symmetries<br>Time-reversal asymmetry"]:::symmetryStyle
        end

        subgraph "Regularity Analysis (Ch 13-14)"
            C13["<b>13: C³ Regularity</b><br>Third derivative bounds<br>Telescoping identities<br>k-uniform, N-uniform<br>K_{V,3}(ρ) ~ O(ρ⁻³)<br>BAOAB validation<br>Time-step constraints"]:::regularityStyle
            C14["<b>14: C⁴ Regularity</b><br>Fourth derivative bounds<br>Hessian Lipschitz<br>K_{V,4}(ρ) ~ O(ρ⁻³)<br>Brascamp-Lieb conditions<br>Bakry-Émery calculus"]:::regularityStyle
        end

        subgraph "Convergence Theory (Ch 15-17)"
            C15["<b>15: Geometric LSI Proof</b><br>N-uniform LSI<br>State-dependent diffusion<br>Hypocoercivity extension<br>Explicit parameter threshold<br>Framework Conj 8.3 resolved"]:::lsiStyle
            C16["<b>16: Mean-Field KL-Conv</b><br>McKean-Vlasov entropy<br>Kinetic dominance condition<br>Explicit rate α_net<br>Residual convergence"]:::meanfield2Style
            C17["<b>17: QSD Exchangeability</b><br>Geometric Gas symmetries<br>Adaptive measure limits"]:::advanced2Style
        end
    end

    %% Foundation flows
    C01 --> C02
    C02 --> C03

    %% Convergence proofs build on foundation
    C02 --> C04
    C02 --> C05
    C03 --> C04
    C03 --> C05
    C04 --> C06
    C05 --> C06

    %% Mean-field builds on convergence
    C06 --> C07
    C07 --> C08

    %% Advanced theory uses all prior results
    C05 --> C09
    C06 --> C09
    C08 --> C09
    C03 --> C10
    C09 --> C10

    %% Part II builds on Part I backbone
    C02 --"Provides stable<br>backbone dynamics"--> C11
    C03 --"Keystone Principle<br>extends to adaptive"--> C11
    C05 --"Hypocoercivity<br>structure"--> C11
    C06 --"Foster-Lyapunov<br>framework"--> C11
    C09 --"LSI structure<br>for extension"--> C15
    C10 --"Symmetry<br>foundations"--> C17

    %% Part II internal flow
    C11 --> C12
    C11 --> C13
    C13 --> C14

    C11 --"UEPH bounds"--> C15
    C13 --"∇³ bounds for<br>commutators"--> C15
    C14 --"Hessian Lipschitz"--> C15
    C12 --"Permutation<br>symmetry"--> C15

    C11 --"ρ-framework"--> C16
    C15 --"N-uniform LSI"--> C16
    C11 --> C17
    C15 --> C17

    %% Styling
    classDef foundationStyle fill:#8c6239,stroke:#d4a574,stroke-width:2px,color:#f4e8d8,font-size:12px
    classDef convergenceStyle fill:#3d6b4b,stroke:#7fc296,stroke-width:2px,color:#d8f4e3,font-size:12px
    classDef meanfieldStyle fill:#6b4a8c,stroke:#a47fd4,stroke-width:2px,color:#f0e8f6,font-size:12px
    classDef advancedStyle fill:#4a5f8c,stroke:#8fa4d4,stroke-width:2px,color:#e8eaf6,font-size:12px
    classDef geometricStyle fill:#8c3d5f,stroke:#d47fa4,stroke-width:3px,color:#f4d8e8,font-size:12px
    classDef symmetryStyle fill:#8c6239,stroke:#d4a574,stroke-width:2px,color:#f4e8d8,font-size:12px
    classDef regularityStyle fill:#3d6b4b,stroke:#7fc296,stroke-width:2px,color:#d8f4e3,font-size:12px
    classDef lsiStyle fill:#4a5f8c,stroke:#8fa4d4,stroke-width:3px,color:#e8eaf6,font-size:12px
    classDef meanfield2Style fill:#6b4a8c,stroke:#a47fd4,stroke-width:2px,color:#f0e8f6,font-size:12px
    classDef advanced2Style fill:#4a5f8c,stroke:#8fa4d4,stroke-width:2px,color:#e8eaf6,font-size:12px
```

---

## The Two-Part Architecture

### Part I: The Euclidean Gas (Chapters 01-10)

The **Euclidean Gas** is the provably stable backbone of the framework. It combines:

1. **Langevin Dynamics** - Underdamped diffusion with momentum $(\gamma, \sigma)$
2. **Fitness-Based Cloning** - Stochastic resampling proportional to virtual rewards
3. **Quasi-Stationary Distribution** - Equilibrium conditioned on survival

**Main Results**:
- **Exponential convergence** to QSD in total variation, Wasserstein, and KL-divergence
- **N-uniform Log-Sobolev inequality** with explicit constant
- **Mean-field limit** via McKean-Vlasov PDE with propagation of chaos
- **Multiple convergence proofs** using complementary techniques (coupling, Lyapunov, hypocoercivity)

**Reading Path**: 01 → 02 → 03 → (04 or 05 or 06) → 07 → 08 → 09 → 10

**Key Innovation**: Extension of hypocoercivity theory to quasi-stationary distributions with absorbing boundaries.

---

### Part II: The Geometric Gas (Chapters 11-17)

The **Geometric Gas** extends the stable backbone with three adaptive mechanisms:

1. **Adaptive Force** $\epsilon_F \nabla V_{\text{fit}}[f_k, \rho]$ - Fitness-driven drift
2. **Viscous Coupling** $\nu \mathbf{F}_{\text{viscous}}$ - Swarm interaction forces
3. **Hessian Diffusion** $\Sigma_{\text{reg}} = (H + \epsilon_\Sigma I)^{-1/2}$ - Geometric noise

**Main Results**:
- **Uniform ellipticity by construction** (avoids probabilistic arguments)
- **Perturbation theory** with explicit threshold $\epsilon_F^*(\rho)$
- **C³ and C⁴ regularity** with k-uniform, N-uniform bounds
- **N-uniform LSI for state-dependent diffusion** (resolves Framework Conjecture 8.3)
- **Mean-field KL-convergence** with explicit rate formula

**Reading Path**: 11 → 12 → 13 → (14 optional) → 15 → 16 → 17

**Key Innovation**: The ρ-parameterized framework continuously interpolates between global backbone ($\rho \to \infty$) and hyper-local adaptation ($\rho$ finite).

---

## Chapter-by-Chapter Guide

### Part I: Euclidean Gas Foundation

| Chapter | Title | Type | Essential? | Reading Time |
|:--------|:------|:-----|:-----------|:-------------|
| **01** | Axiomatic Framework | Foundation | ✅ Yes | 30 min |
| **02** | Euclidean Gas Specification | Core Algorithm | ✅ Yes | 45 min |
| **03** | Cloning Operator & Keystone | Core Theory | ✅ Yes | 60 min |
| **04** | Wasserstein Contraction | Convergence Proof | Optional* | 45 min |
| **05** | Kinetic Operator Contraction | Convergence Proof | Optional* | 60 min |
| **06** | Foster-Lyapunov Analysis | Convergence Proof | ✅ Yes | 60 min |
| **07** | Mean-Field Limit | Continuum Theory | ✅ Yes | 45 min |
| **08** | Propagation of Chaos | Mean-Field Proof | Optional | 45 min |
| **09** | KL-Convergence & LSI | Functional Inequalities | ✅ Yes | 75 min |
| **10** | QSD Exchangeability | Symmetry Theory | Optional | 30 min |

*Choose at least one of Ch 04, 05, or 06 to understand convergence proofs.

---

### Part II: Geometric Gas Extensions

| Chapter | Title | Type | Essential? | Reading Time |
|:--------|:------|:-----|:-----------|:-------------|
| **11** | Geometric Gas Framework | Core Extension | ✅ Yes | 90 min |
| **12** | Symmetries of Geometric Gas | Structural Theory | Optional | 30 min |
| **13** | C³ Regularity Analysis | Technical Bounds | ✅ Yes | 60 min |
| **14** | C⁴ Regularity Analysis | Advanced Theory | Optional | 45 min |
| **15** | N-Uniform LSI Proof | Convergence Theory | ✅ Yes | 90 min |
| **16** | Mean-Field KL-Convergence | Continuum Limit | ✅ Yes | 60 min |
| **17** | QSD Exchangeability (Geometric) | Symmetry Theory | Optional | 30 min |

---

## Main Results Summary

### Convergence Theorems

**Finite-N Euclidean Gas** (Chapters 04-06, 09):

$$
D_{\text{KL}}(\rho_n \| \pi_{\text{QSD}}) \leq e^{-\lambda_{\text{LSI}} n} D_{\text{KL}}(\rho_0 \| \pi_{\text{QSD}})
$$

- Log-Sobolev constant $\lambda_{\text{LSI}}$ is **N-uniform**
- Convergence in TV, Wasserstein, and KL-divergence
- Explicit rates computable from $(\gamma, \sigma, U)$

**Mean-Field Euclidean Gas** (Chapters 07-09):

$$
\frac{d}{dt} D_{\text{KL}}(\rho_t \| \rho_\infty) \leq -\lambda_{\text{LSI}} I_{\text{Fisher}}(\rho_t \| \rho_\infty)
$$

- McKean-Vlasov PDE limit as $N \to \infty$
- Propagation of chaos with Wasserstein-2 bounds
- LSI for infinite-dimensional generator

**Finite-N Geometric Gas** (Chapters 11, 15):

For $\epsilon_F < \epsilon_F^*(\rho)$:

$$
\mathbb{E}[\Delta V_{\text{total}}] \leq -\kappa_{\text{total}}(\rho) V_{\text{total}} + C_{\text{total}}(\rho)
$$

where:

$$
\kappa_{\text{total}}(\rho) = \kappa_{\text{backbone}} - \epsilon_F K_F(\rho) - C_{\text{diff,1}}(\rho) > 0
$$

- Perturbation analysis with explicit threshold
- N-uniform LSI for state-dependent diffusion
- Viscous coupling $\nu$ has no constraint (purely dissipative)

**Mean-Field Geometric Gas** (Chapter 16):

$$
D_{\text{KL}}(\rho_t \| \rho_\infty) \leq e^{-\alpha_{\text{net}} t} D_{\text{KL}}(\rho_0 \| \rho_\infty) + \frac{C_{\text{offset}}}{\alpha_{\text{net}}}
$$

where:

$$
\alpha_{\text{net}} = \lambda_{\text{LSI}} \sigma^2 - 2\lambda_{\text{LSI}}C_{\text{Fisher}}^{\text{coup}} - C_{\text{KL}}^{\text{coup}} - A_{\text{jump}}
$$

- Kinetic dominance condition: $\sigma^2 > \sigma_{\text{crit}}^2$
- Residual convergence (offset term remains)
- Explicit formulas for all constants

---

### Key Structural Results

**Keystone Principle** (Chapter 03):

The cloning operator contracts KL-divergence for fixed particle configurations:

$$
D_{\text{KL}}(\Psi_{\text{clone}}[\mu] \| \Psi_{\text{clone}}[\nu]) \leq D_{\text{KL}}(\mu \| \nu)
$$

This is the foundation of all convergence proofs.

**Uniform Ellipticity by Construction** (Chapter 11):

The regularized Hessian satisfies:

$$
c_{\min}(\rho) I \preceq \Sigma_{\text{reg}}^2 \preceq c_{\max}(\rho) I
$$

with **k-uniform, N-uniform** bounds for all $\rho > 0$, ensuring well-posed dynamics.

**Telescoping Identities** (Chapter 13):

Normalized localization weights satisfy:

$$
\sum_{j \in A_k} \nabla^m w_{ij}(\rho) = 0 \quad \forall m \geq 1
$$

This prevents growth with walker count $k$, ensuring N-uniformity.

---

## Mathematical Innovations

### 1. Hypocoercivity for Quasi-Stationary Distributions

Standard hypocoercivity (Villani 2009) applies to conservative dynamics. We extend to:
- **Absorbing boundaries** with state-dependent killing
- **Cloning revival** operator with proportional resampling
- **Modified Lyapunov functionals** combining position variance and kinetic energy

**Impact**: Enables convergence proofs for non-conservative systems with measurement.

---

### 2. Perturbation Theory for State-Dependent Diffusion

Classical LSI proofs assume **constant diffusion**. We extend via:
- **Uniform ellipticity bounds** ensuring coercivity preservation
- **C³ regularity** controlling commutator errors
- **Modified hypocoercive coupling** with state-dependent metrics

**Impact**: First rigorous LSI for information-geometric diffusion in swarm context.

---

### 3. ρ-Parameterized Measurement Pipeline

Unified framework interpolating between:
- **Global backbone** ($\rho \to \infty$): Proven stability, parameter-independent bounds
- **Hyper-local adaptation** ($\rho \to 0$): Geometric exploration, $O(\rho^{-3})$ scaling

**Impact**: Design guidance for exploration-exploitation tradeoff with explicit stability thresholds.

---

### 4. N-Uniform Functional Inequalities

All bounds are **independent of swarm size** $N$:
- Log-Sobolev constant $\lambda_{\text{LSI}}$
- Poincaré constant $C_P$
- Wasserstein contraction rate
- Foster-Lyapunov drift rate $\kappa_{\text{total}}$

**Impact**: Scalability to large swarms with no performance degradation.

---

## Reading Paths by Interest

### For Algorithm Practitioners

**Goal**: Understand the algorithm and parameter selection

**Path**:
1. Part I Ch 02 (Euclidean Gas specification)
2. Part I Ch 03 §0-1 (Cloning mechanism)
3. Part I Ch 06 §0-1 (Convergence overview)
4. Part II Ch 11 §0-3 (Geometric Gas and ρ-framework)
5. Part II Ch 13 §1.2, §10 (Time-step constraints)
6. Part II Ch 15 §9 (Parameter thresholds)

**Time**: ~4 hours

---

### For Convergence Theory

**Goal**: Understand mathematical proofs

**Path**:
1. Part I Ch 01-03 (Foundations)
2. Part I Ch 05 (Hypocoercivity for kinetic operator)
3. Part I Ch 06 (Foster-Lyapunov framework)
4. Part I Ch 09 (LSI proof for Euclidean Gas)
5. Part II Ch 11 §5-7 (Perturbation analysis)
6. Part II Ch 15 (LSI extension to state-dependent diffusion)

**Time**: ~8 hours

---

### For Mean-Field Theory

**Goal**: Understand continuum limits

**Path**:
1. Part I Ch 02-03 (Algorithm specification)
2. Part I Ch 07 (McKean-Vlasov PDE derivation)
3. Part I Ch 08 (Propagation of chaos)
4. Part I Ch 09 §7-8 (Mean-field LSI)
5. Part II Ch 11 §1-3 (ρ-framework)
6. Part II Ch 16 (Mean-field KL-convergence)

**Time**: ~6 hours

---

### For Framework Designers

**Goal**: Understand architectural principles

**Path**:
1. Part I Ch 01 (Axiomatic framework)
2. Part I Ch 03 (Keystone Principle)
3. Part I Ch 10 (Exchangeability and symmetry)
4. Part II Ch 11 §1.2-1.3 (Stable backbone + adaptive perturbation)
5. Part II Ch 12 (Symmetries)
6. Part II Ch 15 §1.2 (Resolution strategy)

**Time**: ~5 hours

---

### Complete Sequential Reading

**For full mathematical understanding**:

1. **Week 1**: Part I Ch 01-03 (Foundations)
2. **Week 2**: Part I Ch 04-06 (Convergence proofs, choose your favorite)
3. **Week 3**: Part I Ch 07-09 (Mean-field and LSI)
4. **Week 4**: Part I Ch 10 + Part II Ch 11 (Symmetry + Geometric Gas)
5. **Week 5**: Part II Ch 12-14 (Symmetries and regularity)
6. **Week 6**: Part II Ch 15-17 (Geometric convergence theory)

**Total Time**: ~30-40 hours of focused reading

---

## Notation Conventions

### State Space
- $\mathcal{X} \subset \mathbb{R}^d$: Position state space (Valid Domain)
- $\mathcal{V} = \{v : \|v\| \leq V_{\text{alg}}\}$: Velocity ball
- $N$: Total swarm size (walkers + dead)
- $k$: Alive walker count ($k \leq N$)

### Swarm State
- $S = \{(x_i, v_i, s_i)\}_{i=1}^N$: Full swarm configuration
- $f_k = \frac{1}{k}\sum_{i \in A_k} \delta_{(x_i, v_i)}$: Empirical measure of alive walkers
- $A_k$: Set of alive walker indices

### Operators
- $\Psi_{\text{kin}}(\tau)$: Kinetic operator (Langevin evolution)
- $\Psi_{\text{clone}}$: Cloning operator (fitness-based resampling)
- $\mathcal{L}$: Generator of the Markov process
- $\mathcal{L}_{\text{kin}}$: Kinetic generator (underdamped Langevin)

### Parameters (Euclidean Gas)
- $\gamma$: Friction coefficient
- $\sigma$: Velocity diffusion strength
- $U(x)$: Confining potential
- $\tau$: Integration time step

### Parameters (Geometric Gas)
- $\rho$: Localization scale for adaptive measurement
- $\epsilon_F$: Adaptation rate (weight of adaptive force)
- $\epsilon_\Sigma$: Hessian regularization parameter
- $\nu$: Viscous coupling strength

### Key Functions
- $V_{\text{fit}}[f_k, \rho](x_i)$: ρ-localized fitness potential
- $\pi_{\text{QSD}}$: Quasi-stationary distribution
- $D_{\text{KL}}(\mu \| \nu)$: Kullback-Leibler divergence
- $W_2(\mu, \nu)$: Wasserstein-2 distance

### Constants
- $\kappa_{\text{backbone}}$: Backbone stability rate
- $\epsilon_F^*(\rho)$: Critical adaptation rate threshold
- $\lambda_{\text{LSI}}$: Log-Sobolev constant
- $K_{V,3}(\rho), K_{V,4}(\rho)$: Regularity bound constants

---

## Implementation

The mathematical theory is implemented in Python using PyTorch for vectorization:

- **`src/fragile/euclidean_gas.py`**: Base Euclidean Gas implementation
- **`src/fragile/adaptive_gas.py`**: Geometric Gas with adaptive mechanisms
- **`src/fragile/gas_parameters.py`**: Pydantic parameter configurations
- **`src/fragile/shaolin/`**: Visualization tools (HoloViews + Panel)

**Key Design Principles**:
- First dimension is always $N$ (number of walkers)
- PyTorch tensors for all state representations
- Pydantic models for parameter validation
- Mathematical notation matches documentation

See `CLAUDE.md` for detailed development guidelines.

---

## Open Questions and Future Work

While the framework establishes complete convergence theory, several frontiers remain:

**From Part I**:
- Sharpness of convergence rate constants (optimality of LSI constant)
- Extension to non-convex confining potentials
- Adaptive selection of kinetic parameters $(\gamma, \sigma)$

**From Part II**:
- C⁴ regularity for full swarm-dependent measurement (currently simplified model)
- Global convergence to QSD (currently residual neighborhood)
- Necessity of kinetic dominance condition (currently sufficient only)
- Uniform convexity verification for Brascamp-Lieb inequalities

**General**:
- Extension to infinite-dimensional state spaces (PDEs, functional optimization)
- Connection to reinforcement learning (policy gradient methods)
- Numerical analysis of higher-order integrators (beyond BAOAB)

---

## Mathematical Reference

The framework includes comprehensive mathematical documentation:

- **[Mathematical Glossary](glossary)**: Comprehensive index of all mathematical entries
- **[Part I Introduction](source/1_euclidean_gas/intro_euclidean_gas)**: Overview of Euclidean Gas chapters
- **[Part II Introduction](source/2_geometric_gas/00_intro_geometric_gas)**: Overview of Geometric Gas chapters

**Usage**:
1. **Quick lookup**: Use the glossary to find definitions/theorems by tags
2. **Detailed statements**: Read the full chapter documents
3. **Deep understanding**: Follow the reading paths in the chapter introductions

---

## Community and Development

**Contributing**: See `CLAUDE.md` for:
- Mathematical documentation standards (Jupyter Book/MyST markdown)
- Dual review protocol (Gemini + Codex for mathematical proofs)
- Code conventions (vectorization, PyTorch, Pydantic)
- Testing philosophy (algorithmic correctness + physical principles)

**Development Tools**:
- `make test`: Run test suite
- `make docs`: Build Jupyter Book documentation
- `make lint`: Code quality checks (ruff, mypy)
- `make all`: Complete workflow (lint + docs + test)

---

## Navigation

Use the table of contents below to explore the complete framework:

```{tableofcontents}
```

---

:::{epigraph}
"The whole is greater than the sum of its parts, but understanding each part illuminates the whole."

-- Aristotle (adapted)
:::

Welcome to the Fragile framework. From simple axioms to sophisticated adaptive algorithms, from finite swarms to continuum limits, from stable backbones to intelligent perturbations—this is the complete mathematical journey of collective intelligence.

**Start your exploration**: [Part I - Euclidean Gas](source/1_euclidean_gas/intro_euclidean_gas.md) | [Part II - Geometric Gas](source/2_geometric_gas/00_intro_geometric_gas.md)
