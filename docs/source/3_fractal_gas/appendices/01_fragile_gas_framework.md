# Mathematical Foundations of Fragile

## 0. TLDR

**Parametric Axiomatic Framework**: This document establishes Fragile Gas as a general class of adaptive swarm algorithms through a rigorous axiomatic framework where every stability assumption is reframed as a quantifiable parameter. This transforms algorithm debugging from guesswork into systematic diagnosis—when the algorithm fails, the axiomatic parameter values reveal exactly which components need adjustment.

**Guaranteed Revival and Viability**: The framework proves that any instantiation satisfying the Axiom of Guaranteed Revival ($\kappa_{\text{revival}} > 1$) ensures almost-sure resurrection of dead walkers, preventing gradual swarm extinction through attrition. This establishes a fundamental dichotomy: either the swarm operates indefinitely in a quasi-stationary regime, or it faces catastrophic extinction from large coherent noise events—gradual extinction through attrition is mathematically ruled out as long as a single walker remains alive.

**Mean-Square Continuity and Scalability**: The document derives comprehensive continuity bounds for every operator in the Fragile Gas pipeline, proving that under normal operation (low attrition), the measurement noise is $O(1)$ with respect to swarm size $N$, establishing N-uniform stability. However, the analysis reveals extreme sensitivity to regularization parameters, with error amplification scaling as $O(\varepsilon_{\text{std}}^{-6})$, making careful parameter selection critical for practical stability.

**Canonical Instantiation as Existence Proof**: The framework's rigor is validated by proving that a Canonical Fragile Swarm (Euclidean space, Gaussian noise, empirical aggregators) satisfies all foundational axioms (one structural assumption and sixteen parametric axioms). This existence proof demonstrates that the axiomatic constraints are neither vacuous nor contradictory, providing a concrete, verifiable baseline for implementers.

## 1. Introduction

### 1.1. Goal and Scope

The goal of this document is to establish the **Mathematical Foundations of the Fragile Gas Framework**, a general class of adaptive particle swarm algorithms inspired by statistical physics and natural selection. The Fragile Gas framework provides a rigorous analytical foundation for understanding how stochastic particle systems can efficiently explore complex state spaces through a combination of measurement, selection, and adaptive dynamics.

The central object of study is the **Swarm Update Operator** $\Psi_{\mathcal{F}}$, a sophisticated composition of measurement, standardization, fitness evaluation, perturbation, status checking, and cloning operations that evolves a swarm of $N$ walkers from one discrete time step to the next. This document provides complete, self-contained definitions of all components of this operator and establishes the minimal axiomatic requirements any valid instantiation must satisfy.

The main analytical achievements of this work are threefold:

1. **Axiomatic Parametrization**: We reframe every stability assumption as a quantifiable **Axiomatic Parameter** (e.g., $\kappa_{\text{revival}}$, $L_R$, $p_{\text{worst-case}}$), transforming abstract requirements into concrete diagnostic tools. This parametric framework enables systematic debugging: when an algorithm fails, the parameter values identify which components violate their required bounds.

2. **Operator-Level Continuity Analysis**: We prove comprehensive continuity bounds (both deterministic Lipschitz and mean-square probabilistic) for every operator in the pipeline—measurement, aggregation, standardization, rescaling, perturbation, status update, and cloning. These results establish that the Fragile Gas is **N-uniform stable** under normal operation, meaning measurement noise does not grow with swarm size.

3. **Guaranteed Revival Mechanism**: We prove the **Theorem of Almost-Sure Revival** {prf:ref}`thm-revival-guarantee`, establishing that under the Axiom of Guaranteed Revival, any swarm with at least one surviving walker will resurrect all dead walkers with probability 1. This prevents gradual extinction through attrition, confining the system's failure modes to catastrophic events only.

**Scope and Limitations**: This document focuses exclusively on defining the framework and proving operator-level stability properties. It does *not* prove convergence to quasi-stationary distributions—that analysis requires the Keystone Principle and hypocoercive Lyapunov analysis, which are developed in the companion document *"The Keystone Principle and the Contractive Nature of Cloning"* {prf:ref}`03_cloning.md`. This document establishes the *tools* (metrics, axioms, continuity bounds); the companion document establishes the *dynamics* (drift inequalities, convergence rates).

### 1.2. The Fragile Gas as a Parametric Debugging Framework

The Fragile Gas framework represents a paradigm shift in how we design and analyze swarm algorithms. Traditional approaches define algorithms through implementation details (pseudocode, hyperparameters) and validate them empirically. The Fragile framework inverts this: we first establish a set of **functional requirements** (axioms) that any well-behaved swarm must satisfy, then prove that *any instantiation satisfying these axioms* inherits strong stability and convergence guarantees.

The key innovation is **parametrization of axioms**. Rather than stating abstract conditions like "the reward function must be regular," we define a parameter $L_R$ (the Lipschitz constant of the reward) and prove how this parameter propagates through every downstream continuity bound. When an implementation fails, the user can measure $L_R$ in their environment and immediately diagnose whether reward irregularity is the root cause.

This parametric perspective transforms the 21 sections of this document into a **diagnostic manual**:

- **Sections 3** defines the axiomatic parameters and their roles (viability, exploration, measurement quality)
- **Sections 4-16** show how these parameters combine to bound each operator's continuity
- **Section 17** proves the revival mechanism that prevents attrition-based failure
- **Section 18** composes all operators into the full Swarm Update Operator $\Psi_{\mathcal{F}}$
- **Section 20** provides an existence proof via the Canonical Fragile Swarm
- **Section 21** consolidates all continuity constants for quick reference

:::{admonition} Why "Fragile"?
:class: note

The name "Fragile" reflects the algorithm's fundamental property: walkers are fragile entities that can die (status $s=0$) when they violate environmental constraints. This fragility is not a weakness but the *source of adaptation*. Dead walkers mark dangerous regions, and the cloning operator reallocates computational resources away from these regions toward successful paths. The framework proves this fragile-plus-revival architecture is mathematically sound and guarantees long-term viability.
:::

### 1.3. Overview of the Proof Strategy and Document Structure

The document builds the framework in four main parts: **Foundational Definitions** (Sections 2-5), **Measurement and Standardization Pipeline** (Sections 6-12), **Dynamics and Cloning** (Sections 13-17), and **System Composition and Validation** (Sections 18-21). Each part establishes necessary mathematical machinery for the next, culminating in a complete, verified algorithmic definition.

The diagram below illustrates the logical dependencies and information flow through the framework:

```{mermaid}
graph TD
    subgraph "Part I: Foundations (§2-5)"
        A["<b>§3: Axiomatic Foundations</b><br>Defines 16 core axioms + 1 assumption <br>with quantifiable parameters"]:::axiomStyle
        B["<b>§4: Environment</b><br>Reward function R, <br>Valid domain X_valid"]:::stateStyle
        C["<b>§5: Algorithmic Noise</b><br>Perturbation (P_σ) and <br>Cloning (Q_δ) measures"]:::stateStyle
        A --> B
        A --> C
    end

    subgraph "Part II: Measurement Pipeline (§6-12)"
        D["<b>§6: Algorithmic Space</b><br>Projection φ: X → Y, <br>Algorithmic distance d_Y"]:::stateStyle
        E["<b>§7: Swarm Measuring</b><br>Empirical measure aggregation, <br>Lipschitz continuity"]:::lemmaStyle
        F["<b>§8: Companion Selection</b><br>Fitness-weighted selection, <br>Error bounds"]:::lemmaStyle
        G["<b>§9-12: Standardization</b><br>Rescale transformation, <br>Distance measurement, <br>Z-score standardization"]:::lemmaStyle

        B --> D
        D --> E
        E --> F
        F --> G
        A --"Measurement Quality Axioms"--> E
        A --"Measurement Quality Axioms"--> G
    end

    subgraph "Part III: Dynamics (§13-17)"
        H["<b>§13: Fitness Potential</b><br>V_fit composition from <br>rescaled reward & diversity"]:::stateStyle
        I["<b>§14: Perturbation</b><br>Probabilistic continuity, <br>McDiarmid bound"]:::theoremStyle
        J["<b>§15: Status Update</b><br>Death boundary checks, <br>Hölder continuity"]:::lemmaStyle
        K["<b>§16: Cloning Measure</b><br>Threshold-based selection, <br>Continuity bounds"]:::theoremStyle
        L["<b>§17: Revival at k=1</b><br>Single-survivor dynamics, <br>Almost-sure resurrection"]:::theoremStyle

        G --> H
        H --> I
        I --> J
        J --> K
        A --"Axiom of Guaranteed Revival"--> K
        A --"Exploration Axioms"--> I
        A --"Viability Axioms"--> J
        K --> L
    end

    subgraph "Part IV: Composition (§18-21)"
        M["<b>§18: Swarm Update Ψ_F</b><br>Markov kernel composition, <br>W_2 continuity bounds"]:::theoremStyle
        N["<b>§19: Fragile Gas Algorithm</b><br>Time-homogeneous Markov chain, <br>Trajectory generation"]:::stateStyle
        O["<b>§20: Canonical Instantiation</b><br>Existence proof via <br>Euclidean space + Gaussian noise"]:::theoremStyle
        P["<b>§21: Constants Glossary</b><br>Consolidated continuity bounds <br>and parameter impacts"]:::stateStyle

        L --> M
        I --> M
        J --> M
        M --> N
        A --"All axioms"--> O
        M --> O
        O --> P
    end

    classDef stateStyle fill:#4a5f8c,stroke:#8fa4d4,stroke-width:2px,color:#e8eaf6
    classDef axiomStyle fill:#8c6239,stroke:#d4a574,stroke-width:2px,stroke-dasharray: 5 5,color:#f4e8d8
    classDef lemmaStyle fill:#3d6b4b,stroke:#7fc296,stroke-width:2px,color:#d8f4e3
    classDef theoremStyle fill:#8c3d5f,stroke:#d47fa4,stroke-width:3px,color:#f4d8e8
```

**Document Structure**:

- **Part I (Sections 2-5)**: Establishes the foundational objects. Section 2 defines global conventions (walkers, swarms, state spaces, metrics). Section 3 defines the complete axiomatic framework with 16 core axioms plus 1 structural assumption, organized into three categories: Viability (preventing extinction), Environmental (problem structure), and Regularity (algorithmic stability). Section 4 defines the environment (reward function, valid domain, state space) and Section 5 defines the algorithmic noise measures (perturbation and cloning).

- **Part II (Sections 6-12)**: Develops the measurement and standardization pipeline. Section 6 introduces the algorithmic space and projection map. Section 7 proves empirical aggregation is Lipschitz continuous with respect to $W_2$ distance. Section 8 analyzes companion selection under status changes. Sections 9-12 derive the complete standardization pipeline: rescale transformation (smooth patching, monotonicity, Lipschitz bounds), distance-to-companion measurement (error decomposition, structural continuity), and z-score standardization (mean-square continuity, $O(\varepsilon_{\text{std}}^{-6})$ sensitivity).

- **Part III (Sections 13-17)**: Analyzes the dynamical operators. Section 13 defines the fitness potential operator and proves its continuity. Section 14 establishes probabilistic continuity of the perturbation operator ({prf:ref}`def-perturbation-operator`) via McDiarmid's inequality. Section 15 analyzes the status update operator ({prf:ref}`def-status-update-operator`) and its Hölder continuous death probability under boundary regularity. Section 16 defines the cloning transition measure and proves its continuity. Section 17 handles the special case of single-survivor revival ($k=1$), proving almost-sure resurrection under the Axiom of Guaranteed Revival.

- **Part IV (Sections 18-21)**: Composes and validates the complete system. Section 18 defines the Swarm Update Operator $\Psi_{\mathcal{F}}$ as a Markov kernel and derives its $W_2$ continuity bound. Section 19 formally defines the Fragile Gas algorithm as a time-homogeneous Markov chain. Section 20 provides an existence proof by defining a Canonical Fragile Swarm and proving axiom-by-axiom that it satisfies all requirements. Section 21 consolidates all continuity constants in a reference table.

**Reading Recommendations**:

- **For implementers**: Read Sections 1-5 (foundations and axioms), Section 19 (algorithm definition), Section 20 (canonical instantiation), and Section 21 (parameter glossary). This provides the complete algorithmic specification and a validated baseline implementation.

- **For theorists**: The full document is a prerequisite for the companion convergence analysis in *"The Keystone Principle and the Contractive Nature of Cloning."* Pay special attention to Section 12 (standardization continuity), Section 16 (cloning measure), and Section 18 (composed operator bounds).

- **For debuggers**: When an implementation fails, use Section 21 to identify which continuity constants are large, then trace back through the document to find the corresponding axiomatic parameter. For example, if fitness variance is unstable, check $\varepsilon_{\text{std}}$ (Section 9), variance axioms (Section 3), and $L_R$ (Section 4).

## 2. Global Conventions: Foundational Objects and Core Algorithmic Parameters

This section establishes the foundational objects of the system, the assumptions about the environment, and the core tunable parameters that dictate the algorithm's execution. For the mathematical framework to be sound, these conventions must be fixed before the analysis begins.

:::{admonition} 💡 Visualization Trick
:class: tip
Think of this section as setting up the "rules of the game." Just as chess needs to define what pieces exist and how they move before analyzing strategy, we need to define walkers, swarms, and their properties before we can prove anything about their behavior.
:::

### 1.1 The Walker

The fundamental unit of the system is the **walker**. A walker represents a single agent within the swarm, characterized by its position in a state space and its survival status.

:::{note}
Think of each walker as a scout exploring an unknown landscape. They can be either "alive" (actively exploring) or "dead" (hit an obstacle or boundary). The beauty of this simple binary status is that it allows the swarm to remember both successful paths and dangerous areas.
:::

:::{prf:definition} Walker
:label: def-walker

A **walker ({prf:ref}`def-walker`)**, denoted $w$, is a tuple consisting of a position and a status:

$$
w := (x, s)

$$

where:
1.  $x \in \mathcal{X}$ is the walker ({prf:ref}`def-walker`)'s **position** in a state space $\mathcal{X}$.
2.  $s \in \{0, 1\}$ is the walker ({prf:ref}`def-walker`)'s **survival status**. A status of $s=1$ indicates the walker is **alive**, while $s=0$ indicates it is **dead**.
:::

::: {note} Framework Abstraction
This minimal definition isolates the position and status components required by every fragile-gas instantiation. Specific models may enrich the walker ({prf:ref}`def-walker`) state with additional continuous variables—for example, the Euclidean Gas augments the representation to $(x, v, s)$. The status-dependent arguments below apply unchanged to such extensions.
:::

### 1.2 The Swarm and its State Space

A **swarm** is a fixed-size collection of N walker ({prf:ref}`def-walker`)s. The state of the entire system at any point in time is described by the state of the swarm.

:::{admonition} 🎯 Key Insight
:class: important
Why keep dead walker ({prf:ref}`def-walker`)s in the swarm? This is counterintuitive but brilliant! Dead walkers serve as "memory" of dangerous regions. When we later "revive" them by cloning successful walkers, we're essentially saying: "Forget that failed path, try starting from this successful position instead." This creates a natural selection pressure toward promising regions.
:::

:::{prf:definition} Swarm and Swarm State Space
:label: def-swarm-and-state-space

A **swarm**, denoted $\mathcal{S}$, is an N-tuple of walkers ({prf:ref}`def-walker`):

$$
\mathcal{S} := (w_1, w_2, \dots, w_N)

$$

The **Swarm State Space ({prf:ref}`def-swarm-and-state-space`)**, denoted $\Sigma_N$, is the set of all possible swarms of size N. It is the N-fold Cartesian product of the single-walker ({prf:ref}`def-walker`) state space:

$$
\Sigma_N := (\mathcal{X} \times \{0, 1\})^N

$$

:::

For any swarm $\mathcal{S}$, we define two critical index sets that partition the walker ({prf:ref}`def-walker`)s based on their survival status.

:::{prf:definition} Alive and Dead Sets
:label: def-alive-dead-sets

For any swarm state $\mathcal{S} = ((x_1, s_1), \dots, (x_N, s_N)) \in \Sigma_N$ ({prf:ref}`def-swarm-and-state-space`):

1.  The **alive set ({prf:ref}`def-alive-dead-sets`)**, $\mathcal{A}(\mathcal{S})$, is the set of indices of all walker ({prf:ref}`def-walker`)s with a survival status of 1.

$$
\mathcal{A}(\mathcal{S}) := \{i \in \{1, \dots, N\} \mid s_i = 1\}

$$

2.  The **dead set ({prf:ref}`def-alive-dead-sets`)**, $\mathcal{D}(\mathcal{S})$, is the set of indices of all walker ({prf:ref}`def-walker`)s with a survival status of 0.

$$
\mathcal{D}(\mathcal{S}) := \{i \in \{1, \dots, N\} \mid s_i = 0\}

$$

:::
### 1.3 Scope of Analysis

This framework analyzes the dynamics of the swarm as a Markov process on the state space $\Sigma_N$. A critical feature of this process is the existence of an absorbing "cemetery state" where the set of alive walker ({prf:ref}`def-walker`)s is empty.

:::{admonition} Common Pitfall
:class: dropdown warning

The "cemetery state" (all walker ({prf:ref}`def-walker`)s dead) is an absorbing state—once entered, there's no escape! This is why the revival mechanism (Section 16) is so critical. It ensures that as long as even one walker survives, the swarm can rebuild itself. Without this mechanism, the algorithm would inevitably fail through gradual attrition.
:::

:::{hint}
The three-part analytical structure mirrors biological evolution: (1) avoiding extinction, (2) adapting while surviving, and (3) recovering from near-extinction events. Each part requires different mathematical machinery, which is why we partition the analysis this way.
:::

*   **State Space Partition:** The space of all swarms $\Sigma_N$ is partitioned into two disjoint sets:
    1.  The set of **alive swarms**, where there is at least one alive walker ({prf:ref}`def-walker`):

$$
\Sigma_N^{\mathrm{alive}} := \{\mathcal S \in \Sigma_N: |\mathcal A(\mathcal S)| \ge 1\}

$$

    2.  The **cemetery state** $\mathcal{S}_\emptyset$, which represents the unique swarm configuration where the alive set ({prf:ref}`def-alive-dead-sets`) is empty: $\mathcal{A}(\mathcal{S}_\emptyset) = \emptyset$. This corresponds to the state where $s_i = 0$ for all $i \in \{1, \dots, N\}$.

*   **Absorbing State:** The algorithm's definition specifies that it terminates if the alive set ({prf:ref}`def-alive-dead-sets`) becomes empty. Therefore, the cemetery state is an **absorbing state** of the Markov process: once entered, the system remains there.

*   **Analytical Goals:** The stability analysis is therefore threefold:
    1.  To determine the conditions under which the probability of being absorbed into the cemetery state is minimized.
    2.  To characterize the system's dynamics and convergence properties *conditioned on its survival* (i.e., for trajectories that remain within $\Sigma_N^{\mathrm{alive}}$ where the number of alive walker ({prf:ref}`def-walker`)s is greater than one).
    3.  To analyze the system's dynamics in the boundary state where $|\mathcal A(\mathcal S)| = 1$ as a special case, proving the conditions under which it serves as a revival mechanism ({prf:ref}`axiom-guaranteed-revival`) preventing swarm extinction through attrition.

All theorems and proofs concerning long-term dynamic behavior (e.g., convergence to a stationary distribution) are stated for the dynamics within $\Sigma_N^{\mathrm{alive}}$. The risk of absorption into the cemetery state is analyzed separately as a primary measure of the algorithm's robustness.

### 1.4 Defining a Valid State Space

For the analytical framework to be sound, the State Space $(\mathcal{X}, d_{\mathcal{X}}, \mu_{\mathcal{X}})$ in which walkers evolve must satisfy a set of foundational properties. Any space that meets these criteria is considered a **Valid State Space**.

:::{prf:definition} Valid State Space
:label: def-valid-state-space

A **Valid State Space ({prf:ref}`def-valid-state-space`)** is a tuple $(\mathcal{X}, d_{\mathcal{X}}, \mu_{\mathcal{X}})$ with the following properties:

1.  **Topological Structure:** The space $(\mathcal{X}, d_{\mathcal{X}})$ must be a **Polish space** (a complete, separable metric space). This ensures that notions of convergence and probability measures are well-defined.

2.  **Measure Structure:** The space must be equipped with a **reference measure** $\mu_{\mathcal{X}}$ (e.g., Lebesgue measure for Euclidean spaces, or the Riemannian volume measure for manifolds). This measure is used to define probability densities for the noise kernels.

3.  **Existence of Valid Noise:** The space must support a **Valid Noise Measure** ($\mathcal{P}_\sigma$ and $\mathcal{Q}_\delta$) as per {prf:ref}`def-valid-noise-measure`. This is the most critical functional requirement, as it implies the space has enough geometric regularity to satisfy:
    *   The Axiom of Bounded Second Moment of Perturbation ({prf:ref}`axiom-non-degenerate-noise`).
    *   The Axiom of Boundary Regularity ({prf:ref}`axiom-boundary-regularity`).

4.  **Regularity of the Domain:** The **Valid Domain ({prf:ref}`def-valid-state-space`)** $\mathcal{X}_{\mathrm{valid}} \subset \mathcal{X}$ must have a boundary $\partial \mathcal{X}_{\mathrm{valid}}$ that is a null set with respect to any admissible noise measure. For example, if $\mathcal{X}$ is a smooth manifold, requiring a $C^1$ boundary is sufficient.

This axiomatic definition provides flexibility while maintaining rigor. The framework's proofs do not depend on the space being Euclidean, but on it satisfying these functional properties.
:::

:::{admonition} Examples of Valid State Spaces
:class: note

*   **Canonical Example (Euclidean Space):** A bounded subset of $\mathbb{R}^d$ with the Euclidean metric and Lebesgue measure. This is the simplest instantiation.
*   **Generalization (Riemannian Manifold):** A compact, smooth Riemannian manifold $(\mathcal{M}, g)$. This space is Polish, has a natural volume measure, and its heat kernel provides a valid noise measure ({prf:ref}`def-valid-noise-measure`) satisfying the axioms. **This is the setting assumed by the neural network framework in `02_neural_nets.md`**.
*   **Discrete Example (Graphs):** A finite or countably infinite graph with a graph metric. A random walk on the graph could serve as the noise measure, provided its properties can be shown to satisfy the required axioms.
:::

### 1.5 Hierarchy of Parameters

:::{admonition} Understanding the Parameter Hierarchy
:class: note
Think of this like designing a video game: First you create the world (environmental parameters) - the terrain, rules of physics, boundaries. Then you tune the gameplay mechanics (algorithmic parameters) - how fast characters move, how they interact, what strategies work best. The environment sets the stage; the algorithm brings it to life.
:::

The system is defined by a hierarchy of choices and parameters that affect the swarm update operator. We group them to clarify their roles and dependencies:

*   **A. Foundational & Environmental Parameters:** Properties of the fixed environment and state space.
*   **B. Core Algorithmic Parameters:** The primary tunable values that control the algorithm's execution.

---

#### A. Foundational & Environmental Parameters

:::{tip}
These parameters are like the "laws of physics" for your swarm - they define what's possible and what's forbidden. Once set, they don't change during a run. They determine the geometry of the search space, what constitutes success (rewards), and how distances are measured.
:::

These values define the static "universe" in which the algorithm operates.

*   **Number of Walkers ($N$):** The total number of walkers ({prf:ref}`def-walker`) in the swarm, which corresponds to the fixed length of the swarm tuple $\mathcal{S}$ (as defined in {prf:ref}`def-swarm-and-state-space`). This number is fixed for the entire run. Must be an integer $N \ge 2$.
*   **Geometric and Reward Structure:** The algorithm is situated within a specific environment defined by:
    1.  A **State Space** $(\mathcal{X}, d_{\mathcal{X}}, \mu_{\mathcal{X}})$ that is a **Valid State Space** ({prf:ref}`def-valid-state-space`) as defined in Section 1.4. This is a Polish metric space equipped with a reference measure that satisfies the foundational axioms of the framework.
    2.  A **Valid Domain ({prf:ref}`def-valid-state-space`)** $\mathcal{X}_{\mathrm{valid}} \subset \mathcal{X}$ with **$C^1$ boundary** $\partial \mathcal{X}_{\mathrm{valid}}$.
    3.  A **Reward Function** $R: \mathcal{X} \to \mathbb{R}$.
    4.  An **Algorithmic Space** $\mathcal{Y} \subset \mathbb{R}^m$ equipped with the Euclidean metric $d_{\mathcal{Y}}(y,y') = \|y-y'\|_2$. The ambient dimension $m \in \mathbb{N}$ is fixed, and the reference measure on $\mathcal{Y}$ is the $m$-dimensional **Lebesgue measure** $\lambda_m$ (restricted to $\mathcal{Y}$). We assume $\mathcal{Y}$ is **bounded**, so $D_{\mathcal{Y}} := \operatorname{diam}_{d_{\mathcal{Y}}}(\mathcal{Y}) < \infty$.
    5.  A **Projection Map** $\varphi: \mathcal{X} \to \mathcal{Y}$ that is **Lipschitz continuous** with constant $L_{\varphi}$; i.e., $d_{\mathcal{Y}}(\varphi(x),\varphi(x')) \le L_{\varphi}\, d_{\mathcal{X}}(x,x')$ for all $x,x' \in \mathcal{X}$.

    These structures, formally defined in subsequent sections, give rise to a set of fixed **environmental constants**, including the reward bound ($R_{\max}$), reward Lipschitz constant ($L_R$), the diameter of the projected valid domain ($D_{\mathrm{valid}} := \operatorname{diam}_{d_{\mathcal{Y}}}(\varphi(\mathcal{X}_{\mathrm{valid}}))$), the diameter of the entire algorithmic space ($D_{\mathcal{Y}} := \operatorname{diam}_{d_{\mathcal{Y}}}(\mathcal{Y})$), and the **Projection Map Lipschitz Constant ($L_{\varphi}$)**.

:::{prf:assumption} Ambient Euclidean Structure and Reference Measures
:label: def-ambient-euclidean

- The spaces $\mathcal{X} \subset \mathbb{R}^d$ and $\mathcal{Y} \subset \mathbb{R}^m$ are finite-dimensional Euclidean domains with Lebesgue reference measures $\lambda_d$ and $\lambda_m$.
- All linear-algebraic objects (means, variances, covariances) and kernel densities are defined with respect to these Euclidean structures and Lebesgue measures. In particular, KDE normalizations use the standard Euclidean constants (e.g., $\int \exp(-\|y\|_2^2/(2\sigma^2))\,dy = (2\pi\sigma^2)^{m/2}$).
- The ambient dimensions $d$ and $m$ are fixed throughout.

This assumption provides the foundational Euclidean structure used throughout the framework. Referenced by {prf:ref}`02_euclidean_gas` for axiom-by-axiom validation of the Euclidean Gas implementation.
:::

:::{prf:definition} Reference Noise and Kernel Families
:label: def-reference-measures

- **Perturbation kernels on $\mathcal{X}$ (dimension $d$):**
  - Gaussian: $\xi \sim \mathcal{N}(0, \sigma^2 I_d)$ so that $\mathbb{E}[\|\xi\|_2^2] = d\,\sigma^2$.
  - Uniform ball: $\xi$ uniform on $B_d(0,\sigma)$ with density $1/\lambda_d(B_d(0,\sigma))$.
- **Cloning kernels on $\mathcal{X}$:** analogously parameterized by $\delta>0$ (e.g., $\mathcal{N}(0, \delta^2 I_d)$ or uniform on $B_d(0,\delta)$).
- **Smoothing kernels on $\mathcal{Y}$ (dimension $m$):**
  - Gaussian: $K_\sigma(y) = (2\pi\sigma^2)^{-m/2} \exp(-\|y\|_2^2/(2\sigma^2))$.
  - Uniform-ball: $K_\sigma(y) = 1/\lambda_m(B_m(0,\sigma))$ for $y \in B_m(0,\sigma)$ and $0$ otherwise.
:::

#### B. Core Algorithmic Parameters

:::{important}
These are your "control knobs" - the parameters you can adjust to change how the swarm behaves. Think of $\alpha$ and $\beta$ as balancing exploration vs exploitation, noise parameters as controlling randomness, and regularization parameters as preventing mathematical disasters (like division by zero).
:::

These are the primary tunable parameters that dictate the dynamic behavior of the swarm.

*   **Dynamics Weights:**
    *   **$\alpha$ (Exploitation Weight):** Controls the influence of the reward signal. $\alpha \in [0, \infty)$.
    *   **$\beta$ (Exploitation Weight):** Controls the influence of the diversity signal. $\beta \in [0, \infty)$.
*   **Noise Scales:**
    *   **$\sigma$ (Perturbation Noise):** The scale of random walks during the perturbation step. $\sigma > 0$.
    *   **$\delta$ (Cloning Noise):** The scale of displacement for newly created walker ({prf:ref}`def-walker`)s. $\delta > 0$.
*   **Clone Threshold Scale:**
    *   **$p_{\max}$ (Clone Threshold Scale):** A positive constant defining the upper bound of the randomly sampled cloning threshold. It controls the baseline responsiveness of the cloning process. Must satisfy $p_{\max} > 0$.
*   **Aggregation Operators:** The choice of statistical aggregation methods used to process raw reward and distance values. These choices are **critical stability-defining parameters**, as their respective Lipschitz continuity properties directly influence the overall continuity of the swarm update operator. The user of this framework is responsible for selecting operators whose moment functions (mean and variance) have known and acceptable continuity bounds.
    *   **Reward Aggregation Operator ($R_{agg}$):** The operator used to create the swarm's collective reward measure. Its moment function Lipschitz constants, $L_{\mu,r}$ and $L_{\sigma',r}$, are key inputs to the global stability analysis.
    *   **Distance Aggregation Operator ($M_D$):** The operator used to create the swarm's collective distance measure . Its moment function Lipschitz constants, $L_{\mu,d}$ and $L_{\sigma',d}$, are key inputs to the global stability analysis.
*   **Cloning and Rescale Regulation:**
    *   **$\eta$ (Rescale Lower Bound):** Ensures the rescaled value components are bounded away from zero. Must satisfy $\eta \ge \eta_{\min} > 0$ for a fixed floor $\eta_{\min}$.
    *   **$\varepsilon_{\text{std}}$ (Standardization Regularizer):** A small constant to prevent division by zero when calculating standard deviation. Must satisfy $\varepsilon_{\text{std}} \ge \varepsilon_{\text{std},\min} > 0$.
*   **$z_{\max}$ (Rescale Saturation Threshold):** A positive constant defining the upper saturation limit for standardized scores before they are transformed by the rescale function. Must satisfy $z_{\max} > 1$ (cf. Section 8.2.2.1).
*   **Variance Regularization:**
*   **$\kappa_{\text{var,min}}$ (Variance Floor Threshold):** A small positive constant that defines the threshold below which the empirical variance is considered degenerate. This parameter is critical for guaranteeing a uniform Lipschitz control on the smoothed denominator, which underpins the Hölder continuity of the standardization operator ({prf:ref}`def-standardization-operator-n-dimensional`) ({prf:ref}`def-standardization-operator-n-dimensional`) ({prf:ref}`def-standardization-operator-n-dimensional`). Must satisfy $\kappa_{\text{var,min}} > 0$.
*   **$\varepsilon_{\text{clone}}$ (Cloning Denominator Regularizer):** A regularizer in the cloning score formula, whose value is critically coupled to the dynamics weights, the rescale lower bound, and the clone threshold scale to ensure the algorithm's viability.
    *   **Global Constraint:** The parameters $\varepsilon_{\text{clone}}, \eta, \alpha, \beta, p_{\max}$ are not independent. They must collectively satisfy the following inequality for the entire run:

$$
\boxed{\varepsilon_{\text{clone}} \cdot p_{\max} < \eta^{\alpha+\beta}}

$$

:::{admonition} The Revival Guarantee: Why This Constraint Matters
:class: attention
:open:
This seemingly abstract inequality ensures something profound: **dead walker ({prf:ref}`def-walker`)s can always be revived** as long as at least one walker remains alive.

Here's the intuition:
- Left side ($\varepsilon_{\text{clone}} \cdot p_{\max}$): Represents the "difficulty" of revival
- Right side ($\eta^{\alpha+\beta}$): Represents the "strength" of alive walkers to help

When the inequality holds, even the "weakest" alive walker is strong enough to guarantee revival of any dead walker. This prevents the swarm from slowly dying off one walker at a time - a critical survival mechanism!
:::

    *   **Justification:** This constraint formally engineers a revival mechanism for *individual* dead walkers, provided the swarm remains alive ($|\mathcal{A}_t| \ge 1$).

:::{admonition} The Mathematical Mechanics of Revival
:class: dropdown note
Here's how the revival mechanism works step by step:

1. **Random Threshold**: The system generates a random cloning threshold $T_{\text{clone}} \sim \text{Uniform}(0, p_{\max})$
2. **Walker Score**: Each dead walker gets a score $S_i$ based on its fitness potential relative to alive companions
3. **Revival Condition**: If $S_i > T_{\text{clone}}$, the walker is revived ("cloned" from a successful position)

**The Guarantee**: Our constraint ensures that for ANY dead walker, $S_i > p_{\max}$ (the maximum possible threshold). This means revival happens with probability 1, regardless of the random draw.

**Why it works**:
- Alive walkers have fitness potential ≥ $\eta^{\alpha+\beta}$ (by construction)
- Dead walkers "borrow" this potential: $S_i \ge \eta^{\alpha+\beta} / \varepsilon_{\text{clone}}$
- Our constraint ensures: $\eta^{\alpha+\beta} / \varepsilon_{\text{clone}} > p_{\max}$
- Therefore: $S_i > p_{\max} \ge T_{\text{clone}}$ → guaranteed revival!
:::

The cloning decision is made by comparing a walker's score, $S_i$, to a random threshold, $T_{\text{clone}} \sim \text{Uniform}(0, p_{\max})$. For revival to be guaranteed, the dead walker's score must be greater than any possible threshold that can be sampled, i.e., $S_i > p_{\max}$. The fitness potential of any alive walker is strictly bounded below by $\eta^{\alpha+\beta}$, a property which follows directly from the construction of the rescaled potential. For a dead walker $i \in \mathcal{D}_t$, its potential is 0. Since $|\mathcal{A}_t| \ge 1$, it can select a companion $c(i) \in \mathcal{A}_t$. Its cloning score is guaranteed to have a lower bound of $S_i \ge V_{\text{fit},c(i)} / \varepsilon_{\text{clone}} > \eta^{\alpha+\beta} / \varepsilon_{\text{clone}}$. The global constraint is constructed by enforcing that this lower bound on the score is greater than the upper bound of the random threshold: $(\eta^{\alpha+\beta}) / \varepsilon_{\text{clone}} > p_{\max}$. This ensures that $S_i > T_{\text{clone}}$ for any outcome of the random threshold, forcing a "Clone" action with probability 1. This prevents the swarm from collapsing due to gradual attrition of individual walkers. It isolates the risk of swarm failure to the single, catastrophic event where all walkers simultaneously move to invalid states in one timestep, making the analysis of that specific event the primary concern for ensuring long-term swarm viability.
    *   A minimal floor $\varepsilon_{\text{clone}} \ge \varepsilon_{\text{clone},\min} > 0$ must also be respected.

### 1.6 N-Particle Displacement Metric

:::{note}
Think of this metric as a way to measure "how much has the swarm changed overall?" It's like asking: if we compare two swarm snapshots, what's the total difference between them? This includes both where walkers moved (position changes) and whether they died or came alive (status changes).
:::

The N-Particle Displacement function $d_{\text{Disp},\mathcal{Y}}$ is a pseudometric on the space of swarms $\Sigma_N$ that combines the kinematic displacement of walkers in the algorithmic space with the change in their survival statuses. It induces a true metric on the Kolmogorov quotient $(\overline{\Sigma}_N,\overline d_{\text{Disp},\mathcal{Y}})$ defined in §1.5.1. It is parameterized by a positive constant, $\lambda_{\mathrm{status}} > 0$, which sets the relative cost of a status change versus positional change.

:::{tip}
The parameter $\lambda_{\text{status}}$ is like a "conversion rate" between physical movement and life/death changes. A high $\lambda_{\text{status}}$ means that a walker dying or coming alive is considered much more significant than a walker simply moving. This reflects the reality that life/death transitions are often more impactful than gradual position changes.
:::

:::{prf:definition} N-Particle Displacement Pseudometric ($d_{\text{Disp},\mathcal{Y}}$)
:label: def-n-particle-displacement-metric

For any two swarms, $\mathcal{S}_1$ and $\mathcal{S}_2$, define the (pseudo)metric by

$$
d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1, \mathcal{S}_2)
:= \Bigg( \frac{1}{N} \sum_{i=1}^N d_{\mathcal{Y}}\!\big(\varphi(x_{1,i}), \varphi(x_{2,i})\big)^2
\;+
\; \frac{\lambda_{\mathrm{status}}}{N} \sum_{i=1}^N (s_{1,i} - s_{2,i})^2 \Bigg)^{\!1/2}.

$$

For algebraic convenience we will frequently write and bound its square,

$$
d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1, \mathcal{S}_2)^2
= \frac{1}{N} \sum_{i=1}^N d_{\mathcal{Y}}(\varphi(x_{1,i}), \varphi(x_{2,i}))^2 + \frac{\lambda_{\mathrm{status}}}{N} \sum_{i=1}^N (s_{1,i} - s_{2,i})^2.

$$

:::{admonition} Breaking Down the Formula
:class: note
:open:
This formula has two parts:

1. **Position Changes** (first term): For each walker ({prf:ref}`def-walker`) $i$, measure how far it moved in the algorithmic space, square it, then average over all walkers.

2. **Status Changes** (second term): For each walker ({prf:ref}`def-walker`) $i$, check if its status changed (alive ↔ dead). Since status is 0 or 1, $(s_{1,i} - s_{2,i})^2$ equals 1 if status changed, 0 if unchanged. Sum these up and weight by $\lambda_{\text{status}}$.

The $\frac{1}{N}$ factors normalize by swarm ({prf:ref}`def-swarm-and-state-space`) size, so larger swarms don't automatically have larger distances.
:::

#### 1.6.1 Metric identification (Kolmogorov quotient)

:::{prf:definition} Metric quotient of $(\Sigma_N, d_{\text{Disp},\mathcal{Y}})$
:label: def-metric-quotient
Define the equivalence relation $\mathcal{S}_1\sim\mathcal{S}_2$ iff $d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1,\mathcal{S}_2)=0$ ({prf:ref}`def-n-particle-displacement-metric`). The **metric identification** (Kolmogorov quotient ({prf:ref}`def-metric-quotient`)) is $\overline{\Sigma}_N := \Sigma_N/\!\sim$ with metric

$$
\overline d_{\text{Disp},\mathcal{Y}}\big([\mathcal{S}_1],[\mathcal{S}_2]\big):= d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1,\mathcal{S}_2),

$$

which is well‑defined and is a true metric.
:::

We carry out optimal‑transport and Wasserstein‑type arguments on $(\overline{\Sigma}_N,\overline d_{\text{Disp},\mathcal{Y}})$. All bounds stated for $d_{\text{Disp},\mathcal{Y}}$ descend to the quotient.

#### 1.6.2 Borel image and completion of the working space

:::{prf:lemma} Borel image of the projected swarm ({prf:ref}`def-swarm-and-state-space`) space
:label: lem-borel-image-of-the-projected-swarm-space
The swarm space ({prf:ref}`def-swarm-and-state-space`) equipped with the projection map ({prf:ref}`def-algorithmic-space-generic`) has the following property:

Let $(\mathcal X,d_{\mathcal X})$ be Polish and $\varphi:\mathcal X\to\mathcal Y$ continuous. If $\mathcal X$ is $\sigma$‑compact and $\Sigma_N\subset(\mathcal X\times\{0,1\})^N$ is Borel, then the projected image

$$
\widehat{\Phi}(\Sigma_N):=\{((\varphi(x_i),s_i))_{i=1}^N : ((x_i,s_i))\in\Sigma_N\}\subset (\mathcal Y\times\{0,1\})^N

$$

is Borel (indeed, contained in $(\varphi(\mathcal X)\times\{0,1\})^N$ with $\varphi(\mathcal X)$ Borel).
:::

:::{prf:proof}
:label: proof-lem-borel-image-of-the-projected-swarm-space
Write $\mathcal X=\bigcup_m K_m$ with $K_m$ compact. Then $\varphi(\mathcal X)=\bigcup_m \varphi(K_m)$ is $F_\sigma$, hence Borel. Products and intersections with Borel sets are Borel; the status constraints are Borel in $\{0,1\}^N$. Hence the claim.

**Q.E.D.**
:::

:::{prf:remark}
:label: rem-closure-cemetery
Following {prf:ref}`lem-borel-image-of-the-projected-swarm ({prf:ref}`def-swarm-and-state-space`)-space`, if $\widehat{\Phi}(\Sigma_N)$ is not closed, replacing it by its closure in $(\mathcal Y\times\{0,1\})^N$ yields a closed (hence complete) subspace. All probability measures considered are supported on $\widehat{\Phi}(\Sigma_N)$, and optimal couplings for costs continuous in $D$ concentrate on the product of supports, so no generality is lost by completing.
:::

### 1.7. Formal Components of Swarm ({prf:ref}`def-swarm-and-state-space`) Displacement
#### 1.7.1 Polishness and $W_2$ well‑posedness (on the quotient)

:::{prf:lemma} Polishness of the quotient state space and $W_2$
:label: lem-polishness-and-w2

If $(\mathcal{Y}, d_{\mathcal{Y}})$ is Polish and $N<\infty$, then the Kolmogorov quotient ({prf:ref}`def-metric-quotient`) $(\overline{\Sigma}_N, \overline d_{\text{Disp},\mathcal{Y}})$ induced by the displacement pseudometric ({prf:ref}`def-n-particle-displacement-metric`) is Polish. Consequently, $W_2$ on $\mathcal{P}(\overline{\Sigma}_N)$ is well‑posed and finite on measures with finite second moment, which holds automatically under the Axiom of Bounded Algorithmic Diameter ({prf:ref}`axiom-bounded-algorithmic-diameter`).
:::

::{prf:proof}
Finite products of Polish spaces are Polish, so $(\mathcal{Y}\times\{0,1\})^N$ endowed with the product topology is Polish. The pseudometric $d_{\text{Disp},\mathcal{Y}}$ ({prf:ref}`def-n-particle-displacement-metric`) is continuous, hence the zero-distance equivalence relation $x\sim y \iff d_{\text{Disp},\mathcal{Y}}(x,y)=0$ is closed. By a standard result in descriptive set theory (see, e.g., Kechris, *Classical Descriptive Set Theory*, Theorem 5.5), the metric quotient ({prf:ref}`def-metric-quotient`) of a Polish space by a closed equivalence relation is again Polish when endowed with the induced metric $\overline d_{\text{Disp},\mathcal{Y}}$. Under the Axiom of Bounded Algorithmic Diameter ({prf:ref}`axiom-bounded-algorithmic-diameter`), $\overline d_{\text{Disp},\mathcal{Y}}\le D_{\mathcal{Y}}+\sqrt{\lambda_{\mathrm{status}}}$, guaranteeing finite second moments. The classical theory of $W_2$ on Polish metric spaces (e.g., Santambrogio, *Optimal Transport for Applied Mathematicians*, §5) therefore applies.

**Q.E.D.**
:::

:::{admonition} Why Decompose Displacement?
:class: important
By breaking displacement into position and status components, we can analyze each type of change separately. This is crucial for understanding stability: maybe the swarm ({prf:ref}`def-swarm-and-state-space`) is stable against small position changes but sensitive to status changes (life/death events), or vice versa. This decomposition allows us to pinpoint exactly where instabilities come from.
:::

This section formally defines the two components of swarm ({prf:ref}`def-swarm-and-state-space`) displacement that will be used as inputs to the generalized continuity axioms.

:::{prf:definition} Components of Swarm Displacement
:label: def-displacement-components

For any two swarms $\mathcal{S}_1$ and $\mathcal{S}_2$ ({prf:ref}`def-swarm-and-state-space`), their total displacement ({prf:ref}`def-n-particle-displacement-metric`) is decomposed into two fundamental components:

1.  **The Squared Positional Displacement ($\Delta_{\text{pos}}^2$):** The sum of squared distances between corresponding walker ({prf:ref}`def-walker`)s in the algorithmic space.

$$
\Delta_{\text{pos}}^2(\mathcal{S}_1, \mathcal{S}_2) := \sum_{i=1}^N d_{\mathcal{Y}}(\varphi(x_{1,i}), \varphi(x_{2,i}))^2

$$

:::{hint}
Why square the distances? Squaring has three benefits: (1) It makes all contributions positive, (2) It emphasizes larger movements (a walker ({prf:ref}`def-walker`) moving distance 2 contributes 4 times more than one moving distance 1), and (3) It creates the mathematical structure needed for the continuity proofs that follow.
:::

2.  **The Total Status Change ($n_c$):** The number of walker ({prf:ref}`def-walker`)s whose survival status changes between the two swarms. This is equivalent to the squared L2-norm of the difference between the status vectors.

$$
n_c(\mathcal{S}_1, \mathcal{S}_2) := \sum_{i=1}^N (s_{1,i} - s_{2,i})^2

$$

:::{tip}
This formula cleverly counts status changes: since $s_i \in \{0,1\}$, we have $(s_{1,i} - s_{2,i})^2 = 1$ if walker ({prf:ref}`def-walker`) $i$ changed status (alive to dead or vice versa), and $(s_{1,i} - s_{2,i})^2 = 0$ if it kept the same status. So $n_c$ simply counts how many walkers changed their life/death status.
:::

The **N-Particle Displacement Metric ({prf:ref}`def-n-particle-displacement-metric`)** defined in Section 1.5 is a specific weighted average of these components: $d_{\text{Disp},\mathcal{Y}}^2 = \frac{1}{N}\Delta_{\text{pos}}^2 + \frac{\lambda_{\mathrm{status}}}{N}n_c$. The generalized continuity framework will use $\Delta_{\text{pos}}^2$ and $n_c$ as direct inputs to provide a more detailed analysis of error propagation.
:::

## 3. Axiomatic Foundations: A Parametric Debugging Framework

:::{admonition} What Are Axiomatic Foundations?
:class: important
:open:
Think of axioms as the "assumptions we're willing to make" about our system. Just as Euclidean geometry starts with axioms like "two points determine a line," our swarm algorithm needs mathematical assumptions about the environment and operators.

**The Brilliant Insight**: Instead of just stating these assumptions, we turn them into measurable parameters. This lets us:
1. **Diagnose problems**: If the algorithm fails, check which axioms are violated
2. **Predict behavior**: The parameter values tell us exactly what to expect
3. **Debug systematically**: Each parameter points to specific components that might need tuning

It's like having a mathematical "health check" for your swarm!
:::

This section consolidates all fundamental assumptions required for the analytical framework to be sound. We reframe these assumptions as a set of user-provided **Axiomatic Parameters**. Each parameter quantifies the system's potential deviation from an ideal, well-behaved condition. The user of this framework is responsible for selecting an environment, operators, and parameters that satisfy the following axioms and for providing the corresponding axiomatic parameter values. These values are critical inputs for diagnosing and debugging the swarm's behavior, as they directly control the stability and convergence guarantees of the system.

#### Assumption A (In‑Step Independence)

:::{prf:axiom} Conditional product structure within a step
:label: axiom-instep-independence

Fix a time $t$ and swarm ({prf:ref}`def-swarm-and-state-space`) state $\mathcal S_t$. For each walker ({prf:ref}`def-walker`) $i\in\{1,\dots,N\}$, let

$$
X_i \;:=\;\big(U_i^{\mathrm{comp}},\,U_i^{\mathrm{pert}},\,U_i^{\mathrm{status}},\,U_i^{\mathrm{clone}}\big)

$$

be the collection of random inputs used by walker ({prf:ref}`def-walker`) $i$ during the next update (companion selection, perturbation noise, status/death draw, cloning/parent draw). **Conditional on $\mathcal S_t$, the vectors $X_1,\dots,X_N$ are independent**, and the components inside each $X_i$ are mutually independent. Companion/parent indices are sampled **with replacement** from their per‑walker categorical distributions. No shared random variable is used across different walkers in the same update.
:::

:::{admonition} Implementation note: Independent PRNG streams
:class: note
Use counter‑based PRNGs (e.g., Random123 Philox/Threefry) to derive independent, reproducible per‑walker ({prf:ref}`def-walker`) streams keyed by `(global_seed, t, i, stage)`. This prevents accidental sharing of randomness across walkers and enforces Assumption A in practice.
:::

:::{note}
Why this matters: Many concentration tools (e.g., McDiarmid’s inequality) require independence of the inputs $X_1,\dots,X_N$ to a functional of the step (such as the average squared displacement). This assumption pins down the intended probabilistic structure within a single update while leaving cross‑time coupling (for synchronous comparisons) available as a proof device.
:::

### 2.1 Viability Axioms: Parameters of Survival

:::{admonition} Survival First, Optimization Second
:class: note
The most important thing for any swarm ({prf:ref}`def-swarm-and-state-space`) algorithm is simply staying alive! These axioms ensure that the swarm doesn't gradually die off or suddenly collapse. Think of them as the "life support systems" - they must work properly before we can worry about finding optimal solutions.
:::

These axioms govern the most fundamental condition: the swarm ({prf:ref}`def-swarm-and-state-space`)'s ability to avoid collapse.

#### 2.1.1 Axiom of Guaranteed Revival

:::{tip}
Imagine a hospital emergency room: as long as there's one doctor alive, they can revive any "flatlined" patient with 100% certainty. This axiom ensures that individual walker ({prf:ref}`def-walker`) deaths never accumulate into swarm extinction - there's always a resurrection mechanism available.
:::

The framework is designed to prevent gradual swarm death from the attrition of individual walker ({prf:ref}`def-walker`)s. This is enforced by a constraint that guarantees any dead walker (in an otherwise alive swarm) has a 100% chance of being revived. The user must provide a parameter that measures the robustness of this mechanism under the stochastic threshold cloning model.

:::{admonition} The Revival Score Ratio: A Critical Health Metric
:class: attention
The parameter $\kappa_{\text{revival}} = \frac{\eta^{\alpha+\beta}}{\varepsilon_{\text{clone}} \cdot p_{\max}}$ is like a "revival strength indicator."

- **Numerator** ($\eta^{\alpha+\beta}$): The "helping power" of alive walker ({prf:ref}`def-walker`)s
- **Denominator** ($\varepsilon_{\text{clone}} \cdot p_{\max}$): The "difficulty" of revival

When $\kappa_{\text{revival}} > 1$, help overpowers difficulty → guaranteed revival!
When $\kappa_{\text{revival}} \leq 1$, the system is in danger → individual deaths can become permanent.
:::

:::{prf:axiom} Axiom of Guaranteed Revival
:label: axiom-guaranteed-revival

*   **Core Assumption:** The cloning score generated by a dead walker ({prf:ref}`def-walker`) ({prf:ref}`def-alive-dead-sets`) must be guaranteed to exceed the maximum possible random threshold, $p_{\max}$.
*   **Axiomatic Parameter ($\kappa_{\text{revival}}$ - The Revival Score Ratio):** The user must provide the value of the revival score ratio, computed from their chosen parameters:

$$
\kappa_{\text{revival}} := \frac{\eta^{\alpha+\beta}}{\varepsilon_{\text{clone}} \cdot p_{\max}}

$$

*   **Condition:** For the axiom to be satisfied, the user must ensure **$\kappa_{\text{revival}} > 1$**.
*   **Failure Mode Analysis:** If **$\kappa_{\text{revival}}$ ≤ 1**, the axiom is violated. A dead walker ({prf:ref}`def-walker`)'s cloning score is no longer guaranteed to be greater than $p_{\max}$. This means there is a non-zero probability that the sampled threshold $T_{\text{clone}}$ will be larger than the walker's score, causing the revival to fail. This disables the guaranteed revival mechanism, meaning individual walker deaths can be permanent, leading to swarm ({prf:ref}`def-swarm-and-state-space`) collapse through gradual attrition. This parameter reveals a critical trade-off: increasing the clone threshold scale $p_{\max}$ to make cloning more responsive simultaneously makes it harder to satisfy the revival condition, thus increasing the risk of swarm attrition.
:::

:::{prf:theorem} Almost‑sure revival under the global constraint
:label: thm-revival-guarantee
Assume the global constraint $\varepsilon_{\text{clone}}\,p_{\max} < \eta^{\alpha+\beta}$ from the Axiom of Guaranteed Revival ({prf:ref}`axiom-guaranteed-revival`). Let $\mathcal S$ be any swarm ({prf:ref}`def-swarm-and-state-space`) with at least one alive walker ({prf:ref}`def-walker`) ($|\mathcal A(\mathcal S)|\ge 1$, see {prf:ref}`def-alive-dead-sets`) and let $i\in\mathcal D(\mathcal S)$ be dead. Then, under the cloning rule with threshold $T_{\text{clone}}\sim\mathrm{Unif}(0,p_{\max})$ and a per‑dead‑walker score $S_i$ computed from an alive companion as in §16.1, we have

$$
\mathbb P\big[\text{$i$ is revived in the cloning stage}\big] \;=\;1.

$$

In particular, $S_i > p_{\max}$ surely, hence $S_i > T_{\text{clone}}$ for every threshold realization. The conclusion holds also when $|\mathcal A(\mathcal S)|=1$ (single‑survivor case) since the companion selection measure ({prf:ref}`def-companion-selection-measure`) assigns the unique alive index to every dead walker ({prf:ref}`def-walker`).

This revival guarantee is applied in {prf:ref}`02_euclidean_gas` to verify the Euclidean Gas satisfies the viability axioms.
:::
```{admonition} k=1 edge case
:class: note
This mean‑square continuity result is for the $k\ge 2$ regime. The $k=1$ discontinuity is handled by the single‑survivor revival mechanism in §16, after which analysis resumes with $k\ge 2$.
```
:::{prf:proof}
:label: proof-thm-revival-guarantee
Let $j\in\mathcal A(\mathcal S)$ be any alive companion. By construction of the fitness potential with rescale floor $\eta$ and weights $(\alpha,\beta)$, we have $V_{\text{fit},j} \ge \eta^{\alpha+\beta}$. The cloning score of a dead walker ({prf:ref}`def-walker`) $i$ satisfies the lower bound

$$
S_i \;\ge\; \frac{V_{\text{fit},j}}{\varepsilon_{\text{clone}}} \;\ge\; \frac{\eta^{\alpha+\beta}}{\varepsilon_{\text{clone}}}.

$$

By the stated constraint, $\eta^{\alpha+\beta}/\varepsilon_{\text{clone}} > p_{\max}$, hence deterministically $S_i>p_{\max}$. Since $T_{\text{clone}}\in[0,p_{\max}]$, we have $S_i>T_{\text{clone}}$ for every threshold draw, so $i$ is cloned with probability one. When $|\mathcal A(\mathcal S)|=1$, the companion of every dead walker is the unique alive index by {prf:ref}`def-companion-selection-measure`, and the same bound applies. This proves the claim.

Q.E.D.
:::

#### 2.1.2 Axiom of Boundary Regularity ({prf:ref}`axiom-boundary-regularity`)

:::{admonition} The Catastrophic Collapse Problem
:class: caution
:open:
While individual walker ({prf:ref}`def-walker`) deaths can be handled by revival, there's one terrifying scenario: **all walkers dying at once**. This happens when the entire swarm wanders into a "forbidden zone" where no positions are valid.

The question is: how "sharp" are these boundaries? If they're like cliff edges (sudden death), small navigation errors can cause total extinction. If they're like gentle hills (gradual danger increase), the swarm ({prf:ref}`def-swarm-and-state-space`) can "sense" danger and back away.

This axiom quantifies boundary sharpness through the death probability's sensitivity to swarm ({prf:ref}`def-swarm-and-state-space`) configuration changes.
:::

The risk of swarm collapse is primarily the single catastrophic event where all walker ({prf:ref}`def-walker`)s simultaneously become invalid. The stability of this process depends on how erratically the "death probability" for any individual walker changes as a function of the entire swarm's N-particle configuration. The user must quantify this regularity.

:::{note}
**Hölder continuity** is a mathematical way of saying "no sudden jumps." The inequality $|P(s_{\text{out},i}=0 | \mathcal{S}_1) - P(s_{\text{out},i}=0 | \mathcal{S}_2)| \le L_{\text{death}} \cdot d^{\alpha_B}$ means:

- Small changes in swarm ({prf:ref}`def-swarm-and-state-space`) configuration ($d$ is small) → small changes in death probability
- The constants $L_{\text{death}}$ and $\alpha_B$ control how "smooth" this relationship is
- Larger $L_{\text{death}}$ = more unpredictable boundaries = higher risk
:::

:::{prf:axiom} Axiom of Boundary Regularity
:label: axiom-boundary-regularity

*   **Core Assumption:** The marginal probability of a single walker ({prf:ref}`def-walker`) becoming invalid after the perturbation and status update stages must be a smooth (Hölder continuous) function of the initial N-particle swarm state ({prf:ref}`def-swarm-and-state-space`). This axiom applies to any valid noise measure ({prf:ref}`def-valid-noise-measure`), including those with state-dependent coupling between walkers.

*   **Axiomatic Parameters:** The user must provide the constants that bound this relationship, derived from their choice of **Noise Measure**, **Valid Domain** ({prf:ref}`def-valid-state-space`), and **Projection Map**:
    1.  **$L_{\text{death}}$ > 0 (The Boundary Instability Factor):** The Hölder constant for the marginal death probability function.
    2.  **$\alpha_B$ ∈ (0, 1] (The Boundary Smoothing Exponent):** The Hölder exponent.

*   **Condition:** Let $P(s_{\text{out},i}=0 | \mathcal{S})$ be the marginal probability that walker $i$ ({prf:ref}`def-walker`) has a status of 0 after the application of the composed operator $\Psi_{\text{status}} \circ \Psi_{\text{pert}}$ to an initial swarm state $\mathcal{S}$. These constants must satisfy the following inequality for any two swarm states $\mathcal{S}_1, \mathcal{S}_2 \in \Sigma_N$ ({prf:ref}`def-swarm-and-state-space`) and for all walkers $i \in \{1, \dots, N\}$:

$$
|P(s_{\text{out},i}=0 | \mathcal{S}_1) - P(s_{\text{out},i}=0 | \mathcal{S}_2)| \le L_{\text{death}} \cdot d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1, \mathcal{S}_2)^{\alpha_B}

$$

where $d_{\text{Disp},\mathcal{Y}}$ is the N-Particle Displacement Metric ({prf:ref}`def-n-particle-displacement-metric`).

*   **Canonical Bounds:** When the invalid set has finite perimeter and the perturbation kernel ({prf:ref}`def-perturbation-measure`) satisfies the smoothness assumptions below, we may take explicit constants:
    - **Uniform ball kernels.** Section 4.2.3 shows that for $\mathcal P_\sigma(x,\cdot)$ uniform on $B(x,\sigma)$ the death probability is Lipschitz with constant $L_{\text{death}} \le C_d\,\mathrm{Per}(\mathcal X_{\mathrm{invalid}})/\sigma$ and exponent $\alpha_B=1$.
    - **Gaussian/heat kernels.** Section 4.2.4 proves the analogous bound $L_{\text{death}} \le C'_d\,\mathrm{Per}(\mathcal X_{\mathrm{invalid}})/\sigma$ with $\alpha_B=1$ by convolution with the heat kernel.
    - **Projections.** If a nontrivial projection $\varphi$ is used, include the distortion factor from its Lipschitz constant as discussed after these lemmas.

*   **Failure Mode Analysis:** A large **$L_{\text{death}}$** indicates a "sharp" or unpredictable boundary in the N-particle state space. A small change in the overall swarm's configuration (either a small shift in walker ({prf:ref}`def-walker`) positions or a single status change) could lead to a drastic change in a walker's individual survival probability. This makes the swarm's behavior near the boundary highly unstable and risks unexpected, large-scale death events that are not well-correlated with the simple displacement of individual walkers.

:::{warning}
**Red Flag**: If you measure $L_{\text{death}}$ and find it's very large, your environment has dangerous "cliff edges" where small missteps lead to mass casualties. Consider smoothing the boundary (adding buffer zones) or increasing noise to help walker ({prf:ref}`def-walker`)s "probe" dangerous areas more gently.
:::
:::

#### 2.1.3 Axiom of Boundary Smoothness

:::{admonition} Why Boundaries Must Be Smooth
:class: note
Imagine the valid region is like a country, and the boundary is like its border. A "smooth" border (like a gentle curve) has zero area - it's just a line. But a "fractal" border (like a coastline with infinite detail) could have positive area, meaning walker ({prf:ref}`def-walker`)s might get "stuck" exactly on the boundary.

Mathematically, we need the boundary to be a nice, smooth curve so that the probability of landing exactly on it is zero. This keeps our death probability calculations clean and continuous.
:::

:::{prf:axiom} Axiom of Boundary Smoothness
:label: axiom-boundary-smoothness

*   **Core Assumption:** The boundary of the valid domain, $\partial \mathcal{X}_{\mathrm{valid}}$ ({prf:ref}`def-valid-state-space`), must be a $(d-1)$‑dimensional continuously differentiable ($C^1$) submanifold of the $d$‑dimensional state space $\mathcal{X}$.

*   **Rationale:** This is the standard condition in geometric measure theory ensuring the boundary has Lebesgue measure zero in the ambient space. It is a critical prerequisite for proving that $\partial \mathcal{X}_{\mathrm{valid}}$ is a null set for any absolutely continuous perturbation kernel, which is a key step in validating the Axiom of Boundary Regularity ({prf:ref}`axiom-boundary-regularity`).

*   **Framework Application:** This axiom serves as the formal prerequisite for establishing that the integral defining the death probability is a continuous function of the swarm ({prf:ref}`def-swarm-and-state-space`) state, thereby supporting the Axiom of Boundary Regularity ({prf:ref}`axiom-boundary-regularity`).

*   **Failure Mode Analysis:** If the boundary is not a $C^1$ submanifold (e.g., fractal or space‑filling), it may have positive Lebesgue measure. Then the probability of a walker ({prf:ref}`def-walker`) landing exactly on the boundary can be non‑zero, the death‑probability function may fail to be continuous, and the continuity analysis of the swarm ({prf:ref}`def-swarm-and-state-space`) update operator breaks down.

:::

### 2.2 Environmental Axioms: Parameters of the Problem Space

:::{important}
Now we shift from "staying alive" to "being able to learn." These axioms ensure the environment provides enough structure for the swarm ({prf:ref}`def-swarm-and-state-space`) to discover patterns and improve over time. A perfectly flat environment is like trying to learn on a completely uniform landscape - there are no landmarks to guide you!
:::

These axioms quantify the properties of the environment in which the swarm ({prf:ref}`def-swarm-and-state-space`) operates.

#### 2.2.1 Axiom of Environmental Richness

:::{admonition} The "Interesting Environment" Requirement
:class: tip
:open:
Imagine trying to learn navigation in a perfectly flat desert versus a landscape with hills and valleys. In the desert, every direction looks the same - there's no learning signal. Hills and valleys provide gradients that guide you toward better regions.

This axiom ensures your reward landscape isn't "too flat" at any relevant scale. The parameters $r_{\min}$ and $\kappa_{\text{richness}}$ quantify this:
- $r_{\min}$: "At what scale do I expect to find interesting features?"
- $\kappa_{\text{richness}}$: "How much variation is guaranteed at that scale?"

Think of it as ensuring your problem has enough "texture" to learn from!
:::

The algorithm cannot learn if the reward landscape is flat. The user must provide a parameter that guarantees the environment is sufficiently interesting to provide a learning signal.

:::{prf:axiom} Axiom of Environmental Richness
:label: axiom-environmental-richness

*   **Core Assumption:** The reward function ({prf:ref}`def-reward-measurement`) $R$ must not be pathologically flat at a user-defined minimum length scale. The algorithm requires a guaranteed level of reward variation to learn.
*   **Axiomatic Parameters:** The user must provide two parameters that quantify the learnability of the reward landscape:
    1.  **$r_{\min}$ > 0 (The Minimum Richness Scale):** The minimum radius in the algorithmic space ({prf:ref}`def-algorithmic-space-generic`) above which the reward function is guaranteed to exhibit variance. This parameter quantifies the resolution at which the user expects to find a learnable signal.
    2.  **$\kappa_{\text{richness}}$ (The Environmental Richness Floor):** A value that acts as a guaranteed lower bound on the variance of the reward function within any localized region of the projected valid domain ({prf:ref}`def-valid-state-space`) *with a radius greater than or equal to $r_{\min}$*.

*   **Condition:** The user must choose $r_{\min}$ and determine $\kappa_{\text{richness}}$ such that they satisfy the following inequality, which formally links the two parameters:

$$
\kappa_{\text{richness}} \le \inf_{y \in \varphi(\mathcal{X}_{\mathrm{valid}}), r \ge r_{\min}} \left( \text{Var}_{y' \in B(y,r) \cap \varphi(\mathcal{X}_{\mathrm{valid}})} [R_{\mathcal{Y}}(y')] \right)

$$

    The user must then ensure that their chosen scale yields a positive floor: **$\kappa_{\text{richness}}$ > 0**.
*   **Failure Mode Analysis:** If, for a given $r_{\min}$, the resulting **$\kappa_{\text{richness}}$ ≈ 0**, it implies the environment contains large regions of size $r_{\min}$ where the reward is essentially constant. If a swarm ({prf:ref}`def-swarm-and-state-space`)'s spatial extent is smaller than $r_{\min}$, it might perceive the landscape as flat. If the swarm enters a larger, truly flat region, the exploitation component of the fitness potential will have near-zero variance, stalling the learning process and adaptive dynamics. The choice of $r_{\min}$ is therefore a critical parameter that reflects the scale of features in the problem environment.
:::

#### 2.2.2 Axiom of Reward Regularity

:::{note}
While the richness axiom ensures the landscape isn't flat, this axiom ensures it isn't too chaotic. We need rewards to change smoothly - if rewards jump erratically from point to point, the swarm ({prf:ref}`def-swarm-and-state-space`) can't build reliable gradients to follow. The Hölder continuity condition is the mathematical way of saying "no sudden jumps in reward values."
:::

:::{prf:axiom} Axiom of Reward Regularity
:label: axiom-reward-regularity

*   **Core Assumption:** The reward function, when viewed in the algorithmic space ({prf:ref}`def-algorithmic-space-generic`), must be Hölder continuous.

*   **Axiomatic Parameters:** The user must provide the constants that bound the reward function's smoothness:
    1.  **$L_{R,\mathcal{Y}} > 0$ (The Reward Volatility Factor):** The Hölder constant of the reward function in the algorithmic space ({prf:ref}`def-algorithmic-space-generic`).
    2.  **$\alpha_R \in (0, 1]$ (The Reward Smoothing Exponent):** The Hölder exponent for the reward function on $(\mathcal{Y},d_{\mathcal{Y}})$.

*   **Condition:** These constants must satisfy, for any $y_1, y_2 \in \mathcal{Y}$,

$$
|R_{\mathcal{Y}}(y_1) - R_{\mathcal{Y}}(y_2)| \le L_{R,\mathcal{Y}} \cdot d_{\mathcal{Y}}(y_1, y_2)^{\alpha_R}.

$$

*   **Failure Mode Analysis:** A large **$L_{R,\mathcal{Y}}$** signifies a "bumpy" or volatile reward landscape. This can cause the exploitation component of the fitness potential to fluctuate wildly with small movements, making cloning decisions noisy and potentially unstable.

Referenced by {prf:ref}`axiom-projection-compatibility`.
:::

:::{hint}
Think of $L_{R,\mathcal{Y}}$ as a "maximum steepness" parameter. Small values mean rewards change gently; large values allow steeper reward gradients. But even with large $L_{R,\mathcal{Y}}$, the Hölder condition prevents infinite jumps - it puts a mathematical "speed limit" on how fast rewards can change.
:::

#### 2.2.3 Axiom of Bounded Algorithmic Diameter

:::{prf:axiom} Projection compatibility
:label: axiom-projection-compatibility
There exists a function $R_{\mathcal Y}:\varphi(\mathcal X)\to\mathbb R$ such that $R = R_{\mathcal Y}\circ\varphi$ on $\mathcal X$. Equivalently, if $\varphi(x)=\varphi(x')$ then $R(x)=R(x')$.
:::

:::{admonition} Remark
:class: note
The axiom ensures that $R_{\mathcal Y}$ is well‑defined on the image $\varphi(\mathcal X)$. Its regularity on $(\mathcal Y,d_{\mathcal Y})$ is provided by the Axiom of Reward Regularity ({prf:ref}`axiom-reward-regularity`). In special cases (e.g., $\varphi$ injective on $\mathcal X$ or admitting a Lipschitz right‑inverse on $\varphi(\mathcal X)$), one can relate $L_{R,\mathcal Y}$ to $L_R$ and $L_\varphi$; otherwise treat $L_{R,\mathcal Y}$ as an independent parameter fixed by the environment.
:::

:::{prf:axiom} Axiom of Bounded Algorithmic Diameter
:label: axiom-bounded-algorithmic-diameter

- The algorithmic space ({prf:ref}`def-algorithmic-space-generic`) $(\mathcal{Y}, d_{\mathcal{Y}})$ is Polish (complete, separable metric space).
- Its diameter is finite: $D_{\mathcal{Y}} := \operatorname{diam}_{d_{\mathcal{Y}}}(\mathcal{Y}) < \infty$.

These conditions ensure Wasserstein metrics $W_p$ on probability measures over $(\mathcal{Y}, d_{\mathcal{Y}})$ are well‑posed and that all per‑walker ({prf:ref}`def-walker`) squared displacements are bounded by $D_{\mathcal{Y}}^2$.
:::

#### 2.2.4 Axiom of Range‑Respecting Mean (Aggregators)

:::{prf:axiom} Range‑Respecting Mean
:label: axiom-range-respecting-mean
For any finite collection of inputs from walkers ({prf:ref}`def-walker`) in a swarm ({prf:ref}`def-swarm-and-state-space`) $\{v_i\}$, the aggregator’s mean output $\mu$ satisfies

$$
\min_i v_i \;\le\; \mu \;\le\; \max_i v_i.

$$

This property holds for empirical means and is assumed for any user‑chosen mean‑type aggregator in this framework.
:::

### 2.3 Algorithmic & Operator Axioms: Parameters of Dynamic Behavior

:::{prf:definition} Valid Noise Measure
:label: def-valid-noise-measure
A kernel $\mathcal P_\sigma$ (and analogously $\mathcal Q_\delta$) is valid if it is Feller and satisfies:
- Bounded second moment in $\mathcal Y$ with constant $M_{\mathrm{pert}}^2$ (as used in the perturbation continuity bounds);
- Boundary regularity ({prf:ref}`axiom-boundary-regularity`) assumptions required by the status‑continuity theorem (Section 14);
- Non‑degeneracy as stipulated where needed.
This consolidates the standing noise requirements referenced elsewhere in the framework.

Referenced by {prf:ref}`def-valid-state-space`, {prf:ref}`def-perturbation-measure`, and {prf:ref}`def-cloning-measure`.
:::

These axioms concern the user's choices for the internal mechanisms of the algorithm, quantifying their impact on stability and convergence.

#### 2.3.1 Axiom of Sufficient Amplification

The algorithm's dynamics are driven by transforming reward and distance measurements into fitness potential. If this transformation is turned off, the algorithm stalls.

:::{prf:axiom} Axiom of Sufficient Amplification
:label: axiom-sufficient-amplification

*   **Core Assumption:** The dynamics weights must be configured to actively process measurement signals from the reward function ({prf:ref}`def-reward-measurement`).
*   **Axiomatic Parameter ($\kappa_{\text{amplification}}$ - The Amplification Strength):** The user must provide the dynamics weights $\alpha$ and $\beta$, from which the amplification strength is defined as:

$$
\kappa_{\text{amplification}} := \alpha + \beta

$$

*   **Condition:** The user must ensure **$\kappa_{\text{amplification}}$ > 0**.
*   **Failure Mode Analysis:** If **$\kappa_{\text{amplification}}$ = 0**, both $\alpha$ and $\beta$ are zero. The fitness potential $V_i$ becomes $\eta^0 = 1$ for all alive walker ({prf:ref}`def-walker`)s. The cloning score $S_i$ is always zero, meaning no cloning can ever occur. The swarm ({prf:ref}`def-swarm-and-state-space`) becomes a collection of independent, non-interacting random walkers.
:::

#### 2.3.2 Axiom of Non-Degenerate Noise ({prf:ref}`axiom-non-degenerate-noise`)

The swarm ({prf:ref}`def-swarm-and-state-space`) relies on noise to explore the state space and prevent collapsing to a single point.

:::{prf:axiom} Axiom of Non-Degenerate Noise
:label: axiom-non-degenerate-noise

*   **Core Assumption:** The **Perturbation ({prf:ref}`def-perturbation-measure`)** and **Cloning** measures must not be the Dirac delta measure.
*   **Axiomatic Parameters ($\sigma$, $\delta$ - The Noise Scales):** The user provides these parameters directly.
*   **Condition:** The user must ensure **$\sigma > 0$** and **$\delta > 0$**.
*   **Failure Mode Analysis:** If **$\sigma = 0$** and **$\delta = 0$**, the swarm ({prf:ref}`def-swarm-and-state-space`) cannot introduce new positions into the system, leading to a complete loss of exploration and eventual collapse to a few points.
:::

#### 2.3. ({prf:ref}`def-standardization-operator-n-dimensional`)3 Mean-Square Continuity of the Standardization Operator ({prf:ref}`def-standardization-operator-n-dimensional`)

Because the **Raw Value Operator** $V$ (e.g., distance-to-companion) is  ({prf:ref}`def-standardization-operator-n-dimensional`)stochastic, the **N-Dimensional Standardization Operator ({prf:ref}`def-standardization-operator-n-dimensional`)** $z(S)$ is also a stochastic operator. Its output, $z$, is a random variable. Therefore, its continuity must be analyzed in a probabilistic sense. The strongest and most useful form for the subsequent stability analysis is **mean-square continuity**, which bounds the *expected* squared error between the outputs for two different input swarms.

To formalize this analysis, we first define the two fundamental and independent sources of error that contribute to the total mean-square error.

:::{prf:definition} Components of Mean-Square Standardization Error
:label: def-components-mean-square-standardization-error

The total expected squared error of the standardization operator ({prf:ref}`def-standardization-operator-n-dimensional`), $\mathbb{E}[\|\mathbf{z}(\mathcal{S}_1, V, M) - \mathbf{z}(\mathcal{S}_2, V, M)\|_2^2]$, is bounded by the sum of two components for any two swarms $\mathcal{S}_1, \mathcal{S}_2$ ({prf:ref}`def-swarm-and-state-space`):

1.  **The Expected Squared Value Error ($E^2_{V,ms}$):** The error arising from the change in the raw value vector's probability distribution (from $V(\mathcal{S}_1)$ to $V(\mathcal{S}_2)$) while the swarm ({prf:ref}`def-swarm-and-state-space`)'s structure is held fixed at $\mathcal{S}_1$. This component quantifies the propagation of measurement stochasticity.

2.  **The Expected Squared Structural Error ($E^2_{S,ms}$):** The error arising from the change in the swarm ({prf:ref}`def-swarm-and-state-space`)'s structure (from $\mathcal{S}_1$ to $\mathcal{S}_2$) while using a fixed raw value vector sampled from the second swarm's distribution $V(\mathcal{S}_2)$. This component quantifies the operator's sensitivity to walker ({prf:ref}`def-walker`) deaths and revivals ({prf:ref}`def-alive-dead-sets`).
:::

The following theorem, which is a key result of the analysis in Section 11, establishes the asymptotic behavior of these error components.

:::{prf:theorem} Asymptotic Behavior of the Mean-Square Standardization Error
:label: thm-mean-square-standardization-error

The continuity of the **N-Dimensional Standardization Operator ({prf:ref}`def-standardization-operator-n-dimensional`)** $z(\mathcal{S})$ depends on the coupled effects of two distinct error sources ({prf:ref}`def-components-mean-square-standardization-error`) whose expected growth rates are summed.

*   **Core Principle:** The total **expected** squared error in the standardization operator ({prf:ref}`def-standardization-operator-n-dimensional`)'s output, $\mathbb{E}[\| \mathbf{z}_1 - \mathbf{z}_2 \|_2^2]$, is bounded by the sum of the **Expected Squared Value Error** ($E^2_{V,ms}$) and the **Expected Squared Structural Error** ($E^2_{S,ms}$) from {prf:ref}`def-components-mean-square-standardization-error`.

*   **Mathematical Result (General Form):** For a large number of alive ({prf:ref}`def-alive-dead-sets`) walker ({prf:ref}`def-walker`)s, $k_1 = |\mathcal{A}(\mathcal{S}_1)|$, the total expected error has an asymptotic growth rate given by the sum of the growth rates of its two components:

$$
\boxed{
    \mathbb{E}[\| \mathbf{z}_1 - \mathbf{z}_2 \|_2^2] \in O(E_{V,ms}^2(k_1)) + O(E_{S,ms}^2(k_1))
    }

$$

*   **Implications & Failure Modes:** The overall mean-square continuity of the standardization pipeline is governed by the operational regime of the swarm ({prf:ref}`def-swarm-and-state-space`). The analysis reveals a critical distinction between normal operation and catastrophic collapse.
    1.  **Regime 1: Normal Operation (Asymptotically Stable):** Under normal conditions where walker ({prf:ref}`def-walker`) attrition is low and the number of status changes ($n_c$) is small, the structural error term is negligible. The dominant error is the **expected value error**, which, for the benchmark case of an empirical aggregator and distance-to-companion measurement, **is constant with respect to swarm size** ($E^2_{V,ms} \in O(1)$). This is a powerful result, indicating that under stable conditions, the algorithm's average measurement process **does not become noisier as the swarm gets larger**. The primary bottleneck for stability in this regime is not swarm size but the **extreme sensitivity to the regularization parameter**, as all error sources are amplified by a factor of up to **$O(\varepsilon_{\text{std}}^{-6})$**.
    2.  **Regime 2: Catastrophic Collapse (Unstable):** During a catastrophic collapse event where a large fraction of the swarm ({prf:ref}`def-swarm-and-state-space`) dies (e.g., $n_c \propto k_1$), the **expected structural error** term **grows linearly with swarm size** ($E^2_{S,ms} \in O(k_1)$). This confirms that large-scale death events are a fundamental source of instability, and that larger swarms are more vulnerable to continuity breakdown during such events. The choice of a **structurally stable aggregator** ($p_{\text{worst-case}} \le -1/2$) is critical to prevent this expected error from growing even faster.
:::

#### 2.3.4 Axioms for Swarm ({prf:ref}`def-swarm-and-state-space`) Aggregation Operators

For the framework's stability analysis to be sound, any chosen aggregation operator must satisfy the following axioms, and the user must provide the corresponding axiomatic parameters.

:::{prf:axiom} Axiom of Bounded Relative Collapse
:label: axiom-bounded-relative-collapse

*   **Core Assumption:** The scaling analysis for structural error is valid only for transitions that are not catastrophically large relative to the initial swarm ({prf:ref}`def-swarm-and-state-space`) size.
*   **Axiomatic Parameter ($c_{\min}$ - The Relative Collapse Tolerance):** The user must provide a constant $c_{\min} \in (0, 1]$ that defines the minimum fraction of the swarm ({prf:ref}`def-swarm-and-state-space`) that must survive a transition for the structural growth exponent analysis to be considered valid.
*   **Condition:** A transition from a swarm ({prf:ref}`def-swarm-and-state-space`) $S_1$ to $S_2$ is considered **non-catastrophic** if the ratio of alive walker ({prf:ref}`def-walker`)s ({prf:ref}`def-alive-dead-sets`) satisfies:

$$
\frac{|\mathcal{A}(\mathcal{S}_2)|}{|\mathcal{A}(\mathcal{S}_1)|} \ge c_{\min}

$$

*   **Framework Application and Limitations:** All subsequent theorems concerning the asymptotic scaling of structural error are certified to hold only for transitions that satisfy this user-provided condition. This axiom represents a significant limitation on the scope of the structural stability analysis. It implies that the framework's guarantees regarding aggregator scaling properties are valid for assessing stability under operational conditions (i.e., small perturbations and gradual attrition) but are not certified to hold during the very catastrophic collapse events they are intended to help understand. The analysis is therefore most applicable to ensuring the system does not enter such a regime, rather than characterizing its dynamics within it.
:::

:::{prf:axiom} Axiom of Bounded Deviation from Aggregated Variance
:label: axiom-bounded-deviation-variance

*   **Core Assumption:** The sum of squared deviations of the raw input values from the aggregator's computed mean must be controllably related to the variance computed by the aggregator itself. This prevents aggregators from producing statistical moments that are pathologically decoupled from the input data.
*   **Axiomatic Parameter ($\kappa_{\text{var}}$ - The Variance Deviation Factor):** The user must provide a constant $\kappa_{\text{var}} \geq 1$ that bounds this relationship.
*   **Condition:** For any swarm state $S$ ({prf:ref}`def-swarm-and-state-space`) with alive set $A$ ({prf:ref}`def-alive-dead-sets`) and any raw value vector $v_A$, the following must hold:

$$
\sum_{i \in \mathcal{A}} (v_i - \mu(\mathcal{S}, \mathbf{v}_{\mathcal{A}}))^2 \le \kappa_{\text{var}} \cdot |\mathcal{A}| \cdot \text{Var}[M(\mathcal{S}, \mathbf{v}_{\mathcal{A}})]

$$

where $\text{Var}[M]$ is the variance of the aggregator's output measure.

* **Framework Application:** This axiom is critical for deriving a tight continuity bound for the standardization operator's value and structural error components. A smaller $\kappa_{\text{var}}$ indicates a "better-behaved" aggregator.
:::

:::{prf:axiom} Axiom of Bounded Variance Production
:label: axiom-bounded-variance-production

*   **Core Assumption:** The variance of the measure produced by the aggregation operator must itself be bounded by a function of the input value range. This prevents an aggregator from creating arbitrarily large variance from bounded inputs.
*   **Axiomatic Parameter ($\kappa_{\text{range}}$ - The Range-to-Variance Factor):** The user must provide a constant $\kappa_{\text{range}} \geq 0$.
*   **Condition:** For any swarm ({prf:ref}`def-swarm-and-state-space`) $S$ and any value vector $v$ with components bounded by $V_{\max}$, the aggregator $M$ must satisfy:

$$
\text{Var}[M(\mathcal{S}, \mathbf{v})] \le \kappa_{\text{range}} \cdot V_{\max}^2

$$

*   **Framework Application:** This axiom is essential for ensuring the continuity proofs for the standardization pipeline are sound for any valid aggregator. It prevents an uncontrolled amplification of error that would otherwise arise if an aggregator could invent unbounded variance.
:::

### 2.4 Geometric and Activity Axioms

#### 2.4.1 Axiom of Geometric Consistency

This axiom requires the user to quantify how much their chosen noise measure deviates from the "natural" diffusion of the state space, benchmarked by the heat kernel.

:::{prf:axiom} Axiom of Geometric Consistency
:label: axiom-geometric-consistency

*   **Core Assumption:** The algorithmic noise ({prf:ref}`def-valid-noise-measure`) should be unbiased and isotropic, unless intentionally designed otherwise.
*   **Axiomatic Parameters (Practical Proxies):**
    1.  **$\kappa_{\text{drift}}$ (Anomalous Drift):** The maximum magnitude of any local drift introduced by the noise measure:

$$
\kappa_{\text{drift}} := \sup_{x \in \mathcal{X}} \|\mathbb{E}_{x' \sim \mathcal{P}_\sigma(x, \cdot)}[x' - x]\|

$$

    2.  **$\kappa_{\text{anisotropy}}$ (Diffusion Anisotropy):** The maximum condition number of the displacement's covariance matrix:

$$
\kappa_{\text{anisotropy}} := \sup_{x \in \mathcal{X}} \frac{\lambda_{\max}(\text{Cov}_{x' \sim \mathcal{P}_\sigma(x, \cdot)}[x'])}{\lambda_{\min}(\text{Cov}_{x' \sim \mathcal{P}_\sigma(x, \cdot)}[x'])}

$$

*   **Condition:** For ideal geometric consistency, **$\kappa_{\text{drift}}$ = 0** and **$\kappa_{\text{anisotropy}}$ = 1**.
*   **Failure Mode Analysis:** **$\kappa_{\text{drift}}$ > 0** introduces a systematic bias, confounding optimization. **$\kappa_{\text{anisotropy}}$ > 1** causes inefficient or misaligned exploration.
:::

#### 2.4.2 Theorem of Forced Activity

For the algorithm's adaptive and contractive forces to function, a swarm ({prf:ref}`def-swarm-and-state-space`) that is not collapsed to a single point must generate a non-zero probability of cloning. This is the engine of adaptation.

:::{prf:theorem} Theorem of Forced Activity
:label: thm-forced-activity

This theorem demonstrates that {prf:ref}`axiom-guaranteed-revival` ensures all walker ({prf:ref}`def-walker`)s eventually become active during the cloning process.

:::{admonition} The "No Stagnation" Guarantee
:class: important
:open:
This theorem is beautiful because it guarantees the swarm ({prf:ref}`def-swarm-and-state-space`) can never get completely stuck! Here's the intuition:

**The Setup**: If you have:
1. **Spread out walker ({prf:ref}`def-walker`)s** (covering enough space to sense reward differences)
2. **Rich environment** (reward actually varies across space)
3. **Non-zero amplification** (the algorithm pays attention to rewards)
4. **Some noise** (walker ({prf:ref}`def-walker`)s can explore)

**The Conclusion**: Then some cloning MUST happen - the swarm ({prf:ref}`def-swarm-and-state-space`) can't just sit still.

**Why it works**: Spread-out walker ({prf:ref}`def-walker`)s in a rich environment will experience different rewards. This creates fitness differences. With amplification > 0, these differences get magnified into different cloning probabilities. Someone will always be "fit enough" to clone, keeping the swarm active.

Think of it as an "anti-stagnation theorem" - as long as the basic conditions are met, the swarm ({prf:ref}`def-swarm-and-state-space`) is guaranteed to keep exploring and adapting!
:::

*   **Core Principle:** A swarm ({prf:ref}`def-swarm-and-state-space`) that is sufficiently spread out in a sufficiently rich environment will generate a non-zero probability of cloning. This property is an emergent consequence of satisfying the Axiom of Environmental Richness ({prf:ref}`axiom-environmental-richness`), the Axiom of Non-Degenerate Noise ({prf:ref}`axiom-non-degenerate-noise`), and the Axiom of Sufficient Amplification ({prf:ref}`axiom-sufficient-amplification`).
*   **System Property ($p_{\text{clone,min}}$ - The Minimum Average Cloning Probability):** The user is responsible for ensuring their choice of axiomatic parameters ($\kappa_{\text{richness}}$, $r_{\min}$, $\alpha$, $\beta$, etc.) leads to a configuration where, for any "non-degenerate" swarm ({prf:ref}`def-swarm-and-state-space`) state, the expected cloning probability is bounded below by a positive constant.
    *   A swarm is considered **non-degenerate** in this context if its walker ({prf:ref}`def-walker`)s are sufficiently dispersed in the algorithmic space (e.g., spanning a diameter greater than $r_{\min}$) to experience the guaranteed environmental richness $\kappa_{\text{richness}}$, thus generating variance in the fitness potential.

$$
p_{\text{clone,min}} > 0

$$

*   **Condition for Viable Adaptation:** The system configuration must yield **$p_{\text{clone,min}}$ > 0**.
*   **Failure Mode Analysis:** If the system parameters lead to **$p_{\text{clone,min}}$ = 0**, the system can enter states where the contractive force of cloning vanishes, even when the swarm ({prf:ref}`def-swarm-and-state-space`) is not converged. This can occur if the swarm collapses into a region smaller than $r_{\min}$ where the reward landscape appears flat, stalling adaptation and preventing convergence.

:::{warning}
**Stagnation Risk**: When $p_{\text{clone,min}} = 0$, the swarm can enter "dead zones" where everyone looks equally fit (no reward gradients visible), so no one gets cloned. The swarm becomes a collection of independent random walker ({prf:ref}`def-walker`)s, losing its collective intelligence. Always check that your swarm stays spread out enough ($> r_{\min}$) to sense environmental structure!
:::
:::

### 2.5 Summary of Axiomatic Parameters and Key Theorems

This section provides a consolidated reference for the foundational assumptions (Axioms) that the user must satisfy, and the key system-level theorems that emerge from these axioms.

#### 2.5.1 **Summary of Axiomatic Foundations**

| Axiom Name & Section                                                                   | Core Principle                                                                               | Core Assumption                                                                                                                  | Axiomatic Parameters                                                                             | Failure Mode Analysis                                                                                                                                                                                                                                                                                                                                                                                                                          |
|:---------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Viability Axioms**                                                                   | **"The swarm ({prf:ref}`def-swarm-and-state-space`) must survive."**                                                                |                                                                                                                                  |                                                                                                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Axiom of Guaranteed Revival ({prf:ref}`axiom-guaranteed-revival`)                        | Dead walker ({prf:ref}`def-walker`)s (in a living swarm) must be revivable under the stochastic threshold mechanism. | The cloning score of a dead walker must be guaranteed to exceed the maximum random threshold, $p_{\max}$.                        | $\kappa_{\text{revival}} = \eta^{(\alpha+\beta)} / (\varepsilon_{\text{clone}} \cdot p_{\max})$  | If $\kappa_{\text{revival}} \leq 1$, revival becomes probabilistic instead of guaranteed. This leads to swarm collapse through gradual attrition. A large $p_{\max}$ increases this risk.                                                                                                                                                                                                                                                      |
| Axiom of Boundary Regularity ({prf:ref}`axiom-boundary-regularity`)                      | The "death boundary" of the valid domain must be smooth, not a jagged cliff.                 | The probability of a walker becoming invalid must be a smooth (Hölder continuous) function of its position.                      | $L_{\text{death}}$ (Boundary Instability Factor)<br>$\alpha_B$ (Boundary Smoothing Exponent)     | A large $L_{\text{death}}$ indicates a sharp, unpredictable boundary where small movements can cause massive, unexpected walker deaths, risking swarm collapse.                                                                                                                                                                                                                                                                                |
| **Environmental Axioms**                                                               | **"The problem must be learnable."**                                                         |                                                                                                                                  |                                                                                                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Axiom of Environmental Richness ({prf:ref}`axiom-environmental-richness`)                | The environment must have interesting features at a given resolution; it cannot be flat.     | The reward function must have a guaranteed minimum level of variance in any local region with a radius greater than $r_{\min}$.  | $r_{\min}$ (Minimum Richness Scale)<br>$\kappa_{\text{richness}}$ (Environmental Richness Floor) | If $\kappa_{\text{richness}}$ ≈ 0 for a chosen $r_{\min}$, the swarm can get stuck in regions of that scale with no reward gradient, stalling the learning process.                                                                                                                                                                                                                                                                            |
| Axiom of Reward Regularity ({prf:ref}`axiom-reward-regularity`)                          | The reward signal must be smooth, not chaotic or noisy.                                      | The reward function, when viewed in the algorithmic space, must be Hölder continuous.                                            | $L_{R,\Upsilon}$ (Reward Volatility Factor)<br>$\alpha_R$ (Reward Smoothing Exponent)            | A large $L_{R,\Upsilon}$ signifies a "bumpy" reward landscape. This makes the exploitation signal noisy and can destabilize cloning decisions.                                                                                                                                                                                                                                                                                                 |
| **Algorithmic & Operator Axioms**                                                      | **"The algorithm's internal mechanics must be stable and active."**                          |                                                                                                                                  |                                                                                                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Axiom of Sufficient Amplification ({prf:ref}`axiom-sufficient-amplification`)            | The algorithm must actually use the reward and diversity signals.                            | The dynamics weights $\alpha$ and $\beta$ cannot both be zero.                                                                   | $\kappa_{\text{amplification}} = \alpha + \beta$                                                 | If $\kappa_{\text{amplification}} = 0$, fitness is always 1 for alive walkers. No cloning can occur, and the swarm becomes a collection of non-interacting random walkers.                                                                                                                                                                                                                                                                     |
| Axiom of Non-Degenerate Noise ({prf:ref}`axiom-non-degenerate-noise`)                    | Walkers must be able to move and explore the state space.                                    | The perturbation ({prf:ref}`def-perturbation-measure`) and cloning noise scales must not be zero.                                                                      | $\sigma$ (Perturbation Noise)<br>$\delta$ (Cloning Noise)                                        | If $\sigma=0$ and $\delta=0$, no new positions can be introduced. The swarm loses all exploratory capability and eventually collapses to a few points.                                                                                                                                                                                                                                                                                         |
| Axiom of Variance Regularization (Sec 1.4)                                             | The standardization operator's sensitivity must be bounded.                                  | The raw value variance is prevented from being pathologically close to zero by a smooth floor mechanism.                         | $\kappa_{\text{var,min}}$ (Variance Floor Threshold)                                             | This axiom is a prerequisite for the deterministic Lipschitz continuity of the standardization operator. If not enforced (e.g., by setting $\kappa_{\text{var}},min=0$), the operator is only mean-square continuous, and stronger convergence theorems (like FK) do not apply.                                                                                                                                                                |
| **Geometric Axioms**                                                                   | **"Exploration must be well-behaved."**                                                      |                                                                                                                                  |                                                                                                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Axiom of Geometric Consistency ({prf:ref}`axiom-geometric-consistency`)                  | Exploration noise should be unbiased and explore space evenly (isotropic).                   | The algorithmic noise measure should not introduce systematic bias or skewed diffusion unless intended.                          | $\kappa_{\text{drift}}$ (Anomalous Drift)<br>$\kappa_{\text{anisotropy}}$ (Diffusion Anisotropy) | $\kappa_{\text{drift}} > 0$ introduces a systematic bias, confounding optimization. $\kappa_{\text{anisotropy}} > 1$ causes inefficient or misaligned exploration.                                                                                                                                                                                                                                                                             |
| **Aggregator Axioms**                                                                  | **"The statistical engine must be robust and well-behaved."**                                |                                                                                                                                  |                                                                                                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Bounded Relative Collapse ({prf:ref}`axiom-bounded-relative-collapse`)                   | The framework's scaling analysis is only valid for non-catastrophic events.                  | Analysis of structural error scaling is only certified for transitions where the swarm does not shrink below a certain fraction. | $c_{\min}$ (Relative Collapse Tolerance)                                                         | **This axiom represents a critical limitation of the framework.** If the condition is violated (i.e., during a catastrophic collapse), the system is in a regime where the framework's guarantees on aggregator scaling **do not apply**. The analysis is therefore valid for ensuring the system remains in a stable regime, but it is **not certified to describe the dynamics of the very collapse events it is meant to help understand.** |
| Bounded Deviation from Aggregated Variance ({prf:ref}`axiom-bounded-deviation-variance`) | The aggregator's output variance must honestly reflect the input data's spread.              | The sum of squared deviations from the mean must be controllably related to the computed variance.                               | $\kappa_{\text{var}}$ (Variance Deviation Factor)                                                | If $\kappa_{\text{var}} >> 1$, the aggregator produces statistical moments that are pathologically decoupled from the input data, leading to unreliable standardization and potential instability in the fitness potential calculation.                                                                                                                                                                                                        |

#### 2.5.2 **Summary of Key System-Level Theorems**

These theorems are not axioms but are critical, provable consequences of the axiomatic framework that govern the algorithm's stability and adaptive behavior.

| Theorem Name & Section                                                                     | Core Principle                                                                                                     | Mathematical Result (General Form)                                                                                                                                                                                                                                                 | Key Inputs (General Case)                                                                                                                       | Implications & Failure Modes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|:-------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Theorem of Swarm Update Continuity (§18) | The one-step evolution of the swarm is a continuous map, but its non-linearity prevents simple stability analysis. | For $k_1\ge 2$, the expected squared displacement after one timestep is bounded by the sum of a Lipschitz term and a Hölder term of the initial displacement: <br> $\mathbb{E}[d_{\text{out}}^2] \le C_L d_{\text{in}}^2 + C_H (d_{\text{in}}^2)^{\alpha_H^{\mathrm{global}}} + K$ | All axiomatic parameters, especially the **Boundary Smoothing Exponent** ($\alpha_B$).                                                          | **The system is NOT a simple contraction mapping.** The final continuity bound is of the form $\mathbb{E}[d_{\text{out}}^2] \le C_L d_{\text{in}}^2 + C_H (d_{\text{in}}^2)^{\alpha_H^{\mathrm{global}}} + K$. <br> 1. **Hölder Dominance:** For small displacements, the Hölder term $(d_{\text{in}}^2)^{\alpha_H^{\mathrm{global}}}$ dominates. The system's dynamics are fundamentally non-linear, and stability cannot be determined by simply checking if a single coefficient is less than 1. <br> 2. **Loss of Simple Stability Condition:** The concept of a single "amplification factor" is invalid. Proving long-term stability requires analyzing the fixed points of this non-linear map, a much more complex task. The parameter $\alpha_B$ is revealed to be as critical as any other for determining system stability. |
| Theorem of Forced Activity ({prf:ref}`thm-forced-activity`)                                      | A healthy, diverse swarm must have a non-zero chance of adapting via cloning.                                      | The minimum average cloning probability $p_{\text{clone,min}}$ is strictly positive under the right conditions.                                                                                                                                                                    | Axiomatic parameters from Richness ($\kappa_{\text{richness}}$, $r_{\min}$), Amplification ($\alpha$, $\beta$), and Noise ($\sigma$, $\delta$). | If $p_{\text{clone,min}} = 0$, the system can enter states where adaptation stops. The contractive force of cloning vanishes, stalling the algorithm and preventing convergence.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |

#### 2.4.3 Axiom of Position‑Only Status Margin

:::{prf:axiom} Axiom of Position‑Only Status Margin
:label: axiom-margin-stability
There exists a uniform margin $r_{\mathrm{pos}}>0$ such that for any two swarms $\mathcal{S}_1,\mathcal{S}_2\in\Sigma_N$ ({prf:ref}`def-swarm-and-state-space`) with

$$
\frac{1}{N}\sum_{i=1}^N d_{\mathcal Y}\!\big(\varphi(x_{1,i}),\varphi(x_{2,i})\big)^2\;\le\; r_{\mathrm{pos}},

$$

the alive/dead decisions are invariant under the status update:

$$
n_c(\mathcal{S}_1,\mathcal{S}_2)=0.

$$

In words, sufficiently small positional perturbations (average squared displacement) cannot flip any walker ({prf:ref}`def-walker`)’s status; the alive/dead decision has a uniform buffer that is independent of the metric’s status penalty.
:::

:::{prf:remark}
:label: rem-margin-stability
The Axiom of Margin Stability ({prf:ref}`axiom-margin-stability`) expresses a deterministic stability of the status update ({prf:ref}`def-status-update-operator`) in terms of the positional component alone. It is strictly stronger than the trivial consequence of the identity

$$
d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1,\mathcal{S}_2)^2 = \tfrac{1}{N}\,\Delta_{\text{pos}}^2 + \tfrac{\lambda_{\mathrm{status}}}{N}\,n_c,

$$

which would otherwise allow a tautological “margin” by tuning $\lambda_{\mathrm{status}}$.
n_c\;\le\; \frac{N}{\lambda_{\mathrm{status}}}\, d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1,\mathcal{S}_2)^2,\qquad
n_c^2\;\le\; \left(\frac{N}{\lambda_{\mathrm{status}}}\right)^2 d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1,\mathcal{S}_2)^4.

$$
The margin-based axiom strengthens this near zero by ensuring $n_c=0$ whenever the displacement is small enough, which is crucial to guarantee deterministic continuity of downstream operators.
:::
| Theorem of Deterministic Potential Continuity ([](#thm-deterministic-potential-continuity)) | The fitness potential operator can be made globally Lipschitz continuous.                                          | The deterministic squared error $                                                                                                                                                                                                                |                                                                                                                                                 | V_1 - V_2 \|^2$ is bounded by a Lipschitz-Hölder function of the input displacement and raw value difference.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | The **Axiom of Variance Regularization** ($\kappa_{\text{var}},min$) and all other axiomatic parameters.                                                              | This is the **strongest continuity result**, available when using the patched standardization operator. It proves the potential is a well-behaved, deterministic function suitable for worst-case analysis. This property is the key prerequisite for validating stronger convergence results like those from Feynman-Kac theory. If the axiom is not enforced, this theorem does not hold. |
## 4. The Environment: State and Reward Measurement ({prf:ref}`def-reward-measurement`)
The environment provides the static context for the swarm ({prf:ref}`def-swarm-and-state-space`)'s evolution. Its core properties—the state space and the reward function—are defined axiomatically in Section 2.2. The algorithm interacts with the environment through a formal measurement process.
### 4.1 Reward Measurement ({prf:ref}`def-reward-measurement`)
A walker ({prf:ref}`def-walker`) determines the value of its location by evaluating the global Reward Function.
:::{prf:definition} Reward Measurement
:label: def-reward-measurement
The reward value $r_i$ for walker ({prf:ref}`def-walker`) $i$ at position $x_i$ is the result of integrating the global Reward Function $R$ against the walker's **positional measure**, which is the Dirac delta measure $\delta_{x_i}$ on $\mathcal{X}$.

$$
r_i := \mathbb{E}_{\delta_{x_i}}[R] = \int_{\mathcal{X}} R(x) \, d\delta_{x_i}(x) = R(x_i)

$$
This formalizes the act of "evaluating the reward" as a measurement process.
:::
## 5. Algorithmic Noise Measures
The algorithm's random movements are sourced from probability measures that must satisfy the core properties defined in the **Valid Noise Measure ({prf:ref}`def-valid-noise-measure`) Axiom (Def. 2.3)**. The user is responsible for providing concrete instantiations of these measures.
### 5.1 Algorithmic Instantiations
The algorithm uses two distinct noise measures, both of which are required to be instantiations of a Valid Noise Measure ({prf:ref}`def-valid-noise-measure`).
:::{prf:definition} Perturbation Measure
:label: def-perturbation-measure
For a given noise scale $\sigma > 0$ ({prf:ref}`axiom-non-degenerate-noise`), the **Perturbation Measure ({prf:ref}`def-perturbation-measure`)**, $\mathcal{P}_\sigma(x, \cdot)$, is a **Valid Noise Measure** according to {prf:ref}`def-valid-noise-measure`. It governs the random walks during the perturbation step of the algorithm.
:::
:::{prf:definition} Cloning Measure
:label: def-cloning-measure
For a given cloning noise scale $\delta > 0$ ({prf:ref}`axiom-non-degenerate-noise`), the **Cloning Measure ({prf:ref}`def-cloning-measure`)**, $\mathcal{Q}_\delta(x, \cdot)$, is a **Valid Noise Measure** according to {prf:ref}`def-valid-noise-measure`. It governs the displacement for newly created walker ({prf:ref}`def-walker`)s ({prf:ref}`def-alive-dead-sets`) during the cloning step.
:::
### 5.2 Guidance on Validating Noise Measures (Illustrative Examples)
The axiomatic framework requires that any chosen noise measure satisfies two key properties: uniform displacement ({prf:ref}`axiom-non-degenerate-noise`) and boundary regularity ({prf:ref}`axiom-boundary-regularity`). The user of this framework is responsible for selecting a specific measure and providing a formal proof that it satisfies these axioms. The following lemmas are provided not as part of the core framework, but as illustrative templates for how such a validation proof would be constructed for two canonical examples.
#### 5.2.1 Lemma: Validation of the Heat Kernel
:::{prf:lemma} Validation of the Heat Kernel
:label: lem-validation-of-the-heat-kernel
If the state space $(\mathcal{X}, d_{\mathcal{X}}, \mu)$ is a Polish metric measure space with a canonical heat kernel $p_t(x, \cdot)$ that has a uniformly bounded second moment, then defining the perturbation noise measure ({prf:ref}`def-valid-noise-measure`) as $\mathcal{P}_\sigma(x, \cdot) := p_{\sigma^2}(x, \cdot)$ satisfies the required axioms, provided the boundary ({prf:ref}`axiom-boundary-smoothness`)valid set $\mathcal{X}_{\mathrm{valid}}$ is sufficiently regular.
:::{prf:proof}
:label: proof-lem-validation-of-the-heat-kernel
**Proof.**
1.  **Axiom of Bounded Second Moment of Perturbation** ({prf:ref}`axiom-non-degenerate-noise`): The definition of the state space requires that the heat kernel has a uniformly bounded second moment, i.e., $\sup_{x \in \mathcal{X}} \mathbb{E}_{x' \sim p_t(x, \cdot)} [ d_{\mathcal{Y}}(\varphi(x'), \varphi(x))^2 ] \le M_{\text{pert}}^2$. This directly satisfies the axiom.
2.  **Axiom of Boundary Regularity** ({prf:ref}`axiom-boundary-regularity`): The death probability is given by the function $P(s_{\text{out}}=0 | x) = \int_{\mathcal{X} \setminus \mathcal{X}_{\mathrm{valid}}} p_{\sigma^2}(x, dx')$. This is the convolution of the indicator function of the invalid set with the heat kernel. For non-pathological boundaries (e.g., boundaries that are not space-filling curves), the heat kernel is a well-known smoothing operator. Standard results in analysis show that the convolution of a smooth kernel with an indicator function results in a continuous function. For heat kernels specifically, the resulting function $P$ is smooth and therefore locally Hölder continuous. Global Hölder continuity follows on compact subsets of $\mathcal X$ by a finite subcover argument.
**Q.E.D.**
:::
##### 11.3.8 Remark: Explicit Constants for Standardization Bounds
For quick reference, the constants appearing in the deterministic and mean-square bounds are given explicitly as follows (see the cited definitions):
- C_{V,\text{total}}(\mathcal{S}): 3\big(C_{V,\text{direct}} + C_{V,\mu}(\mathcal{S}) + C_{V,\sigma}(\mathcal{S})\big) from [](#def-value-error-coefficients) and [](#def-lipschitz-value-error-coefficients).
- C_{S,\text{direct}}: \big(2 V_{\max} / \sigma'_{\min\,\text{bound}}\big)^2 from [](#def-lipschitz-structural-error-coefficients).
- C_{S,\text{indirect}}(\mathcal{S}_1,\mathcal{S}_2): $2 k_{\text{stable}} (L_{\mu,S})^2 / \sigma'^2_{\min\,\text{bound}} + 2 k_1 \big(2V_{\max}/\sigma'_{\min\,\text{bound}}\big)^2 (L_{\sigma',S})^2 / \sigma'^2_{\min\,\text{bound}}$ from [](#def-lipschitz-structural-error-coefficients).
- $L_{\sigma'_{\text{reg}}}$: $\sup_{V\ge 0} |(\sigma'_{\text{reg}})'(V)| = \frac{1}{2\sigma'_{\min}}$, the global Lipschitz constant of the regularized standard deviation from [](#lem-sigma-reg-derivative-bounds).
These constants depend only on the fixed algorithmic parameters and the pair $(\mathcal{S}_1, \mathcal{S}_2)$ via the alive set ({prf:ref}`def-alive-dead-sets`)s and aggregation Lipschitz functions, and are finite under the axioms stated in Section 2.
:::
#### 5.2.2 Lemma: Validation of the Uniform Ball Measure
:::{prf:lemma} Validation of the Uniform Ball Measure
:label: lem-validation-of-the-uniform-ball-measure
This lemma validates that the uniform ball measure from {prf:ref}`def-reference-measures` satisfies {prf:ref}`def-axiom-bounded-second-moment-perturbation`.

Let the noise measure ({prf:ref}`def-valid-noise-measure`) $\mathcal{P}_\sigma(x, \cdot)$ be defined as the uniform probability measure over a ball of radius $\sigma$ centered at $x$ in the state space $\mathcal{X}$. This measure satisfies theboundary ({prf:ref}`axiom-boundary-smoothness`), provided the boundary of the valid set is sufficiently regular. In particular, the death‑probability map is continuous under mild assumptions; to claim a global Lipschitz modulus with respect to $d_{\text{Disp},\mathcal{Y}}$, assume $\mathcal{X}_{\mathrm{valid}}$ has Lipschitz boundary or finite perimeter so that boundary layer estimates apply. In that case one obtains an explicit bound of the form

$$
L_{\text{death}}\;\le\; \frac{C_{\text{perim}}}{\sigma},

$$
where $C_{\text{perim}}$ depends on the perimeter (surface measure) of $\partial\mathcal{X}_{\mathrm{valid}}$ in the algorithmic metric.
:::

:::{prf:proof}
:label: proof-lem-validation-of-the-uniform-ball-measure
**Proof.**
1.  **Axiom of Bounded Second Moment of Perturbation** ({prf:ref}`axiom-non-degenerate-noise`): A sample $x'$ is drawn from the ball $B(x, \sigma)$. The displacement is, by definition, $d_{\mathcal{X}}(x', x) \le \sigma$. The expected squared projected displacement is therefore bounded:

$$
    \mathbb{E}_{x' \sim \mathcal{P}_\sigma(x, \cdot)} \left[ d_{\mathcal{Y}}(\varphi(x'), \varphi(x))^2 \right] \le L_\varphi^2 \sigma^2

$$
This bound holds for all $x \in \mathcal{X}$, so the supremum is also bounded. The axiom is satisfied.
2.  **Axiom of Boundary Regularity** ({prf:ref}`axiom-boundary-regularity`): Let $\mathbb{1}_{\text{invalid}}(x')$ be the indicator function for the invalid set. The death probability is the convolution of this indicator function with the indicator function of the ball:

$$
    P(s_{\text{out}}=0 | x) = \frac{1}{\text{Volume}(B(x, \sigma))} \int_{B(x, \sigma)} \mathbb{1}_{\text{invalid}}(x') dx' = \frac{\text{Volume}(\mathcal{X}_{\mathrm{invalid}} \cap B(x, \sigma))}{\text{Volume}(B(x, \sigma))}

$$
The function $f(x) = \text{Volume}(\mathcal{X}_{\mathrm{invalid}} \cap B(x, \sigma))$ measures the volume of the intersection of a fixed set with a moving ball. As long as the boundary of $\mathcal{X}_{\mathrm{invalid}}$ is not pathological (e.g., is a Lipschitz submanifold), this function is continuous. For a small displacement of the ball's center, the change in the intersection volume is proportional to the surface area of the boundary segment that enters or leaves the ball. This geometric relationship ensures the function is locally Lipschitz, which implies it is also Hölder continuous with an exponent of 1. Thus, the axiom is satisfied.
**Q.E.D.**
:::
#### 5.2.3 Lemma: BV/perimeter Lipschitz bound for uniform‑ball death probability
:::{prf:lemma} Uniform‑ball deatboundary ({prf:ref}`axiom-boundary-smoothness`) Lipschitz ({prf:ref}`axiom-reward-regularity`)nite perimeter
:label: lem-boundary-uniform-ball
This lemma provides the quantitative Lipschitz ({prf:ref}`axiom-reward-regularity`)quired by {prf:ref}`axiom-boundary-regularity` for the uniform ball perturbation measure ({prf:ref}`def-perturbation-mboundary ({prf:ref}`axiom-boundary-smoothness`)E=\mathcal{X}_{\mathrm{invalid}}\subset\mathcal X$ have finite perimeter (BV boundary) and let $\mathcal P_\sigma(x,\cdot)$ be the uniform law on $B(x,\sigma)$. Define

$$

P_\sigma(x)\;:=\;\mathcal P_\sigma(x, E)\;=\;\frac{1}{\mathrm{Vol}(B_\sigma)}\int \mathbb 1_E(y)\,\mathbb 1_{B_\sigma}(y-x)\,\mathrm dy.

$$
Then there exists a constant $C_d>0$ depending only on the dimension such that for all $x,y\in\mathcal X$,

$$

|P_\sigma(x)-P_\sigma(y)|\;\le\; C_d\,\frac{\mathrm{Per}(E)}{\sigma}\, d_{\mathcal X}(x,y).

$$
If $\varphi$ is $L_\varphi$‑Lipschitz and distances are measured in the algorithmic space ({prf:ref}`def-algorithmic-space-generic`), the bound becomes $L_{\text{death}}\le C_d (\mathrm{Per}(\varphi(E))/\sigma)\,L_\varphi$.
:::
:::{prf:proof}
:label: proof-lem-boundary-uniform-ball
Write $P_\sigma= (\chi_E * K_\sigma)$ with $K_\sigma= \mathbb 1_{B_\sigma}/\mathrm{Vol}(B_\sigma)$. Approximate $K_\sigma$ in $W^{1,1}$ by smooth mollifiers $\{K_\sigma^{(\varepsilon)}\}$ with $\|\nabla K_\sigma^{(\varepsilon)}\|_1\le C_d/\sigma$. For $f\in BV$, $\nabla(f*K)=(Df)*K$ and $\|\nabla(f*K)\|_\infty\le \|Df\|(\mathbb R^d)\,\|\nabla K\|_1$. Taking $f=\chi_E$ gives a Lipschitz bound $\le C_d\,\mathrm{Per}(E)/\sigma$ for $\chi_E*K_\sigma^{(\varepsilon)}$. Passing to the $\varepsilon\to 0$ limit yields the stated bound. The projection to algorithmic space ({prf:ref}`def-algorithmic-space-generic`) introduces the $L_\varphi$ factor.
:::{prf:remark} Projection choice
:label: rem-projection-choice
In this document we take $\varphi=\mathrm{Id}$ so that $L_\varphi=1$ and no perimeter distortion arises from projection. If a nontrivial projection is used, insert the BV/coarea bound for $\mathrm{Per}(\varphi(E))$ with the appropriate distortion factor.
:::
**Q.E.D.**
:::
#### 5.2.4 Lemma: Heat‑kernel Lipschitz bound via BV smoothing
:::{prf:lemma} Heat‑Lipschitz ({prf:ref}`axiom-reward-regularity`)bility is Lipschitz with constant $\lesssim boundary ({prf:ref}`axiom-boundary-smoothness`): lem-boLipschitz ({prf:ref}`axiom-reward-regularity`)
This lemma provides the quantitative Lipschitz bouboundary ({prf:ref}`axiom-boundary-smoothness`)prf:ref}`axiom-boundary-regularity` for the heat kernel perturbation measure ({prf:ref}`def-perturbation-measure`).

Let $E=\mathcal{X}_{\mathrm{invalid}}\subset\mathcal X$ have finite perimeter and let $p_{\sigma^2}$ be the heat kernel at scale $\sigma$. Define $P_\sigma(x)=\int \chi_E(y)\,p_{\sigma^2}(x,\mathrm dy)$. Then

$$

|P_\sigma(x)-P_\sigma(y)|\;\le\; C_d'\,\frac{\mathrm{Per}(E)}{\sigma}\, d_{\mathcal X}(x,y),

$$
with a constant $C_d'$ depending on dimension. Consequently $L_{\text{death}}\lesssim (\mathrm{Per}(\varphi(E))/\sigma)\,L_\varphi$ in the algorithmic metric.
:::
:::{prf:proof}
:label: proof-lem-boundary-heat-kernel
As above, $P_\sigma=\chi_E * p_{\sigma^2}$ and $\nabla(\chi_E * p_{\sigma^2})=(D\chi_E)*p_{\sigma^2}$. Since $\|\nabla p_{\sigma^2}\|_1\asymp 1/\sigma$, convolution with the BV measure $D\chi_E$ yields a Lipschitz bound $\lesssim (\mathrm{Per}(E)/\sigma)$. The projection factor $L_\varphi$ carries distances to the algorithmic space ({prf:ref}`def-algorithmic-space-generic`).
**Q.E.D.**
:::
:::
## 6. Algorithm Space and Distance Measurement
### 6.1 Specification of the Algorithmic Space ({prf:ref}`def-algorithmic-space-generic`)
:::{prf:definition} Algorithmic Space
:label: def-algorithmic-space-generic
An **algorithmic space ({prf:ref}`def-algorithmic-space-generic`)** is a pair $(\mathcal{Y}, d_{\mathcal{Y}})$ consisting of a real vector space $\mathcal{Y}$ and a true metric $d_{\mathcal{Y}}$ ({prf:ref}`axiom-bounded-algorithmic-diameter`) on $\mathcal{Y}$, built on the Ambient Euclidean Structure ({prf:ref}`def-ambient-euclidean`).
:::
### 6.2 Distance Between Positional Measures
The distance between two walker ({prf:ref}`def-walker`)s is formally defined as the distance between the probability measures representing their positions.
:::{prf:definition} Distance Between Positional Measures
:label: def-distance-positional-measures
Let two walkers ({prf:ref}`def-walker`), $i$ and $j$, have their positions represented by the Dirac positional measures $\delta_{x_i}$ and $\delta_{x_j}$. The distance between them in the algorithmic space ({prf:ref}`def-algorithmic-space-generic`) is the **1-Wasserstein distance ($W_1$)** between their **projected positional measures**, with $d_{\mathcal{Y}}$ ({prf:ref}`axiom-bounded-algorithmic-diameter`) as the ground metric.
The projected positional measure for walker ({prf:ref}`def-walker`) $i$ is the pushforward measure $\varphi_* \delta_{x_i} = \delta_{\varphi(x_i)}$. The distance is then:

$$

d(\varphi_* \delta_{x_i}, \varphi_* \delta_{x_j}) := W_1(\delta_{\varphi(x_i)}, \delta_{\varphi(x_j)})

$$
For Dirac measures, the Wasserstein distance simplifies to the ground metric distance between their points of support.
:::
### 6.3 Algorithmic Distance
:::{prf:definition} Algorithmic Distance
:label: def-alg-distance
The **algorithmic distance ({prf:ref}`def-alg-distance`)** $d_{\text{alg}}\colon\mathcal{X}\times\mathcal{X}\to\mathbb{R}_{\ge0}$ is the distance between the projected positional measures ({prf:ref}`def-distance-positional-measures`) of two walkers ({prf:ref}`def-walker`). In practice, this is the distance or semidistance function $d_{\mathcal{Y}}$ ({prf:ref}`axiom-bounded-algorithmic-diameter`) applied to the projected points in the algorithmic space ({prf:ref}`def-algorithmic-space-generic`):

$$

\boxed{d_{\text{alg}}(x_1, x_2) := d_{\mathcal{Y}}(\varphi(x_1), \varphi(x_2))}

$$
This is the practical implementation of the Wasserstein distance between the walker ({prf:ref}`def-walker`)s' projected Dirac measures and serves as the ground distance for all subsequent calculations.
:::
## 7. Swarm ({prf:ref}`def-swarm-and-state-space`) Measuring
To analyze the collective behavior of the finite N-particle system, we define several distances between swarm ({prf:ref}`def-swarm-and-state-space`) states.
These definitions do not involve a mean-field approximation; instead, they provide different lenses through which to view
and measure the dissimilarity between two finite swarms, $\mathcal{S}_1$ and $\mathcal{S}_2$. Each distance allows for different
analytical approaches to the N-particle system's dynamics.
:::{admonition} 🎯 The N-Particle Displacement ({prf:ref}`def-n-particle-displacement-metric`) Metric: Why It Matters
:class: important
:open:
The key insight here is that we need to measure how "different" two swarms are, but swarms have two types of changes:
1. **Position changes**: Walker ({prf:ref}`def-walker`)s move in space (continuous)
2. **Status changes**: Walker ({prf:ref}`def-walker`)s die or revive (discrete)
The N-Particle Displacement Metric ({prf:ref}`def-n-particle-displacement-metric`) elegantly combines both by treating status changes as "jumps" in an extended space. Think of it as measuring distance in a hybrid space where death is just another dimension—a walker ({prf:ref}`def-walker`) dying is like it jumping a fixed distance $\sqrt{\lambda_{\text{status}}}$ in the "survival dimension."
:::
All distances are constructed in the simplified **algorithmic space ({prf:ref}`def-algorithmic-space-generic`)** $\mathcal{Y}$ and combine two components:
1.  A term measuring the dissimilarity between the distributions of alive walker ({prf:ref}`def-walker`)s.
2.  A term measuring the difference in the fraction of surviving walker ({prf:ref}`def-walker`)s.
The configuration of alive walker ({prf:ref}`def-walker`)s in a swarm can be represented as a measure in the algorithmic space. We define two types of representations:
a discrete one capturing the exact positions, and a smoothed one for analyzing coarse-grained behavior.
Each walker ({prf:ref}`def-walker`) performs two key measurements at the start of the algorithmic pipeline: one of its own value (reward) and one of its
relationship to the swarm (distance to a random companion). The stability of the entire algorithm is critically dependent on the continuity properties of the operators that map a swarm state $S$ to the N-dimensional vectors representing these measurements.
### 7.1 N-Particle Displacement Metric ({prf:ref}`def-n-particle-displacement-metric`)
The primary metric used to measure the distance between two N-particle swarm states is the **N-Particle Displacement Metric ({prf:ref}`def-n-particle-displacement-metric`) ($d_{\text{Disp},\mathcal{Y}}$)**. This metric is a foundational component of the framework and is established as a global convention in **Section 1.5**.
:::{note}
The formula $d_{\text{Disp},\mathcal{Y}}^2 = \frac{1}{N}\Delta_{\text{pos}}^2 + \frac{\lambda_{\text{status}}}{N}n_c$ reveals a fundamental trade-off: the parameter $\lambda_{\text{status}}$ controls how much we "care" about walker ({prf:ref}`def-walker`) deaths versus position changes. A large $\lambda_{\text{status}}$ means the algorithm treats death as a major event; a small value means it focuses more on spatial exploration.
:::
### 7.2 Swarm Aggregation Operator
:::{admonition} 💡 Intuitive Understanding
:class: tip
The Swarm Aggregation Operator is like taking a "group photo" of the swarm's measurements. Instead of tracking every individual walker ({prf:ref}`def-walker`)'s reward or distance, we compute summary statistics (mean, variance) that capture the collective behavior. This compression is essential—it reduces N-dimensional chaos to manageable 2D statistics while preserving the information needed for decision-making.
:::
#### 7.2.1 Definition: Swarm Aggregation Operator
:::{prf:definition} Swarm Aggregation Operator
:label: def-swarm-aggregation-operator-axiomatic
A **Swarm Aggregation Operator ({prf:ref}`def-swarm-aggregation-operator-axiomatic`)**, denoted $M$, is a function that maps a swarm state $\mathcal{S}$ ({prf:ref}`def-swarm-and-state-space`) and a raw value vector $\mathbf{v}$ (defined on the alive set $\mathcal{A}(\mathcal{S})$ from {prf:ref}`def-alive-dead-sets`) to a probability measure $\mu_{\mathbf{v}}$ on $\mathbb{R}$.
**Signature:** $M: \Sigma_N \times \mathbb{R}^{|\mathcal{A}(\mathcal{S})|} \to \mathcal{P}(\mathbb{R})$
For the operator to be valid, it must satisfy the foundational axioms for aggregators defined in Section 2.3.4. Furthermore, the user must provide proofs and explicit functions for the following continuity and structural properties.
1.  **Value Continuity (Lipschitz):** For a fixed swarm structure $\mathcal{S}$, the moment functions must be Lipschitz continuous with respect to the L2-norm of the input value vector $\mathbf{v}$. The user must provide the **Value Lipschitz Functions**, $L_{\mu,M}(\mathcal{S})$ and $L_{m_2,M}(\mathcal{S})$, such that for any two value vectors $\mathbf{v}_1, \mathbf{v}_2$:

$$
|\mu(\mathcal{S}, \mathbf{v}_1) - \mu(\mathcal{S}, \mathbf{v}_2)| \le L_{\mu,M}(\mathcal{S}) \cdot \|\mathbf{v}_1 - \mathbf{v}_2\|_2

$$
$$
|m_2(\mathcal{S}, \mathbf{v}_1) - m_2(\mathcal{S}, \mathbf{v}_2)| \le L_{m_2,M}(\mathcal{S}) \cdot \|\mathbf{v}_1 - \mathbf{v}_2\|_2

$$

2.  **Structural Continuity (Quadratic):** For a fixed value vector $\mathbf{v}$, the change in the moment functions due to a change in the swarm's alive set ({prf:ref}`def-alive-dead-sets`) must be bounded. The user must provide the **Structural Continuity Functions**, $L_{\mu,S}(\mathcal{S}_1, \mathcal{S}_2)$ and $L_{m_2,S}(\mathcal{S}_1, \mathcal{S}_2)$, which may depend on both the initial and final swarm state ({prf:ref}`def-swarm-and-state-space`)s. These functions must satisfy the following inequalities for any two swarms $\mathcal{S}_1, \mathcal{S}_2$:

$$
|\mu(\mathcal{S}_1, \mathbf{v}) - \mu(\mathcal{S}_2, \mathbf{v})| \le L_{\mu,S}(\mathcal{S}_1, \mathcal{S}_2) \cdot \|\mathbf{s}_1 - \mathbf{s}_2\|_2^2

$$

$$
|m_2(\mathcal{S}_1, \mathbf{v}) - m_2(\mathcal{S}_2, \mathbf{v})| \le L_{m_2,S}(\mathcal{S}_1, \mathcal{S}_2) \cdot \|\mathbf{s}_1 - \mathbf{s}_2\|_2^2

$$
*Note: The structural continuity is defined with respect to the squared L2-norm of the status change vector. For notational convenience in subsequent sections, we define the **total number of status changes** between two swarms as $n_c := \|\mathbf{s}_1 - \mathbf{s}_2\|_2^2 = \sum_{j=1}^N (s_{1,j} - s_{2,j})^2$. This quadratic form is a natural choice because the error in common aggregators (like the empirical mean) is directly proportional to the number of added or removed data points, not its square root.*
:::{hint}
Why quadratic dependence on status changes? When a walker ({prf:ref}`def-walker`) dies or revives, it's like suddenly adding or removing a data point from your dataset. The resulting error in statistics (like the mean) jumps discontinuously. The quadratic form $n_c$ counts these discontinuous jumps, making it the natural measure for how much the aggregated statistics can change.
:::
:::
#### 7.2.2 Example Instantiation: The Empirical Measure Aggregator
:::{admonition} The "Simple Average" Operator
:class: tip
:open:
The empirical measure aggregator is just a fancy name for computing simple statistics (mean, variance) from your alive walker ({prf:ref}`def-walker`)s' measurements. Think of it as:
**Input**: A bunch of numbers from alive walker ({prf:ref}`def-walker`)s (their rewards or distances)
**Output**: Summary statistics (average reward, spread of values)
**Why "measure"?** In probability theory, a "measure" is a way of assigning weights to different outcomes. The empirical measure gives equal weight (1/k) to each of the k alive walker ({prf:ref}`def-walker`)s' values.
**Why this matters**: This is the "statistical engine" that turns individual walker ({prf:ref}`def-walker`) measurements into collective swarm intelligence. It's the bridge between "what each walker sees" and "what the swarm as a whole believes."
:::
##### 7.2.2.1 Lemma: Lipschitz constants for empirical moments (mean and second moment)
:::{prf:lemma} Empirical moments are Lipschitz in L2
:llipschitz ({prf:ref}`axiom-reward-regularity`)l-moments-lipschitz

Let $\mathbf v\in\mathbb R^k$ collect the values of the $k=|\mathcal A(\mathcal S)|$ alive walkers ({prf:ref}`def-alive-dead-sets`). Consider the empirical mean and second raw moment

$$
\mu(\mathbf v) = \frac{1}{k}\sum_{i=1}^k v_i,\qquad m_2(\mathbf v) = \frac{1}{k}\sum_{i=1}^k v_i^{\,2}.

$$

Assume $|v_i|\le V_{\max}$ for all $i$. Then, with respect to the L2 norm on $\mathbb R^k$,

$$
|\mu(\mathbf v_1)-\mu(\mathbf v_2)|\;\le\; \frac{1}{\sqrt{k}}\,\|\mathbf v_1-\mathbf v_2\|_2,\qquad
|m_2(\mathbf v_1)-m_2(\mathbf v_2)|\;\le\; \frac{2 V_{\max}}{\sqrt{k}}\,\|\mathbf v_1-\mathbf v_2\|_2.

$$

In particular, for the empirical aggregator we may take $L_{\mu,M}=1/\sqrt{k}$ and $L_{m_2,M}=2V_{\max}/\sqrt{k}$.
:::
:::{prf:proof}
:label: proof-lem-empirical-moments-lipschitz

Gradients are $\nabla\mu = (1/k)\,\mathbf 1$ and $\nabla m_2 = (2/k)\,(v_1,\dots,v_k)$. Thus

$$
\|\nabla\mu\|_2 = \frac{\sqrt{k}}{k} = \frac{1}{\sqrt{k}},\qquad \|\nabla m_2\|_2 = \frac{2}{k}\,\|\mathbf v\|_2\;\le\; \frac{2}{k}\,\sqrt{k}\,V_{\max}\;=\; \frac{2V_{\max}}{\sqrt{k}}.

$$

Lipschitz constants equal the suprema of these gradient norms, giving the stated bounds.
**Q.E.D.**
:::
The most fundamental aggregation operator is one that produces the standard empirical measure of the raw values from the alive set ({prf:ref}`def-alive-dead-sets`). We now formally prove that this operator satisfies the axiomatic requirements of {prf:ref}`def-swarm-aggregation-operator-axiomatic` and derive its specific continuity constants and growth exponents.
:::{prf:lemma} Axiomatic Properties of the Empirical Measure Aggregator
:label: lem-empirical-aggregator-properties
Let the aggregation operator $M$ be defined such that for any swarm state $\mathcal{S}$ ({prf:ref}`def-swarm-and-state-space`) with alive set $\mathcal{A}(\mathcal{S})$ ({prf:ref}`def-alive-dead-sets`) of size $k = |\mathcal{A}(\mathcal{S})| \ge 1$, and any raw value vector $\mathbf{v}$, it produces the discrete empirical measure:

$$
M(\mathcal{S}, \mathbf{v}) = \frac{1}{k} \sum_{i \in \mathcal{A}(\mathcal{S})} \delta_{v_i}

$$

:::{note}
**Breaking down this formula**:
- $k = |\mathcal{A}(\mathcal{S})|$ is the number of alive walker ({prf:ref}`def-walker`)s
- $\delta_{v_i}$ is a "spike" (Dirac delta) at value $v_i$ - it puts all probability mass exactly at that point
- The sum creates a collection of spikes, one for each alive walker ({prf:ref}`def-walker`)'s value
- The $\frac{1}{k}$ factor gives each walker ({prf:ref}`def-walker`) equal weight
**Result**: A probability distribution that treats each alive walker ({prf:ref}`def-walker`)'s measurement as equally important. This is the foundation for computing means, variances, and other statistics!
:::
This operator is a valid **Swarm ({prf:ref}`def-swarm-and-state-space`) Aggregation Operator**. Assuming the raw values are bounded by $|v_i| \le V_{\max}$, its moment functions, continuity functions, and axiomatic parameters are as follows:
*   **Moments:**
    *   Mean: $\mu(\mathcal{S}, \mathbf{v}) = \frac{1}{k} \sum_{i \in \mathcal{A}(\mathcal{S})} v_i$
    *   Second Moment: $m_2(\mathcal{S}, \mathbf{v}) = \frac{1}{k} \sum_{i \in \mathcal{A}(\mathcal{S})} v_i^2$
:::{hint}
**From moments to decisions**: The mean tells us the "typical" value among alive walker ({prf:ref}`def-walker`)s. The second moment (average of squares) helps compute variance = $m_2 - \mu^2$, which measures how spread out the values are. High variance means diverse measurements → exploration. Low variance means consensus → potential convergence.
:::
*   **Value Continuity (Fixed Structure $\mathcal{S}$):**
    *   $L_{\mu,M}(\mathcal{S}) = k^{-1/2}$
    *   $L_{m_2,M}(\mathcal{S}) \le 2V_{\max} k^{-1/2}$
    *   $L_{\mathrm{var},M}(\mathcal{S}) := L_{m_2,M}(\mathcal{S}) + 2V_{\max} L_{\mu,M}(\mathcal{S}) \le 4V_{\max} k^{-1/2}$
*   **Structural Continuity (Fixed Values $\mathbf{v}$):**
    *   $L_{\mu,S}(\mathcal{S}_1, \mathcal{S}_2) \le \frac{2V_{\max}}{|\mathcal{A}(\mathcal{S}_2)|}$
    *   $L_{m_2,S}(\mathcal{S}_1, \mathcal{S}_2) \le \frac{2V_{\max}^2}{|\mathcal{A}(\mathcal{S}_2)|}$
*   **Axiomatic Parameters:**
    *   Variance Deviation Factor: $\kappa_{\text{var}} = 1$
    *   Range-to-Variance Factor: $\kappa_{\text{range}} = 1$
    *   Structural Growth Exponents: $p_{\mu,S} = -1$, $p_{m_2,S} = -1$, $p_{\text{worst-case}} = -1$
:::
:::{prf:proof}
:label: proof-lem-empirical-aggregator-properties

**Proof.**
Let $k = |\mathcal{A}(\mathcal{S})|$, $k_1 = |\mathcal{A}(\mathcal{S}_1)|$, and $k_2 = |\mathcal{A}(\mathcal{S}_2)|$ ({prf:ref}`def-alive-dead-sets`). Let the raw values be bounded by $|v_i| \le V_{\max}$.
1.  **Value Continuity:**
    We bound the change in moments for a fixed swarm ({prf:ref}`def-swarm-and-state-space`) $\mathcal{S}$ of size $k$ and two value vectors $\mathbf{v}_1, \mathbf{v}_2$.
    *   **Mean:** By the Cauchy-Schwarz inequality:

$$
        |\mu_1 - \mu_2| = \frac{1}{k} \left| \sum_{i \in \mathcal{A}} (v_{1,i} - v_{2,i}) \right| \le \frac{1}{k} \sqrt{k} \sqrt{\sum_{i \in \mathcal{A}} (v_{1,i} - v_{2,i})^2} = k^{-1/2} \|\mathbf{v}_1 - \mathbf{v}_2\|_2

        $$
Thus, $L_{\mu,M}(\mathcal{S}) = k^{-1/2}$.
    *   **Second Moment:**

$$
        |m_{2,1} - m_{2,2}| = \frac{1}{k} \left| \sum_{i \in \mathcal{A}} (v_{1,i}^2 - v_{2,i}^2) \right| = \frac{1}{k} \left| \sum_{i \in \mathcal{A}} (v_{1,i} - v_{2,i})(v_{1,i} + v_{2,i}) \right|

$$
The term $|v_{1,i} + v_{2,i}| \le 2V_{\max}$. Applying Cauchy-Schwarz:

$$
        \le \frac{1}{k} \sqrt{\sum (v_{1,i} - v_{2,i})^2} \sqrt{\sum (v_{1,i} + v_{2,i})^2} \le \frac{1}{k} \|\mathbf{v}_1 - \mathbf{v}_2\|_2 \sqrt{k(2V_{\max})^2} = 2V_{\max} k^{-1/2} \|\mathbf{v}_1 - \mathbf{v}_2\|_2

$$
Thus, $L_{m_2,M}(\mathcal{S}) \le 2V_{\max} k^{-1/2}$.
2.  **Structural Continuity:**
    We bound the change in moments for a fixed value vector $\mathbf{v}$ and two swarms $\mathcal{S}_1, \mathcal{S}_2$. Let $n_c = \|\mathbf{s}_1 - \mathbf{s}_2\|_2^2 = |\mathcal{A}_1 \Delta \mathcal{A}_2|$. We decompose the error by adding and subtracting the intermediate term $\frac{1}{k_2}\sum_{i \in \mathcal{A}_1} v_i$.
    *   **Mean:**

$$
        |\mu_1 - \mu_2| = \left| \frac{1}{k_1}\sum_{i \in \mathcal{A}_1} v_i - \frac{1}{k_2}\sum_{i \in \mathcal{A}_2} v_i \right| \le \left|\frac{1}{k_1}\sum_{i \in \mathcal{A}_1} v_i - \frac{1}{k_2}\sum_{i \in \mathcal{A}_1} v_i \right| + \left|\frac{1}{k_2}\sum_{i \in \mathcal{A}_1} v_i - \frac{1}{k_2}\sum_{i \in \mathcal{A}_2} v_i \right|

$$
The first term is bounded by $\left|\frac{1}{k_1} - \frac{1}{k_2}\right| |\sum_{\mathcal{A}_1} v_i| \le \frac{|k_2 - k_1|}{k_1 k_2} (k_1 V_{\max}) = \frac{|k_2 - k_1|}{k_2}V_{\max}$.
        The second term is $\frac{1}{k_2}|\sum_{i \in \mathcal{A}_1 \setminus \mathcal{A}_2} v_i - \sum_{i \in \mathcal{A}_2 \setminus \mathcal{A}_1} v_i| \le \frac{V_{\max}}{k_2}|\mathcal{A}_1 \Delta \mathcal{A}_2|$.
        Since $|k_2 - k_1| \le n_c$ and $|\mathcal{A}_1 \Delta \mathcal{A}_2| = n_c$, the sum is bounded by $\frac{2V_{\max}}{k_2} n_c$.
        Thus, $L_{\mu,S}(\mathcal{S}_1, \mathcal{S}_2) \le \frac{2V_{\max}}{k_2}$.
    *   **Second Moment:** The derivation is identical, replacing $v_i$ with $v_i^2$ and the bound $V_{\max}$ with $V_{\max}^2$. This yields $L_{m_2,S}(\mathcal{S}_1, \mathcal{S}_2) \le \frac{2V_{\max}^2}{k_2}$.
3.  **Axiom of Bounded Deviation from Aggregated Variance ($\kappa_{\text{var}}$):**
    The axiom requires $\sum_{i \in \mathcal{A}} (v_i - \mu)^2 \le \kappa_{\text{var}} \cdot k \cdot \text{Var}[M]$. For the empirical aggregator ({prf:ref}`lem-empirical-aggregator-properties`), $\mu$ is the sample mean and $\text{Var}[M]$ is the sample variance $\frac{1}{k}\sum_{i \in \mathcal{A}} (v_i - \mu)^2$. The axiom becomes an identity for $\kappa_{\text{var}} = 1$.
4.  **Axiom of Bounded Variance Production ($\kappa_{\text{range}}$):**
    The axiom requires $\text{Var}[M] \le \kappa_{\text{range}} \cdot V_{\max}^2$. The sample variance is $\frac{1}{k}\sum v_i^2 - \mu^2$. Since $v_i^2 \le V_{\max}^2$ and $\mu^2 \ge 0$, we have $\text{Var}[M] \le \frac{1}{k}\sum V_{\max}^2 - \mu^2 = V_{\max}^2 - \mu^2 \le V_{\max}^2$. The axiom is satisfied with $\kappa_{\text{range}} = 1$.
5.  **Structural Growth Exponents:**
    Under the **Axiom of Bounded Relative Collapse ({prf:ref}`axiom-bounded-relative-collapse`)**, $k_2 \ge c_{\min} k_1$. We analyze the asymptotic behavior of the structural continuity functions for large $k_1$:
    *   $L_{\mu,S} \propto k_2^{-1} \le (c_{\min}k_1)^{-1} \propto k_1^{-1}$, implying $p_{\mu,S} = -1$.
    *   $L_{m_2,S} \propto k_2^{-1} \le (c_{\min}k_1)^{-1} \propto k_1^{-1}$, implying $p_{m_2,S} = -1$.
    *   The worst-case exponent is $p_{\text{worst-case}} = \max(-1, -1) = -1$.
**Q.E.D.**
:::
#### 7.3 Smoothed Gaussian Measure
For analyses where the precise locations of individual walkers are less important than their overall density, we represent the swarm's state using a kernel density estimate.
:::{prf:definition} Smoothed Gaussian Measure
:label: def-smoothed-gaussian-measure
This measure provides a smooth noise option for {prf:ref}`def-perturbation-measure` and {prf:ref}`def-cloning-measure`.

The creation of a smoothed measure requires a key analytical parameter:
*   **Smoothed Measure Kernel Scale ($\ell$):** The length scale (standard deviation), $\ell > 0$, of the Gaussian kernel ({prf:ref}`def-reference-measures`) used for smoothing. A larger $\ell$ results in a smoother, less detailed density estimate.
Let $K_\ell(y, y')$ be a Gaussian kernel with length scale $\ell$. The **smoothed Gaussian measure**, denoted $\tilde{\nu}_{\mathcal{S}, \ell}$, is the probability measure whose density is given by:

$$
\tilde{\rho}_{\mathcal{S}, \ell}(y) := \frac{1}{|\mathcal{A}(\mathcal{S})|} \sum_{i \in \mathcal{A}(\mathcal{S})} K_\ell(y, \varphi(x_i))

$$

where $\mathcal{A}(\mathcal{S})$ is the alive set ({prf:ref}`def-alive-dead-sets`).
This representation provides a smooth, differentiable approximation of the swarm ({prf:ref}`def-swarm-and-state-space`)'s distribution in the algorithmic space ({prf:ref}`def-algorithmic-space-generic`) and is the foundation for the $d_{\Sigma_N, L_2}$ distance.
:::
### 7.4 The Cemetery State Measure
To ensure that our swarm metrics are well-defined for all possible outcomes, including the absorption of the swarm into the cemetery state, we must formally define its distributional representation. This is achieved by introducing a unique, abstract measure that represents this terminal state.
:::{prf:definition} Algorithmic space with cemetery point
:label: def-algorithmic-cemetery-extension
Define $\mathcal{Y}^{\dagger}:=\mathcal{Y}\cup\{\dagger\}$ with metric $d_\dagger$ given by

$$

d_\dagger(y_1,y_2)=d_{\mathcal{Y}}(y_1,y_2),\quad d_\dagger(y,\dagger)=D_{\mathrm{valid}}\quad \text{for all }y\in\mathcal{Y}.

$$
Identifying a dead walker ({prf:ref}`def-walker`) with the point $\dagger$ makes the Wasserstein distance to the cemetery law canonical: for any living swarm ({prf:ref}`def-swarm-and-state-space`) law $\nu$ and the cemetery $\delta_{\dagger}$ we have $W_p(\nu,\delta_{\dagger})=D_{\mathrm{valid}}$.
:::
:::{prf:remark} Maximal cemetery distance (design choice)
:label: rem-maximal-cemetery-distance-design-choice
This convention for the distance to the cemetery state ({prf:ref}`def-distance-to-cemetery-state`) selects a maximal, state‑independent distance to the cemetery law so that absorption events represent the largest possible jump in distributional metrics. It simplifies comparisons (no ad‑hoc offsets) and keeps $W_p(\nu,\delta_{\dagger})$ constant across all living $\nu$.
:::
:::{prf:definition} Cemetery State Measure
:label: def-cemetery-state-measure
Let $\mathcal{S}$ be a swarm ({prf:ref}`def-swarm-and-state-space`). Its distributional representation on the algorithmic space ({prf:ref}`def-algorithmic-space-generic`), denoted $\mu_{\mathcal{S}}$, is defined as:
1.  If $|\mathcal{A}(\mathcal{S})| > 0$ ({prf:ref}`def-alive-dead-sets`), $\mu_{\mathcal{S}}$ is the empirical or smoothed measure as defined in 5.1 and 5.2 (i.e., $\mu_{\mathcal{S}} = \nu_{\mathcal{S}}$ or $\mu_{\mathcal{S}} = \tilde{\nu}_{\mathcal{S}, \ell}$).
2.  If $|\mathcal{A}(\mathcal{S})| = 0$, its representation is the unique **Cemetery State Measure**, denoted $\mu_{\mathcal{S}} := \nu_{\emptyset}$.
The Cemetery State Measure $\nu_{\emptyset}$ is an abstract object that does not have a density on $\mathcal{Y}$. Its properties are defined entirely by its interaction with the distance functions used in the swarm ({prf:ref}`def-swarm-and-state-space`) metrics.
:::
:::{prf:definition} Distance to the Cemetery State
:label: def-distance-to-cemetery-state
The distance between any valid probability measure $\nu$ (representing a living swarm ({prf:ref}`def-swarm-and-state-space`)) and the Cemetery State Measure $\nu_{\emptyset}$ is defined to be a maximal constant, ensuring that entering the cemetery state represents the largest possible jump in distributional terms.
*   **For the Wasserstein Metric:** Using the algorithmic cemetery extension, for any measure $\nu$ corresponding to a living swarm ({prf:ref}`def-swarm-and-state-space`):

$$
    W_p(\nu, \nu_{\emptyset}) := D_{\mathrm{valid}} \quad \text{and} \quad W_p(\nu_{\emptyset}, \nu_{\emptyset}) := 0

$$
*   **For the MMD (on living swarms):** We evaluate $\mathrm{MMD}_k$ only between measures supported on $\mathcal Y$ (living swarms). No cemetery extension is defined here. If an extension is required, one must specify an explicit positive‑definite kernel $k^{\dagger}$ on $\mathcal Y^{\dagger}$ that yields the desired constant distances while preserving metric properties. When $k$ is characteristic (e.g., Gaussian/RBF), MMD is a true metric on probability measures over $\mathcal Y$.
*   **For the $L_2$ Distance:** The distance is a pre-defined maximal value $M_{L2}$ for the norm. For any density $\tilde{\rho}$ corresponding to a living swarm:

$$
    \|\tilde{\rho} - \tilde{\rho}_{\emptyset}\|_{L_2} := M_{L2} \quad \text{and} \quad \|\tilde{\rho}_{\emptyset} - \tilde{\rho}_{\emptyset}\|_{L2} := 0

$$
:::
## 8. Companion Selection ({prf:ref}`def-companion-selection-measure`)
The process of choosing a companion is governed by a uniform probability measure on a set of available walker ({prf:ref}`def-walker`) indices. This selection is fundamental to the algorithm's diversity measurement. The stability of this selection process—how the expected outcome changes when the set of available companions changes—is a critical component of the overall system's continuity analysis.
:::{admonition} 🎯 Key Insight: Social Dynamics in the Swarm
:class: important
:open:
Companion selection is where the "swarm intelligence" emerges! Each walker ({prf:ref}`def-walker`) measures its distance to a randomly chosen companion, creating a web of pairwise comparisons. Think of it as each walker asking: "How different am I from a typical member of the swarm?" This creates pressure toward diversity—walkers that are far from others (high distance) get different treatment than those clustered together.
The genius is in the randomness: by selecting companions uniformly at random, we get an unbiased estimate of the swarm's spatial distribution without expensive all-to-all comparisons.
:::
### 8.1 Companion Selection ({prf:ref}`def-companion-selection-measure`) Measure
:::{note}
The companion selection ({prf:ref}`def-companion-selection-measure`) rules handle three distinct scenarios:
1. **Normal operation** (multiple alive walker ({prf:ref}`def-walker`)s): Choose from other alive walkers
2. **Revival mode** (dead walker ({prf:ref}`def-walker`), alive swarm): Choose from any alive walker
3. **Isolation** (single survivor): Be your own companion (self-loop)
The self-loop in isolation is crucial—it ensures the math doesn't break when only one walker ({prf:ref}`def-walker`) survives, allowing the revival mechanism to kick in.
:::
:::{prf:definition} Companion Selection Measure
:label: def-companion-selection-measure
For each walker $i \in \{1, \dots, N\}$ ({prf:ref}`def-walker`) in a swarm $\mathcal{S}$ ({prf:ref}`def-swarm-and-state-space`) with alive set $\mathcal{A}$ ({prf:ref}`def-alive-dead-sets`), the **Companion Selection Measure**, $\mathbb{C}_i(\mathcal{S})$, is a **uniform discrete probability measure** over a support set $S_i \subseteq \{1, \dots, N\}$ of valid companion indices. The support set is defined as:
*   If walker ({prf:ref}`def-walker`) $i$ is alive ($i \in \mathcal{A}$) and there is at least one other alive walker ($|\mathcal{A}| \ge 2$), the support set is all other alive walkers: $S_i := \mathcal{A} \setminus \{i\}$.
*   If walker ({prf:ref}`def-walker`) $i$ is dead ($i \notin \mathcal{A}$) and the alive set ({prf:ref}`def-alive-dead-sets`) is not empty ($|\mathcal{A}| \ge 1$), the support set is all alive walkers: $S_i := \mathcal{A}$.
*   If walker ({prf:ref}`def-walker`) $i$ is the only one alive ($|\mathcal{A}| = 1$ and $\mathcal{A} = \{i\}$), it is its own companion: $S_i := \{i\}$.
*   If the swarm ({prf:ref}`def-swarm-and-state-space`) is empty ($|\mathcal{A}|=0$), the support set is empty: $S_i := \emptyset$.
The measure is defined as $\mathbb{C}_i(\mathcal{S})(\{j\}) = 1/|S_i|$ if $j \in S_i$ and $|S_i|>0$, and 0 otherwise. The expectation of any function $f$ under this measure is $\mathbb{E}_{j \sim \mathbb{C}_i(\mathcal{S})}[f(j)] = \frac{1}{|S_i|} \sum_{j \in S_i} f(j)$.
:::
#### 8.1.1 Sampling Policy (with replacement; independent across walker ({prf:ref}`def-walker`)s)
:::{admonition} Allowed sampling schemes for analysis under McDiarmid
:class: important
- Companions are drawn independently **with replacement**: for each walker ({prf:ref}`def-walker`) $i$, draw an independent $U_i^{\mathrm{comp}}\sim \mathrm{Unif}(0,1)$ and map it through the per‑walker CDF of $\mathbb C_i(\mathcal S)$.
- **Allowed:** multinomial resampling (N independent categorical draws) or **stratified** resampling (independent $U_i\in[(i-1)/N, i/N]$ for each $i$).
- **Disallowed for McDiarmid‑based analysis:** **systematic** resampling that uses a single shared uniform to generate all draws, as it induces within‑step dependence across walker ({prf:ref}`def-walker`)s.
If an implementation uses systematic resampling for efficiency, the per‑step independence in Assumption A is violated and the McDiarmid tail bound in the continuity section does not apply; use a dependent‑variables inequality instead.
:::
### 8.2 Lipschitz Continuity of the Companion Selection ({prf:ref}`def-companion-selection-measure`) Operator
This section establishes a continuity bound for the expectation of a function under the **Companion Selection ({prf:ref}`def-companion-selection-measure`) Measure**. The core challenge is to bound how much this expectation can change when the underlying swarm state ({prf:ref}`def-swarm-and-state-space`) changes from $\mathcal{S}_1$ to $\mathcal{S}_2$. This change is driven by the modification of the companion support set from $S_1$ to $S_2$. We derive the bound by decomposing the total error into two parts: error from the change in the set itself, and error from the change in the normalization constant.
:::{admonition} 💡 Why Lipschitz Continuity Matters Here
:class: tip
Lipschitz continuity means "no sudden jumps"—if the swarm changes slightly, the companion selection statistics change proportionally. This is essential for algorithm stability. Without it, a single walker ({prf:ref}`def-walker`) death could cause wild fluctuations in distance measurements, leading to chaotic, unpredictable behavior. The proof shows that the change is bounded by the number of status changes, ensuring smooth degradation rather than catastrophic failure.
:::
#### 8.2.1 Lemma: Bound on the Error from Companion Set Change
This lemma bounds the component of the error that arises purely from the change in the set of companions, holding the normalization constant fixed.
:::{prf:lemma} Bound on the Error from Companion Set Change
:label: lem-set-difference-bound
Let $S_1$ and $S_2$ be two companion support sets, with $|S_1| > 0$. Let $f_j = f(x_j)$ be a real-valued function bounded by a constant $M_f$ such that $|f_j| \le M_f$ for all $j$.
The absolute difference between the sums over these two sets, normalized by the initial set size $|S_1|$, is bounded by the size of the symmetric difference between the sets, $|S_1 \Delta S_2|$.

$$

\left| \frac{1}{|S_1|} \sum_{j \in S_1} f_j - \frac{1}{|S_1|} \sum_{j \in S_2} f_j \right| \le \frac{M_f}{|S_1|} |S_1 \Delta S_2|

$$

Used in {prf:ref}`proof-thm-total-error-status-bound`.
:::
:::{prf:proof}
:label: proof-lem-set-difference-bound
**Proof.**
1.  **Isolate the Difference in Sums:** Factoring out the common normalization constant $1/|S_1|$, we need to bound $\frac{1}{|S_1|} \left| \sum_{j \in S_1} f_j - \sum_{j \in S_2} f_j \right|$.
2.  **Decompose the Sums:** We partition the sums over disjoint regions: $S_1 = (S_1 \setminus S_2) \cup (S_1 \cap S_2)$ and $S_2 = (S_2 \setminus S_1) \cup (S_1 \cap S_2)$. The difference of sums becomes:

$$
    \left( \sum_{j \in S_1 \setminus S_2} f_j + \sum_{j \in S_1 \cap S_2} f_j \right) - \left( \sum_{j \in S_2 \setminus S_1} f_j + \sum_{j \in S_1 \cap S_2} f_j \right) = \sum_{j \in S_1 \setminus S_2} f_j - \sum_{j \in S_2 \setminus S_1} f_j

$$
3.  **Apply Bounds:** By the triangle inequality and the uniform bound $|f_j| \le M_f$:

$$
    \left| \sum_{j \in S_1 \setminus S_2} f_j - \sum_{j \in S_2 \setminus S_1} f_j \right| \le \sum_{j \in S_1 \setminus S_2} |f_j| + \sum_{j \in S_2 \setminus S_1} |f_j| \le M_f |S_1 \setminus S_2| + M_f |S_2 \setminus S_1|

$$
4.  **Relate to Symmetric Difference:** By definition, $|S_1 \setminus S_2| + |S_2 \setminus S_1| = |S_1 \Delta S_2|$. Combining this with the previous steps gives the final bound:

$$
    \frac{1}{|S_1|} \left| \dots \right| \le \frac{M_f}{|S_1|} |S_1 \Delta S_2|

$$
**Q.E.D.**
:::
#### 8.2.2 Lemma: Bound on the Error from Normalization Change
This lemma bounds the component of the error that arises from changing the normalization constants, from $1/|S_1|$ to $1/|S_2|$, while holding the summation set fixed.
:::{prf:lemma} Bound on the Error from Normalization Change
:label: lem-normalization-difference-bound
Let $S_1$ and $S_2$ be two companion support sets, with $|S_1|, |S_2| > 0$. Let $f_j = f(x_j)$ be a real-valued function bounded by a constant $M_f$.
The absolute difference between two sums over the *same* set $S_2$, but with different normalization constants, is bounded by the absolute difference in the set sizes.

$$

\left| \frac{1}{|S_1|} \sum_{j \in S_2} f_j - \frac{1}{|S_2|} \sum_{j \in S_2} f_j \right| \le \frac{M_f}{|S_1|} \big||S_1| - |S_2|\big|

$$

Used in {prf:ref}`proof-thm-total-error-status-bound`.
:::
:::{prf:proof}
:label: proof-lem-normalization-difference-bound
**Proof.**
1.  **Factor out the Common Sum:** The expression is $\left| \frac{1}{|S_1|} - \frac{1}{|S_2|} \right| \left| \sum_{j \in S_2} f_j \right|$.
2.  **Bound the Sum:** Using the triangle inequality, $\left| \sum_{j \in S_2} f_j \right| \le \sum_{j \in S_2} |f_j| \le |S_2| \cdot M_f$.
3.  **Combine and Finalize:** Substituting the bound on the sum gives:

$$

    \left| \frac{|S_2| - |S_1|}{|S_1| |S_2|} \right| \cdot \left( |S_2| \cdot M_f \right) = \frac{\big||S_2| - |S_1|\big|}{|S_1| |S_2|} \cdot |S_2| M_f = \frac{M_f}{|S_1|} \big||S_1| - |S_2|\big|

$$
**Q.E.D.**
:::
#### 7.2.3 Theorem: Total Error Bound in Terms of Status Changes
:::{prf:theorem} Total Error Bound in Terms of Status Changes
:label: thm-total-error-status-bound
Let $\mathcal{S}_1$ and $\mathcal{S}_2$ be two swarm ({prf:ref}`def-swarm-and-state-space`) states, and for a given walker ({prf:ref}`def-walker`) $i$, let $S_1$ and $S_2$ be its companion support sets, with $|S_1| > 0$. Let the status of each potential companion $j$ be given by $s_{1,j}$ and $s_{2,j}$ in the respective swarms. Let $f_j = f(x_{2,j})$ be a function bounded by $M_f$. Let $n_c$ be the total number of status changes in the swarm.
The total error $E = |\mathbb{E}_{j \sim \mathbb{C}_i(\mathcal{S}_1)}[f_j] - \mathbb{E}_{j \sim \mathbb{C}_i(\mathcal{S}_2)}[f_j]|$ is bounded by:

$$

E \le \frac{2 M_f}{|S_1|} \cdot n_c

$$

This general error bound is applied in {prf:ref}`02_euclidean_gas` for distance operator analysis and in {prf:ref}`09_kl_convergence` for KL-divergence convergence proofs.
:::
:::{prf:proof}
:label: proof-thm-total-error-status-bound
**Proof.**
1.  **Decompose the Total Error:** We introduce an intermediate term and apply the triangle inequality:

$$

    E \le \left| \frac{1}{|S_1|} \sum_{j \in S_1} f_j - \frac{1}{|S_1|} \sum_{j \in S_2} f_j \right| + \left| \frac{1}{|S_1|} \sum_{j \in S_2} f_j - \frac{1}{|S_2|} \sum_{j \in S_2} f_j \right|

$$
2.  **Substitute Proven Bounds:** We substitute the results from {prf:ref}`lem-set-difference-bound` and {prf:ref}`lem-normalization-difference-bound`:

$$

    E \le \frac{M_f}{|S_1|} |S_1 \Delta S_2| + \frac{M_f}{|S_1|} \big||S_1| - |S_2|\big| = \frac{M_f}{|S_1|} \left( |S_1 \Delta S_2| + \big||S_1| - |S_2|\big| \right)

$$
3.  **Relate Set Metrics to Status Changes:** A change in a potential companion's status is what drives changes in the support set $S_i$. A single status change for a walker ({prf:ref}`def-walker`) $j$ can change the size of the symmetric difference $|S_1 \Delta S_2|$ by at most one, and the difference in set sizes $||S_1| - |S_2||$ by at most one. Therefore, both of these set-based metrics are bounded by the total number of status changes among the set of potential companions. This local count is, in turn, bounded by the total number of status changes in the entire swarm ({prf:ref}`def-swarm-and-state-space`), $n_c = \sum_{j=1}^N (s_{1,j}-s_{2,j})^2$. Thus, we have $|S_1 \Delta S_2| \le n_c$ and $||S_1| - |S_2|| \leq n_c$.
4.  **Substitute and Finalize:** We substitute these two bounds into the inequality from step 2:

$$

    E \le \frac{M_f}{|S_1|} \left( n_c + n_c \right) = \frac{2 M_f}{|S_1|} \cdot n_c

$$
**Q.E.D.**
:::
## 9. Rescale Transformation
The core non-linearity of the algorithm is encapsulated in the rescale transformation, which maps standardized Z-scores to positive values that form the components of the fitness potential. For the stability and continuity of the entire system to hold, this transformation cannot be arbitrary. It must be governed by a function that is smooth, bounded, and preserves the ordering of inputs.
:::{admonition} 🎯 The Magic of the Rescale Function
:class: important
:open:
The rescale transformation is where standardized measurements become "fitness components." Think of it as a "enthusiasm function"—it takes a walker ({prf:ref}`def-walker`)'s Z-score (how many standard deviations from average) and converts it to a positive number representing its contribution to fitness.
The key insight: we need this function to be:
1. **Smooth** (no sudden jumps that destabilize the algorithm)
2. **Bounded** (no infinite fitness that breaks the math)
3. **Monotonic** (better scores always mean higher fitness)
This creates a "soft selection" mechanism—walker ({prf:ref}`def-walker`)s aren't simply "selected" or "rejected," but have continuously varying fitness levels.
:::
### 8.1. Axiom of a Well-Behaved Rescale Function
:::{note}
Why these four properties? Each prevents a specific failure mode:
- **Smoothness** prevents discontinuous jumps in fitness
- **Monotonicity** ensures logical ordering (better → higher fitness)
- **Boundedness** prevents infinite fitness that breaks probability calculations
- **Lipschitz continuity** ensures small input changes cause proportional output changes
Violating any property makes the algorithm unpredictable or unstable.
:::
:::{prf:axiom} Axiom of a Well-Behaved Rescale Function
:label: def-axiom-rescale-function
Any function $g_A: \mathbb{R} \to \mathbb{R}_{>0}$ chosen for the rescale transformation must satisfy the following properties to be considered valid within the Fragile framework. The user is responsible for proving that their chosen function complies with these conditions.
*   **1. $C^1$ Smoothness:** The function $g_A(z)$ must be continuously differentiable on its entire domain $\mathbb{R}$. This ensures that the fitness potential is a smooth function of the standardized scores, which is critical for the stability of the cloning dynamics.
*   **2. Monotonicity:** The function must be monotonically non-decreasing. This is equivalent to its first derivative being non-negative for all inputs:

$$

    g'_A(z) \ge 0 \quad \forall z \in \mathbb{R}

$$
This property is essential to guarantee that a higher (better) standardized score never results in a lower fitness potential component.
*   **3. Uniform Boundedness:** The function's range must be a bounded interval $(0, g_{A,\max}]$ for some finite constant $g_{A,\max} > 0$. This prevents the fitness potential from becoming infinite, which is a non-negotiable requirement for the stability of the selection and cloning operators.
*   **4. Global Lipschitz Continuity:** The function must be globally Lipschitz continuous. For a $C^1$ function, this is equivalent to its first derivative being uniformly bounded. There must exist a finite constant $L_{g_A}$, the Lipschitz constant, such that:

$$

    \sup_{z \in \mathbb{R}} |g'_A(z)| = L_{g_A} < \infty

$$
This property is the cornerstone for proving the mean-square continuity of the composite Fitness Potential Operator, as it ensures that small changes in the standardized scores cannot be pathologically amplified.
*   **Framework Application:** Any function satisfying these four conditions can be used as a rescale function ({prf:ref}`def-axiom-rescale-function`). The subsequent sections provide two distinct examples of valid functions: an asymmetric piecewise function and a canonical logistic function, along with proofs that they satisfy this axiom. The specific choice of function and its corresponding constants, $g_{A,\max}$ and $L_{g_A}$, will affect the quantitative continuity bounds of the full system.
:::
### 8.2 Example Instantiation 1: A Smooth Piecewise Rescale Function ({prf:ref}`def-axiom-rescale-function`)
This section provides a concrete example of a rescale function ({prf:ref}`def-axiom-rescale-function`) that satisfies the **Axiom of a Well-Behaved Rescale Function**. It is constructed from exponential and logarithmic segments, smoothly joined by a unique cubic polynomial patch.
:::{admonition} 💡 Intuition Behind the Piecewise Design
:class: tip
This clever design combines three behaviors:
1. **Exponential decay** for negative scores (harsh penalty for below-average)
2. **Logarithmic growth** for positive scores (diminishing returns for excellence)
3. **Saturation** at high scores (preventing runaway fitness)
The cubic polynomial "patch" is like a smooth bridge ensuring no discontinuities where the pieces meet. This creates a fitness landscape that strongly penalizes poor performance while preventing any single "superwalker ({prf:ref}`def-walker`)" from dominating.
:::
#### 8.2.1 Definition of the Asymmetric Rescale Function ({prf:ref}`def-axiom-rescale-function`)
:::{prf:definition} Smooth Piecewise Rescale Function
:label: def-asymmetric-rescale-function
This function is a concrete instantiation that satisfies the {prf:ref}`def-axiom-rescale-function`.

The rescale function ({prf:ref}`def-axiom-rescale-function`) is parameterized by the **Rescale Saturation Threshold ($z_{\max}$)**, which defines the upper saturation limit. For the function to be well-posed with distinct segments, this parameter must satisfy the constraint $z_{\max} > 1$.
The **Smooth Piecewise Rescale Function ({prf:ref}`def-axiom-rescale-function`)** $g_A: \mathbb{R} \to \mathbb{R}_{>0}$ is defined as:

$$

g_A(z) :=
\begin{cases}
\exp(z), & z \le 0 \\
\log(1 + z) + 1, & 0 < z < z_{\max} - 1 \\
P(z), & z_{\max} - 1 \le z \le z_{\max} \\
\log(1 + z_{\max}) + 1, & z > z_{\max}
\end{cases}

$$
The function $P(z)$ is a unique cubic polynomial defined on the interval $[z_{\max}-1, z_{\max}]$. It serves as a $C^1$ (continuously differentiable) patch that smoothly connects the logarithmic curve to the constant saturation value. Its four coefficients are uniquely determined by solving for the following four boundary conditions:
1.  **Match Value at Start:** $P(z_{\max}-1) = \log(z_{\max}) + 1$
2.  **Match Slope at Start:** $P'(z_{\max}-1) = 1 / z_{\max}$
3.  **Match Value at End:** $P(z_{\max}) = \log(1 + z_{\max}) + 1$
4.  **Match Slope at End:** $P'(z_{\max}) = 0$
This construction ensures that $g_A(z)$ is continuously differentiable across its entire domain.
:::
#### 8.2.2 Continuity and Properties of the Asymmetric Rescale Function ({prf:ref}`def-axiom-rescale-function`)
This section formally proves the key properties of the Asymmetric Rescale Function ({prf:ref}`def-axiom-rescale-function`): its existence, $C¹$ smoothness, monotonicity, and global Lipschitz continuity. These properties are essential for analyzing the stability of the entire measurement pipeline, as they ensure that the non-linear transformation of standardized scores is well-behaved and does not introduce pathological instabilities.
##### 8.2.2.1 Existence and Uniqueness of the Smooth Rescale Patch
The construction of the $Smooth Piecewise Rescale Function ({prf:ref}`def-axiom-rescale-function`)$ relies on the existence of a unique cubic polynomial that seamlessly connects the logarithmic and constant segments. The following lemma provides the formal proof that such a polynomial is always uniquely determined by the specified boundary conditions.
:::{prf:lemma} Existence and Uniqueness of the Smooth ({prf:ref}`axiom-boundary-smoothness`)le Patch
:label: lem-cubic-patch-uniqueness
This lemma establishes the mathematical foundation for the cboundary ({prf:ref}`axiom-boundary-smoothness`)rf:ref}`def-asymmetric-rescale-function`.

For any $z_max > 1$, there exists boundary ({prf:ref}`axiom-boundary-smoothness`)olynomial $P(z)$ that satisfies the four $C¹$ boundary conditions specified in the asymmetric rescale function definition.
:::
:::boundary ({prf:ref}`axiom-boundary-smoothness`)el: proof-lem-cubic-patch-uniqueness
**Proof.**
Let the generic cubic polynomial be $P(z) = az³ + bz² + cz + d$, and its derivative be $P'(z) = 3az² + 2bz + c$. The four boundary conditions are defined at the endpoints of the interval, $z₀ = z_max - 1$ and $z₁ = z_max$. Let the required values and derivatives at these points be:
*   $y₀ = P(z₀) = log(z_max) + 1$
*   $y'₀ = P'(z₀) = 1/z_max$
*   $y₁ = P(z₁) = log(1 + z_max) + 1$
*   $y'₁ = P'(z₁) = 0$
1.  **Formulate as a System of Linear Equations:**
    The four boundary conditions create a system of four linear equations for the four unknown coefficients $(a, b, c, d)$:
    1.  $a z₀³ + b z₀² + c z₀ + d = y₀$
    2.  $3a z₀² + 2b z₀ + c + 0d = y'₀$
    3.  $a z₁³ + b z₁² + c z₁ + d = y₁$
    4.  $3a z₁² + 2b z₁ + c + 0d = y'₁$
2.  **Represent as a Matrix Equation:**
    This system can be written in the matrix form $M * **x** = **y**$, where $**x** = [a, b, c, d]ᵀ$:

$$

    \begin{pmatrix}
    z₀³ & z₀² & z₀ & 1 \\
    3z₀² & 2z₀ & 1 & 0 \\
    z₁³ & z₁² & z₁ & 1 \\
    3z₁² & 2z₁ & 1 & 0
    \end{pmatrix}
    \begin{pmatrix} a \\ b \\ c \\ d \end{pmatrix}
    =
    \begin{pmatrix} y₀ \\ y'₀ \\ y₁ \\ y'₁ \end{pmatrix}

$$
3.  **Prove Uniqueness of the Solution:**
    A unique solution for the coefficients $(a, b, c, d)$ exists if and only if the determinant of the coefficient matrix $M$ is non-zero. The matrix $M$ is a confluent Vandermonde matrix. Its determinant has a known closed-form solution:

$$

    \det(M) = (z₁ - z₀)⁴

$$
From the problem definition, $z₁ - z₀ = z_max - (z_max - 1) = 1$.
    Therefore, $det(M) = 1⁴ = 1$.
4.  **Conclusion:**
    Since the determinant of the coefficient matrix is $1$, it is non-zero. This guarantees that the matrix is invertible and the system of linear equations has exactly one unique solution for $(a, b, c, d)$. Consequently, the cubic polynomial $P(z)$ that satisfies the four boundary conditions not only exists but is also unique for any valid choice of $z_max$.
**Q.E.D.**
:::
##### 8.2.2.2 Explicit Coefficients of the Smooth Rescale Patch
To analyze the properties of the derivative of the polynomial patch $P(z)$, we must first find its explicit coefficients. While {prf:ref}`lem-cubic-patch-uniqueness` guarantees their existence and uniqueness, the following lemma provides their exact form in terms of the single governing parameter, $z_{\max}$.
:::{prf:lemma} Explicit Coefficients of the Smooth ({prf:ref}`axiom-boundary-smoothness`)le Patch
:label: lem-cubic-patch-coefficients
Let the cubic polynomial patch $P(z)$ be defined on the interval $[z_{\max}-1, z_{\max}]$. Let this interval beboundary ({prf:ref}`axiom-boundary-smoothness`)he variable $s = z - (z_{\max}-1)$, such that $z \in [z_{\max}-1, z_{\max}]$ boundary ({prf:ref}`axiom-boundary-smoothness`)s \in [0, 1]$. In this normalized coordinate system, the polynomial can be wrboundary ({prf:ref}`axiom-boundary-smoothness`)z(s)) = Q(s) = As^3 + Bs^2 + Cs + D

$$
The four coefficients, $A, B, C, D$, are uniquely determined by the boundary conditions and the parameter $z_{\max} > 1$:
*   $A = \frac{1}{z_{\max}} - 2\log\left(1 + \frac{1}{z_{\max}}\right)$
*   $B = 3\log\left(1 + \frac{1}{z_{\max}}\right) - \frac{2}{z_{\max}}$
*   $C = \frac{1}{z_{\max}}$
*   $D = \log(z_{\max}) + 1$
:::
:::{prf:proof}
:label: proof-lem-cubic-patch-coefficients
**Proof.**
Let the interval be $[z_0, z_1]$, where $z_0 = z_{\max}-1$ and $z_1 = z_{\max}$. The four boundary conditions from the asymmetric rescale function ({prf:ref}`def-axiom-rescale-function`) are:
*   $y_0 = P(z_0) = \log(z_{\max}) + 1$
*   $y'_0 = P'(z_0) = 1/z_{\max}$
*   $y_1 = P(z_1) = \log(1 + z_{\max}) + 1$
*   $y'_1 = P'(z_1) = 0$
We define the polynomial $Q(s)$ for $s = z-z_0 \in [0, 1]$. Its derivative is $Q'(s) = 3As^2 + 2Bs + C$. The boundary conditions are transformed into conditions on $Q(s)$ at $s=0$ and $s=1$.
1.  **Condition at $s=0$:**
    *   The value at the start of the interval must match: $Q(0) = y_0$.
        $A(0)^3 + B(0)^2 + C(0) + D = y_0 \implies D = y_0 = \log(z_{\max}) + 1$.
    *   The derivative at the start must also match. Note that by the chain rule, $P'(z) = \frac{d}{dz}Q(z-z_0) = Q'(z-z_0)$.
        $Q'(0) = y'_0 \implies 3A(0)^2 + 2B(0) + C = y'_0 \implies C = y'_0 = \frac{1}{z_{\max}}$.
2.  **Condition at $s=1$:**
    *   The value at the end of the interval must match: $Q(1) = y_1$.
        $A(1)^3 + B(1)^2 + C(1) + D = y_1 \implies A + B + C + D = y_1$.
    *   The derivative at the end must match: $Q'(1) = y'_1$.
        $3A(1)^2 + 2B(1) + C = y'_1 \implies 3A + 2B + C = y'_1$.
3.  **Solve the System for A and B:**
    We now have a system of two linear equations for the two remaining unknown coefficients, $A$ and $B$.
    *   From the value condition at $s=1$: $A + B = y_1 - D - C = y_1 - y_0 - y'_0$.
    *   From the derivative condition at $s=1$: $3A + 2B = y'_1 - C = y'_1 - y'_0$.
    Let's define the change in value, $\Delta y = y_1 - y_0 = \log(1+z_{\max}) - \log(z_{\max}) = \log(1+1/z_{\max})$. The system becomes:
    1. $A + B = \Delta y - y'_0$
    2. $3A + 2B = y'_1 - y'_0$
    Multiply the first equation by 2:
    $2A + 2B = 2\Delta y - 2y'_0$
    Subtract this from the second equation to solve for $A$:
    $A = (y'_1 - y'_0) - (2\Delta y - 2y'_0) = y'_1 + y'_0 - 2\Delta y$
    Substitute the known values ($y'_1=0, y'_0=1/z_{\max}, \Delta y=\log(1+1/z_{\max}))$:
    $A = 0 + \frac{1}{z_{\max}} - 2\log\left(1 + \frac{1}{z_{\max}}\right)$
    Now, solve for $B$ using the first equation:
    $B = (\Delta y - y'_0) - A = (\Delta y - y'_0) - (y'_1 + y'_0 - 2\Delta y) = 3\Delta y - 2y'_0 - y'_1$
    Substitute the known values:
    $B = 3\log\left(1 + \frac{1}{z_{\max}}\right) - \frac{2}{z_{\max}} - 0$
    The expressions for $A, B, C, D$ match those stated in the lemma. This completes the proof.
**Q.E.D.**
:::
##### 8.2.2.3 Explicit Form of the Polynomial Patch Derivative
Having derived the explicit coefficients for the cubic polynomial patch, we can now define the exact functional form of its first derivative. This isolates the quadratic function whose bounds we must analyze to establish the global Lipschitz continuity of the complete rescale function ({prf:ref}`def-axiom-rescale-function`).
:::{prf:lemma} Explicit Form of the Polynomial Patch Derivative
:label: lem-cubic-patch-derivative
Let $P(z)$ be the cubic polynomial patch on the interval $[z_{\max}-1, z_{\max}]$, and let $s = z - (z_{\max}-1)$ be the normalized coordinate on $[0, 1]$. The first derivative of the polynomial, $P'(z)$, is a quadratic function of the normalized coordinate $s$, given by:

$$

P'(z(s)) = 3\left(\frac{1}{z_{\max}} - 2\log\left(1 + \frac{1}{z_{\max}}\right)\right)s^2 + 2\left(3\log\left(1 + \frac{1}{z_{\max}}\right) - \frac{2}{z_{\max}}\right)s + \frac{1}{z_{\max}}

$$
:::
:::{prf:proof}
:label: proof-lem-cubic-patch-derivative
**Proof.**
The proof is a direct substitution of the coefficients found in {prf:ref}`lem-cubic-patch-coefficients` into the general form for the derivative of a cubic polynomial expressed in the normalized coordinate system.
1.  **Recall the Polynomial and its Derivative:**
    From {prf:ref}`lem-cubic-patch-coefficients`, the polynomial patch is expressed as $Q(s) = As^3 + Bs^2 + Cs + D$ for $s \in [0, 1]$. Its derivative with respect to $s$ is:

$$

    Q'(s) = 3As^2 + 2Bs + C

$$
By the chain rule, $P'(z) = Q'(s)$.
2.  **Substitute the Explicit Coefficients:**
    We substitute the explicit expressions for the coefficients $A, B,$ and $C$ as determined in {prf:ref}`lem-cubic-patch-coefficients`:
    *   $A = \frac{1}{z_{\max}} - 2\log\left(1 + \frac{1}{z_{\max}}\right)$
    *   $B = 3\log\left(1 + \frac{1}{z_{\max}}\right) - \frac{2}{z_{\max}}$
    *   $C = \frac{1}{z_{\max}}$
    Substituting these into the expression for $Q'(s)$ yields:

$$

    P'(z(s)) = 3\left(\frac{1}{z_{\max}} - 2\log\left(1 + \frac{1}{z_{\max}}\right)\right)s^2 + 2\left(3\log\left(1 + \frac{1}{z_{\max}}\right) - \frac{2}{z_{\max}}\right)s + \frac{1}{z_{\max}}

$$
This provides the exact analytical form of the derivative on the interval of interest.
**Q.E.D.**
:::
##### 8.2.2.4 Lemma: Monotonicity of the Polynomial Patch (Fritsch–Carlson/Hyman)
:::{prf:lemma} Monotonicity of the Polynomial Patch
:label: lem-polynomial-patch-monotonicity
Let $z_0=z_{\max}-1$, $z_1=z_{\max}$, $y_0=\log(z_{\max})+1$, $y_1=\log(1+z_{\max})+1$, and endpoint slopes $m_0=P'(z_0)=1/z_{\max}$, $m_1=P'(z_1)=0$. Set the secant $\Delta:=(y_1-y_0)/(z_1-z_0)=\log(1+1/z_{\max})>0$. Then the Hermite cubic $P$ on $[z_0,z_1]$ with $(y_0,m_0),(y_1,m_1)$ is monotonically non‑decreasing.
:::
:::{prf:proof}
:label: proof-lem-polynomial-patch-monotonicity
By Fritsch–Carlson/Hyman sufficient conditions for monotone cubic Hermite interpolation, it suffices that $m_0,m_1\ge 0$ and $m_0, m_1 \le 3\Delta$ on the interval. Here $m_1=0$ and $m_0=1/z_{\max}>0$. Moreover, for all $z_{\max}>1$, $1/z_{\max} \le 1 < 3\log(1+1/z_{\max})=3\Delta$. Thus $0\le m_0,m_1\le 3\Delta$, and the interpolant is monotone on $[z_0,z_1]$ by the cited criterion.
**Q.E.D.**
:::
:::{prf:remark}
:label: rem-cubic-hermite-construction
This construction ({prf:ref}`lem-cubic-patch-derivative`, {prf:ref}`lem-polynomial-patch-monotonicity`) is the standard monotone cubic Hermite approach (PCHIP/PCHIM). The global derivative bound $L_P\approx 1.0054$ from §8.2.2.5 provides an explicit Lipschitz constant for the rescale segment.
:::
##### 8.2.2.5 Lemma: Bounds on the Polynomial Patch Derivative
:::{prf:lemma} Bounds on the Polynomial Patch Derivative
:label: lem-cubic-patch-derivative-bounds
Let $P'(z(s))$ be the derivative of the cubic patch on the interval $s \in [0, 1]$, as defined in {prf:ref}`lem-cubic-patch-derivative`. This derivative is uniformly bounded on its domain for any choice of $z_{\max} > 1$. Specifically, it satisfies:

$$

0 \le P'(z(s)) \le L_P

$$
where $L_P$ is a constant slightly greater than 1, given by:

$$

L_P = 1 + \frac{(3\log(2)-2)^2}{3(2\log(2)-1)} \approx 1.0054

$$
:::
:::{prf:proof}
:label: proof-lem-cubic-patch-derivative-bounds
**Proof.**
The proof proceeds by analyzing the function $q(s, x) = P'(z(s))$ for $s \in [0, 1]$ and $x = 1/z_{\max} \in (0, 1)$.
1.  **Non-Negativity (Monotonicity):**
    As formally established in {prf:ref}`lem-polynomial-patch-monotonicity`, the polynomial patch $P(z)$ is monotonically non-decreasing. Since the function is proven to be monotonic, its first derivative must be non-negative. Therefore, we have $P'(z(s)) \ge 0$ for all $s \in [0, 1]$. The minimum value is 0, achieved at the boundary $s=1$.
2.  **Analysis for the Upper Bound:**
    To find the maximum value of $P'(z(s))$, we must find the supremum of the function $q(s, x)$ over its two-dimensional domain $s \in [0,1], x \in (0, 1)$. From {prf:ref}`lem-cubic-patch-derivative`, the function can be expressed in terms of $s$ and $x$:

$$

    q(s, x) = 3\left(x - 2\log\left(1 + x\right)\right)s^2 + 2\left(3\log\left(1 + x\right) - 2x\right)s + x

$$
*   **Dependence on $x = 1/z_{\max}$:** We first analyze the partial derivative of $q$ with respect to $x$ to determine where its maximum is located.

$$

        \frac{\partial q}{\partial x} = (3s^2 - 4s + 1) + \frac{6s - 6s^2}{1+x} = (3s-1)(s-1) + \frac{6s(1-s)}{1+x}

$$
We factor out the non-negative term $(1-s)$:

$$

        \frac{\partial q}{\partial x} = -(1-s)(3s-1) + \frac{6s(1-s)}{1+x} = (1-s)\left[ -(3s-1) + \frac{6s}{1+x} \right] = (1-s)\left[ 1 - 3s + \frac{6s}{1+x} \right]

$$
Let the term in the brackets be $T(s,x) = 1 - 3s + \frac{6s}{1+x}$. Since $x \in (0, 1)$, we have $1+x \in (1,2)$, which implies that the coefficient of $s$, $\frac{6}{1+x}$, is in the range $(3, 6)$. The derivative of $T$ with respect to $s$ is $\frac{\partial T}{\partial s} = -3 + \frac{6}{1+x} > 0$. Thus, for any fixed $x$, $T(s,x)$ is monotonically increasing in $s$. Its minimum value on the interval $s \in [0,1]$ must occur at $s=0$, which gives $T(0,x)=1$. Since the term $T(s,x)$ is always greater than or equal to 1, and the factor $(1-s)$ is non-negative on the domain, the entire partial derivative is non-negative: $\frac{\partial q}{\partial x} \ge 0$. This proves that for any fixed $s$, the function $q(s,x)$ is monotonically increasing with $x$.
    *   **Finding the Supremum:** Because $q(s,x)$ is increasing in $x$, its supremum over the domain must occur at the boundary where $x \to 1$ (which corresponds to $z_{\max} \to 1^+$). We therefore analyze the function in this limit:

$$

        q_{sup}(s) = \lim_{x \to 1} q(s,x) = (6s(1-s))\log(2) + (3s-1)(s-1)

$$
This is a quadratic function of $s$:

$$

        q_{sup}(s) = (3 - 6\log(2))s^2 + (6\log(2) - 4)s + 1

$$
*   **Maximize the Bounding Quadratic:** This is a downward-opening parabola, as its leading coefficient $(3 - 6\log(2)) \approx -1.15$ is negative. Its maximum value occurs at its vertex, $s_v = \frac{-(6\log(2)-4)}{2(3-6\log(2))} = \frac{4-6\log(2)}{6-12\log(2)} \approx 0.068$. Since this vertex lies within the interval $[0,1]$, the maximum of the function is the value at this vertex. The value of a quadratic $as^2+bs+c$ at its vertex $s_v = -b/(2a)$ is given by $c - b^2/(4a)$.

$$

        \max_s q_{sup}(s) = 1 - \frac{(6\log(2)-4)^2}{4(3-6\log(2))} = 1 - \frac{4(3\log(2)-2)^2}{12(1-2\log(2))} = 1 + \frac{(3\log(2)-2)^2}{3(2\log(2)-1)}

$$
Substituting the approximate values gives:

$$

        L_P = 1 + \frac{(2.079 - 2)^2}{3(1.386 - 1)} \approx 1 + \frac{0.00624}{1.158} \approx 1.0054

$$
3.  **Conclusion:**
    The derivative of the polynomial patch is bounded below by 0 and its supremum is the constant $L_P \approx 1.0054$. Therefore, $0 \le P'(z(s)) \le L_P$. This proves the derivative is uniformly bounded on its domain.
**Q.E.D.**
:::
##### 8.2.2.6 Monotonicity of the Smooth Rescale Function ({prf:ref}`def-axiom-rescale-function`)
A critical property of the rescale function ({prf:ref}`def-axiom-rescale-function`) $g_A(z)$ is that it must be monotonically non-decreasing. This ensures that a higher standardized score always results in a higher (or equal) rescaled value, preserving the intended ordering of walker potentials. The following lemma formalizes this property for the entire function.
:::{prf:lemma} Monotonicity of the Smooth Rescale Function
:label: lem-rescale-monotonicity

The $Smooth Piecewise Rescale Function ({prf:ref}`def-axiom-rescale-function`)$ $g_A(z)$ is monotonically non-decreasing on $ℝ$.
:::
:::{prf:proof}
:label: proof-lem-rescale-monotonicity
**Proof.**
To prove that $g_A(z)$ is monotonically non-decreasing, we must show that its first derivative, $g'_A(z)$, is non-negative for all $z ∈ ℝ$. We have already established that $g_A(z)$ is $C¹$, so its derivative is well-defined and continuous everywhere. We now analyze the sign of $g'_A(z)$ on each of the four segments of its piecewise definition.
1.  **For $z ≤ 0$:**
    The function is $g_A(z) = \exp(z)$. The derivative is:

$$

    g'_A(z) = \exp(z)

$$
The exponential function is strictly positive for all real inputs. Thus, $g'_A(z) > 0$ on this interval.
2.  **For $0 < z < z_{\max} - 1$:**
    The function is $g_A(z) = \log(1 + z) + 1$. The derivative is:

$$

    g'_A(z) = \frac{1}{1 + z}

$$
Since $z > 0$ for this interval, the denominator $1 + z$ is always strictly greater than 1. Therefore, $g'_A(z) > 0$ on this interval.
3.  **For $z_{\max} - 1 ≤ z ≤ z_{\max}$:**
    The function is the cubic polynomial patch $g_A(z) = P(z)$. As formally proven in {prf:ref}`lem-cubic-patch-derivative-bounds`, its derivative $P'(z)$ is non-negative on this interval. Therefore, $g'_A(z) = P'(z) ≥ 0$ on this interval.
4.  **For $z > z_{\max}$:**
    The function is the constant $g_A(z) = \log(1 + z_{\max}) + 1$. The derivative is:

$$

    g'_A(z) = 0

$$
The derivative is non-negative on this interval.
5.  **Conclusion:**
    We have shown that $g'_A(z) ≥ 0$ for all $z ∈ ℝ$. Since the derivative is non-negative everywhere, the function $g_A(z)$ is monotonically non-decreasing across its entire domain.
**Q.E.D.**
:::
##### 8.2.2.7 Theorem: Global Lipschitz Continuity of the Smooth Rescale Function ({prf:ref}`def-axiom-rescale-function`)
:::{prf:theorem} Global Lipschitz Continuity of the Smlipschitz ({prf:ref}`axiom-reward-regularity`)`axiom-boundary-smoothness`)Function
:label: thm-rescale-function-lipschitz
The **SLipschitz ({prf:ref}`axiom-reward-regularity`)scale Function** $g_A(z)$ is globally Lipschitz continuous on $\mathbb{R}$. Its Lipschitz constant, $L_{g_A}$, is the supremum of the absolute value of its first derivative, and is given by:

$$

\boxed{
L_{g_A} = \sup_{z \in \mathbb{R}} |g'_A(z)| = L_P = 1 + \frac{(3\log(2)-2)^2}{3(2\log(2)-1)} \approx 1.0054
}

$$
where $L_P$ is the uniform upper bound on the derivative of the polynomial patch from {prf:ref}`lem-cubic-patch-derivative-bounds`.

This Lipschitz continuity result enables the standardization and rescale continuity analysis in {prf:ref}`02_euclidean_gas`.
:::
:::{prf:proof}
:label: proof-thm-rescale-function-lipschitz
**Proof.**
A function that is continuously differentiable on $\mathbb{R}$ is globally Lipschitz if the absolute value of its first derivative is uniformly bounded. We analyze each segment of $g_A$:
1.  **$z \le 0$:** $g'_A(z) = \exp(z)$, whose supremum on $(-\infty, 0]$ is $1$.
2.  **$0 < z < z_{\max} - 1$:** $g'_A(z) = 1/(1+z)$, whose supremum on this interval is $1$ (as $z \to 0^+$).
3.  **$z_{\max}-1 \le z \le z_{\max}$:** $g'_A(z) = P'(z)$; by {prf:ref}`lem-cubic-patch-derivative-bounds`, $0 \le P'(z) \le L_P$.
4.  **$z > z_{\max}$:** $g'_A(z) = 0$.
Taking the supremum over segments yields $L_{g_A} = \max\{1, 1, L_P, 0\} = L_P$, completing the proof.
**Q.E.D.**
:::
### 8.3. Example Instantiation 2: The Canonical Logistic Rescale Function ({prf:ref}`def-axiom-rescale-function`)
##### 8.2.2.8 Global Lipschitz bound for the standardized map
:::{prf:lemma} Lipschitz constant of the patched standardization
:label: lem-lipschitz-constant-of-the-patched-standardization
Let $z=\sigma\'_{\teLipschitz ({prf:ref}`axiom-reward-regularity`)enote the patched standardization of raw values with variance flooLipschitz ({prf:ref}`axiom-reward-regularity`)ext{std}}>0$, and let $g_A$ be the piecewise rescale in §8.2.2. Then $g_A\circ z$ Lipschitz ({prf:ref}`axiom-reward-regularity`)itz. In particular, if the variance functional of the chosen aggregator is $L_{\mathrm{var}}$‑Lipschitz (see §8.2.2.10), then

$$

L_{g_A\circ z}\;\le\; L_{g_A}\cdot L_{z},\qquad L_{g_A}\le \max\{L_P,1\},\quad L_{z}\le L_{\sigma\'_{\text{reg}}}\,L_{\mathrm{var}}.

$$
Here $L_P=\|P'\|_\infty$ is the uniform derivative bound from {prf:ref}`lem-cubic-patch-derivative-bounds`. The factor $L_{\sigma\'_{\text{reg}}}$ is the global Lipschitz constant of the regularized standard deviation, provided by {prf:ref}`lem-sigma-patch-derivative-bound`.
:::
##### 8.2.2.9 Lemma: Patched standard deviation derivative bound
:::{prf:lemma} Derivative bound for \sigma\'_{\text{reg}}
:label: lem-sigma-patch-derivative-bound
Let $\sigma\'_{\text{reg}}(V) = \sqrt{V + \sigma'^2_{\min}}$ be the regularized standard deviation, where $\sigma'_{\min} = \sqrt{\kappa_{\text{var,min}}Lipschitz ({prf:ref}`axiom-reward-regularity`)ext{std}}^2}$. Its global derivative bound is:

$$

L_{\sigma\'_{\text{reg}}}=\sup_{V\ge 0}\big|\,(\sigma\'_{\text{reg}})'(Lipschitz ({prf:ref}`axiom-reward-regularity`)}{2\sigma'_{\min}} = \frac{1}{2\sqrt{\kappa_{\text{var,min}} + \varepsilon_{\text{std}}^2}}

$$
:::
##### 8.2.2.10 Lemma: Lipschitz constant of the empirical variance
:::{prf:lemma} LipschitLipschitz ({prf:ref}`axiom-reward-regularity`)riance functional
:label: lem-lipschitz-bound-for-the-variance-functional
Let $\mu(\mathbf v)$ and $m_2(\mathbf v)$ denote, respectively, the (aggregaLipschitz ({prf:ref}`axiom-reward-regularity`)nd raw moment computed from a value vector $\mathbf v\in\mathbb{R}^k$ over the alive set ({prf:ref}`def-alive-dead-sets`). Assume the moment maps are Lipschitz in $\mathbf v$ with constants $L_{\mu,M}$ and $L_{m_2,M}$ as in §6.2.1, and suppose $|v_i|\le V_{\max}$. Define the variance functional $\mathrm{Var}(\mathbf v):=m_2(\mathbf v) - \mu(\mathbf v)^2$. Then, for all $\mathbf v_1,\mathbf v_2$,

$$

\big|\,\mathrm{Var}(\mathbf v_1)-\mathrm{Var}(\mathbf v_2)\,\big|\;\le\;\Big( L_{m_2,M}\; +\; 2 V_{\max}\,L_{\mu,M}\Big)\,\|\mathbf v_1-\mathbf v_2\|_2.

$$
In particular, the variance functional is $L_{\mathrm{var}}$‑Lipschitz with $L_{\mathrm{var}}:=L_{m_2,M}+2 V_{\max} L_{\mu,M}$.
:::
:::{prf:proof}
:label: proof-lem-lipschitz-bound-for-the-variance-functional
By the triangle inequality,

$$

\big|\,\mathrm{Var}(\mathbf v_1)-\mathrm{Var}(\mathbf v_2)\,\big|\;\le\; |m_2(\mathbf v_1)-m_2(\mathbf v_2)|\; +\; |\mu(\mathbf v_1)^2-\mu(\mathbf v_2)^2|.

$$
The first term is bounded by $L_{m_2,M}\,\|\mathbf v_1-\mathbf v_2\|_2$. For the second, factor the difference of squares:

$$

|\mu(\mathbf v_1)^2-\mu(\mathbf v_2)^2|\;=\; |\mu(\mathbf v_1)+\mu(\mathbf v_2)|\,\cdot\,|\mu(\mathbf v_1)-\mu(\mathbf v_2)|.

$$
With $|\mu(\mathbf v_j)|\le V_{\max}$, we have $|\mu(\mathbf v_1)+\mu(\mathbf v_2)|\le 2 V_{\max}$, while $|\mu(\mathbf v_1)-\mu(\mathbf v_2)|\le L_{\mu,M}\,\|\mathbf v_1-\mathbf v_2\|_2$. Combine to obtain the stated bound.
**Q.E.D.**
:::
:::{prf:corollary} Chain‑rule bound for \sigma\'_{\text{reg}}\circ \mathrm{Var}
:label: cor-chain-rule-sigma-reg-var

Under the conditions of the lemma and §8.2.2.9, the composite map \sigma\'_{\text{reg}}\circ\mathrm{Var} is Lipschitz with

$$

L_{\sigma\'_{\text{reg}}\circ\mathrm{Var}}\;\le\; L_{\sigma\'_{\text{reg}}}\,\Big( L_{m_2,M}+2 V_{\max} L_{\mu,M}\Big).

$$
In particular, for the empirical aggregator of §6.2.2.a (see {prf:ref}`lem-empirical-aggregator-properties`), $L_{\mu,M}=1/\sqrt{k}$ and $L_{m_2,M}=2V_{\max}/\sqrt{k}$, hence

$$

L_{\sigma\'_{\text{reg}}\circ\mathrm{Var}}\;\le\; L_{\sigma\'_{\text{reg}}}\,\Big( \tfrac{2 V_{\max}}{\sqrt{k}} + 2 V_{\max}\,\tfrac{1}{\sqrt{k}}\Big)\n\;=\; \frac{2 L_{\sigma\'_{\text{reg}}} V_{\max}}{\sqrt{k}}.

$$
:::
:::{prf:corollary} Closed‑form bound for $L_{g_A\circ z}$ (empirical aggregator)
:label: cor-closed-form-lipschitz-composite

Let $k=|\mathcal A(\mathcal S)|$ and assume $|v_i|\le V_{\max}$. For the empirical aggregator of §6.2.2.a (see {prf:ref}`lem-empirical-aggregator-properties`) with the regularized standardization and piecewise rescale of §8.2.2, the composite map $g_A\circ z$ is globally Lipschitz with

$$

\boxed{\;L_{g_A\circ z}\;\le\; \max\{\,L_P,\,1\,\}\;\cdot\; \frac{2 L_{\sigma\'_{\text{reg}}} V_{\max}}{\sqrt{k}}\;}

$$
where $L_P=\|P'\|_\infty$ is the uniform derivative bound of the cubic patch from §8.2.2.5.
:::
:::{prf:proof}
:label: proof-cor-closed-form-lipschitz-composite
Combine the bound on $L_{\sigma\'_{\text{reg}}\circ\mathrm{Var}}$ with the Lipschitz constant for $g_A$ from {prf:ref}`thm-rescale-function-lipschitz` and apply the chain rule.
**Q.E.D.**
:::
This section provides a second, more common example of a rescale function ({prf:ref}`def-axiom-rescale-function`) that satisfies the **Axiom of a Well-Behaved Rescale Function**. It is based on the logistic sigmoid, which is widely used for its smoothness and bounded properties.
#### 8.3.1. Definition: Canonical Logistic Rescale Function ({prf:ref}`def-axiom-rescale-function`)
:::{prf:definition} Canonical Logistic Rescale Function
:label: def-canonical-logistic-rescale-function-example
The **Canonical Logistic Rescale Function ({prf:ref}`def-axiom-rescale-function`)** $g_A: \mathbb{R} \to \mathbb{R}_{>0}$ is defined as:

$$

g_A(z) := \frac{2}{1 + e^{-z}}

$$

This canonical rescale function ({prf:ref}`def-axiom-rescale-function`) is used in {prf:ref}`02_euclidean_gas` as the standard choice for the logistic rescaling step in the Euclidean Gas implementation.
:::
#### 8.3.2. Theorem: The Canonical Logistic Function is a Valid Rescale Function ({prf:ref}`def-axiom-rescale-function`)
:::{prf:theorem} The Canonical Logistic Function is a Valid Rescale Function
:label: thm-canonical-logistic-validity
The **Canonical Logistic Rescale Function ({prf:ref}`def-axiom-rescale-function`)** defined in {prf:ref}`def-canonical-logistic-rescale-function-example` satisfies all four conditions of the **Axiom of a Well-Behaveboundary ({prf:ref}`axiom-boundary-smoothness`)on**.
:::
:::{prf:proof}
:label: proof-thm-canonical-logistic-validity
**Proof.**
The proof consists of verifying each of the four axiomatic condiSmooth ({prf:ref}`axiom-boundary-smoothness`) was previously done in $02_relativistic_gas.md$ and is consolidated here.
1.  **$C^1$ Smoothness:** The function is a composition of the exponential function, addition, and division. As the denominator is always non-zero, the function is infinitely differentiable ($C^\infty$) and therefore $C^1$.
2.  **Monotonicity:** The first derivative is $g'_A(z) = 2e^{-z} / (1+e^{-z})^2$. Since $e^{-z} > 0$ and the denominator is a squared real number, $g'_A(z) > 0$ for all $z$. The function is strictly increasing, which satisfies the axiom.
3.  **Uniform Boundedness:** We analyze the limits:
    *   As $z \to \infty$, $e^{-z} \to 0$, so $g_A(z) \to 2 / (1+0) = 2$.
    *   As $z \to -\infty$, $e^{-z} \to \infty$, so $g_A(z) \to 0$.
    The range is the open interval $(0, 2)$. The function is uniformly bounded with $g_{A,\max} = 2$.
4.  **Global Lipschitz Continuity:** As proven previously, the derivative $g'_A(z)$ has a global maximum at $z=0$, where its value is $g'_A(0) = \frac{1}{2}$. The derivative is uniformly bounded by this value. The function is therefore globally Lipschitz with constant $L_{g_A} = \frac{1}{2}$.
Since all four conditions are met, the Canonical Logistic Rescale Function ({prf:ref}`def-axiom-rescale-function`) is a valid instantiation.
**Q.E.D.**
:::
## 10. Abstract Raw Value Measurement
The initial stage of the algorithm's measurement pipeline involves each walker ({prf:ref}`def-walker`) generating a raw scalar value based on its state and its relation to the swarm. To create a modular and reusable analytical framework for the subsequent standardization process, we abstract this initial measurement into a generic operator. Any specific measurement, such as reward or distance-to-companion, will be treated as a concrete instantiation of this abstract operator. The continuity of the entire system depends on this operator being well-behaved in a probabilistic sense.
### 9.1 The Raw Value Operator
:::{prf:definition} Raw Value Operator
:label: def-raw-value-operator
This operator is a generic abstraction used for both {prf:ref}`def-reward-measurement` and {prf:ref}`def-distance-to-companion-measurement`.

A **Raw Value Operator ({prf:ref}`def-raw-value-operator`)**, denoted $V$, is a function that maps a swarm ({prf:ref}`def-swarm-and-state-space`) state $S \in \Sigma_N$ to a **probability distribution** over N-dimensional real-valued vectors, $P(\mathbb{R}^N)$.
**Signature:** $V: \Sigma_N \to P(\mathbb{R}^N)$
For any swarm ({prf:ref}`def-swarm-and-state-space`) state $S$, a single sample $v \sim V(S)$ produces a raw value vector. This process must adhere to the following rules:
1.  **Alive Set Dependency:** The sampling process for the components of $v$ corresponding to the alive set ({prf:ref}`def-alive-dead-sets`) $\mathcal{A}(S)$ may depend on the entire state $S$.
2.  **Dead Set Determinism:** For any walker ({prf:ref}`def-walker`) $i$ that is not in the alive set ({prf:ref}`def-alive-dead-sets`) ($i \notin \mathcal{A}(S)$), the corresponding component of the raw value vector is deterministically zero: $v_i = 0$.
This definition encapsulates both deterministic measurements (like reward, where the distribution is a Dirac delta) and stochastic measurements (like distance-to-companion).
:::
### 9.2 Axiom of Mean-Square Continuity for Raw Values
For the standardization pipeline to be stable, the Raw Value Operator ({prf:ref}`def-raw-value-operator`) cannot change arbitrarily with small changes in the swarm ({prf:ref}`def-swarm-and-state-space`) state. We formalize this by requiring that any valid $V$ must satisfy a mean-square continuity bound. This axiom is the cornerstone of the entire system's stability analysis, as the function it requires, $F_{V,ms}$, will propagate through the continuity proofs of all subsequent operators.
:::{prf:axiom} Axiom of Mean-Square Continuity for Raw Values
:label: axiom-raw-value-mean-square-continuity
Let $V$ be a Raw Value ({prf:ref}`def-raw-value-operator`) Operator. Let $\mathcal{S}_1$ and $\mathcal{S}_2$ be two swarm ({prf:ref}`def-swarm-and-state-space`) states, and let $\mathbf{v}_1 \sim V(\mathcal{S}_1)$ and $\mathbf{v}_2 \sim V(\mathcal{S}_2)$ be independently sampled raw value vectors.
*   **Core Assumption:** The expected squared Euclidean distance between two sampled raw value vectors must be deterministically bounded by a function of the input swarm ({prf:ref}`def-swarm-and-state-space`) states.
*   **Axiomatic Bounding Function:** The user must prove that their chosen operator $V$ satisfies this axiom by providing an explicit, deterministic **Expected Squared Value Error Bound**, $F_{V,ms}(\mathcal{S}_1, \mathcal{S}_2)$, such that:

$$

    \mathbb{E}[\|\mathbf{v}_1 - \mathbf{v}_2\|_2^2] \le F_{V,ms}(\mathcal{S}_1, \mathcal{S}_2)

$$
*   **Failure Mode Analysis:** If an operator violates this axiom, the expected error introduced at the measurement stage can be unbounded. This would lead to a breakdown of continuity in the standardization and cloning decisions, resulting in chaotic and unstable swarm ({prf:ref}`def-swarm-and-state-space`) behavior.
:::
### 9.3 Proving Mean-Square Continuity
A common and effective strategy to prove that an operator satisfies the Axiom of Mean-Square Continuity is to decompose the total expected squared error into a bound on the change of the operator's mean and a bound on its variance. The following axiom, which requires the operator's variance to be uniformly bounded, is a powerful tool for establishing the latter part of the proof.
:::{prf:axiom} Axiom of Bounded Measurement Variance
:label: axiom-bounded-measurement-variance
*   **Core Assumption:** The variance of the raw value ({prf:ref}`def-raw-value-operator`) measurement process ({prf:ref}`axiom-bounded-measurement-variance`), summed over all N walker ({prf:ref}`def-walker`)s, must be uniformly bounded across all possible swarm ({prf:ref}`def-swarm-and-state-space`) states. This axiom prevents the stochastic noise from having pathologically heavy tails that would make the expectation of the squared error diverge.
*   **Axiomatic Parameter ($\kappa^2_{\text{variance}}$ - The Maximum Measurement Variance):** The user of this framework must provide a constant $\kappa^2_{\text{variance}} \ge 0$ that provides a uniform upper bound on the expected squared deviation of a sampled raw value vector from its mean.
*   **Condition:** For any swarm ({prf:ref}`def-swarm-and-state-space`) state $\mathcal{S} \in \Sigma_N$, let $\mathbf{v} \sim V(\mathcal{S})$ be a sampled raw value vector and let $\mathbb{E}[\mathbf{v}]$ be its expectation. The operator must satisfy:

$$

    \mathbb{E}[\|\mathbf{v} - \mathbb{E}[\mathbf{v}]\|_2^2] \le \kappa^2_{\text{variance}}

$$
*   **Framework Application:** This axiom is a key ingredient for proving that a stochastic operator is mean-square continuous. It allows the framework to bound the stochastic fluctuations around the mean, isolating the remaining analytical challenge to bounding the change in the mean itself.
*   **Failure Mode Analysis:** If this axiom is violated, the raw measurement process could have an unbounded variance. This would allow for rare but arbitrarily large measurement errors, causing the *expected* squared error to be infinite. This would break the mean-square continuity of the operator, leading to unstable swarm ({prf:ref}`def-swarm-and-state-space`) behavior.
:::
## 11. Distance-to-Companion Measurement
The second intrinsic measurement is the distance of each walker ({prf:ref}`def-walker`) to a randomly chosen companion. This operator is the primary source of state-dependent instability in the algorithm.
### 10.1 Definition of Distance-to-Companion Measurement
:::{prf:definition} Distance-to-Companion Measurement
:label: def-distance-to-companion-measurement
The distance value $d_i$ for walker ({prf:ref}`def-walker`) $i$ is the result of a two-stage sampling process. First, a **potential companion** index, denoted $c_{\text{pot}}(i)$, is sampled from the **Companion Selection ({prf:ref}`def-companion-selection-measure`) Measure** $\mathbb{C}_i$. Second, the **Algorithmic Distance ({prf:ref}`def-alg-distance`)** is computed to that specific companion.
This process is equivalent to sampling a single value from the **Distance-to-Companion Measure** $\mathbb{D}_i$, which is the pushforward of $\mathbb{C}_i$ by the distance function $D_i(j) = d_{\text{alg}}(x_i, x_j)$.

$$

d_i := d_{\text{alg}}(x_i, x_{c_{\text{pot}}(i)}) \quad \text{where} \quad c_{\text{pot}}(i) \sim \mathbb{C}_i(\cdot)

$$
The **Raw Distance Vector Operator**, denoted $\mathbf{d}(\mathcal{S})$, maps a swarm ({prf:ref}`def-swarm-and-state-space`) state $\mathcal{S}$ to a *distribution* over N-dimensional vectors. A single realization of this vector is produced by performing the distance-to-companion measurement independently for each alive walker ({prf:ref}`def-walker`). For dead walkers, the component is zero.
:::
The continuity of the raw distance vector operator, $\mathcal{S} \mapsto \mathbf{d}(\mathcal{S})$, is substantially more complex to analyze than that of the reward vector. The operator is stochastic, and the measurement for each walker ({prf:ref}`def-walker`) depends on the state of the *entire* swarm via the **Companion Selection Measure**. Its continuity is therefore probabilistic and state-dependent.
Our analytical strategy is to rigorously decompose the problem. We will first establish a continuity bound for the *expectation* of the raw distance vector, $\mathbb{E}[\mathbf{d}(\mathcal{S})]$. This expected vector can be viewed as the deterministic "signal" that the measurement process is trying to capture. We will show that its continuity degrades as the number of alive walker ({prf:ref}`def-walker`)s decreases.
Second, we will establish a concentration inequality that bounds the probabilistic deviation of a single sampled vector, $\mathbf{d}(\mathcal{S})$, from its expectation. This deviation can be viewed as the "noise" inherent in the stochastic measurement.
Finally, by combining the bound on the signal's change with the bound on the noise, we will derive a unified probabilistic continuity bound for the operator itself. This final bound is a critical input to the stability analysis of the **Standardization Operator ({prf:ref}`def-standardization-operator-n-dimensional`)**.
### 10.2 Continuity of the Expected Raw Distance Vector ($k \geq 2$ Regime)
The continuity analysis of the expected raw distance operator, $\mathbb{E}[\mathbf{d}(\mathcal{S})]$, is partitioned into two distinct regimes based on the number of alive walker ({prf:ref}`def-walker`)s, $k=|\mathcal{A}(\mathcal{S})|$. This is necessary because the diversity measurement via companion selection is fundamentally different when only one walker remains, leading to a singularity in the continuity bounds.
This section establishes the continuity bounds for the **continuous regime**, where the number of alive walker ({prf:ref}`def-walker`)s in the initial state is at least two ($k_1 \ge 2$). The subsequent section will address the special boundary case of $k=1$.
The first step in analyzing the stochastic distance operator is to establish the continuity of its expectation, $\mathbb{E}[\mathbf{d}(\mathcal{S})]$. The change in this expected vector, when the swarm state ({prf:ref}`def-swarm-and-state-space`) changes from $\mathcal{S}_1$ to $\mathcal{S}_2$, represents the deterministic "signal" component of the operator's response. A key challenge is that the measurement for each walker ({prf:ref}`def-walker`) $i$ depends on the entire swarm state, introducing complex couplings.
Our strategy is to rigorously derive a bound for the total change by first establishing bounds for the three distinct sources of error that affect a single walker ({prf:ref}`def-walker`)'s expected distance: positional displacement, structural change in the companion set, and the walker's own status change. These modular lemmas provide the foundation for the main theorem, which composes them into a unified, state-dependent continuity bound for the N-dimensional operator.
#### 10.2.1 Lemma: Bound on Single-Walker ({prf:ref}`def-walker`) Error from Positional Change
:::{prf:lemma} Bound on Single-Walker Error from Positional Change
:label: lem-single-walker-positional-error
Let $\mathcal{S}_1$ and $\mathcal{S}_2$ be two swarm ({prf:ref}`def-swarm-and-state-space`) states. For a given walker ({prf:ref}`def-walker`) $i$ that is alive in swarm $\mathcal{S}_1$ ($s_{1,i}=1$), let $\mathbb{C}_i(\mathcal{S}_1)$ be its companion selection measure.
The absolute error in its expected distance due to the positional displacement of the walker ({prf:ref}`def-walker`)s between the two states, evaluated over the fixed companion set from $\mathcal{S}_1$, is bounded by the sum of its own displacement and the average displacement of its potential companions.

$$

\left| \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\text{alg}}(x_{1,i}, x_{1,c}) \right] - \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\text{alg}}(x_{2,i}, x_{2,c}) \right] \right| \le d_{\text{alg}}(x_{1,i}, x_{2,i}) + \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\text{alg}}(x_{1,c}, x_{2,c}) \right]

$$
:::
:::{prf:proof}
:label: proof-lem-single-walker-positional-error
**Proof.**
Let $\Delta_{\text{pos},i}$ denote the absolute error term we wish to bound for walker ({prf:ref}`def-walker`) $i$ in swarm ({prf:ref}`def-swarm-and-state-space`) states $\mathcal{S}_1$ and $\mathcal{S}_2$.
1.  **Apply Linearity of Expectation:**
    We combine the two terms into a single expectation.

$$

    \Delta_{\text{pos},i} = \left| \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\text{alg}}(x_{1,i}, x_{1,c}) - d_{\text{alg}}(x_{2,i}, x_{2,c}) \right] \right|

$$
2.  **Move the Absolute Value Inside the Expectation:**
    Using Jensen's inequality, $|\mathbb{E}[X]| \le \mathbb{E}[|X|]$, we move the absolute value inside, which provides an upper bound:

$$

    \Delta_{\text{pos},i} \le \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ \left| d_{\text{alg}}(x_{1,i}, x_{1,c}) - d_{\text{alg}}(x_{2,i}, x_{2,c}) \right| \right]

$$
3.  **Apply the Reverse Triangle Inequality:**
    The term inside the expectation is the absolute difference between two distance values. We apply the **reverse triangle inequality**, which states that for any points $a,b,c,d$ in a metric space $(M,d)$, $|d(a,b) - d(c,d)| \le d(a,c) + d(b,d)$. Applying this to the Euclidean algorithmic distance ({prf:ref}`def-alg-distance`) $d_{\text{alg}}$ yields:

$$

    \left| d_{\text{alg}}(x_{1,i}, x_{1,c}) - d_{\text{alg}}(x_{2,i}, x_{2,c}) \right| \le d_{\text{alg}}(x_{1,i}, x_{2,i}) + d_{\text{alg}}(x_{1,c}, x_{2,c})

$$
4.  **Finalize the Bound:**
    We substitute this inequality back into the expression from Step 2. By linearity of expectation, the first term, $d_{\text{alg}}(x_{1,i}, x_{2,i})$, is a constant with respect to the expectation over the companion index $c$. This gives the final bound.
**Q.E.D.**
:::
#### 10.2.2 Lemma: Bound on Single-Walker ({prf:ref}`def-walker`) Error from Structural Change
:::{prf:lemma} Bound on Single-Walker Error from Structural Change
:label: lem-single-walker-structural-error
Let $\mathcal{S}_1$ and $\mathcal{S}_2$ be two swarm ({prf:ref}`def-swarm-and-state-space`) states with alive set ({prf:ref}`def-alive-dead-sets`)s $\mathcal{A}_1$ and $\mathcal{A}_2$. Let walker ({prf:ref}`def-walker`) $i$ be alive in both swarms ($i \in \mathcal{A}_1 \cap \mathcal{A}_2$), and let the initial swarm have at least two alive walkers, $k_1=|\mathcal{A}_1| \ge 2$. Let the walker positions $\mathbf{x}_2$ from the second swarm be fixed for the analysis. Let $n_c$ be the total number of status changes in the swarm.
The absolute error in the expected distance for walker ({prf:ref}`def-walker`) $i$ due to the change in the companion selection measure is bounded by:

$$

\left| \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\text{alg}}(x_{2,i}, x_{2,c}) \right] - \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_2)} \left[ d_{\text{alg}}(x_{2,i}, x_{2,c}) \right] \right| \le \frac{2 D_{\mathcal{Y}}}{k_1 - 1} \cdot n_c

$$
where $D_{\mathcal{Y}}$ is the diameter of the algorithmic space ({prf:ref}`def-algorithmic-space-generic`).
:::
:::{prf:proof}
:label: proof-lem-single-walker-structural-error
**Proof.**
This result is a direct application of the [](#thm-total-error-status-bound).
1.  **Identify the Function and Bound:**
    Let the function being evaluated be $f(c) := d_{\text{alg}}(x_{2,i}, x_{2,c})$. This function measures the distance in the algorithmic space ({prf:ref}`def-algorithmic-space-generic`) from walker ({prf:ref}`def-walker`) $i$ to a potential companion $c$ in swarm ({prf:ref}`def-swarm-and-state-space`) states $\mathcal{S}_1$ and $\mathcal{S}_2$ with alive/dead sets ({prf:ref}`def-alive-dead-sets`). The distance is, by definition, bounded by the space's diameter, $D_{\mathcal{Y}}$ ({prf:ref}`axiom-bounded-algorithmic-diameter`). Therefore, we have a uniform bound $M_f = D_{\mathcal{Y}}$.
2.  **Identify the Support Sets:**
    Let $S_1 = \mathbb{C}_i(\mathcal{S}_1)$ and $S_2 = \mathbb{C}_i(\mathcal{S}_2)$ be the companion support sets for walker ({prf:ref}`def-walker`) $i$ in the two swarms. Since walker $i$ is alive in $\mathcal{S}_1$ and the precondition states $k_1 \ge 2$, the initial support set is $S_1 = \mathcal{A}_1 \setminus \{i\}$, and its size is $|S_1| = k_1 - 1 > 0$.
3.  **Apply [](#thm-total-error-status-bound):**
    [](#thm-total-error-status-bound) provides a general bound for the change in expectation due to a change in the support set:

$$

    \text{Error} \le \frac{2 M_f}{|S_1|} \cdot n_c

$$
4.  **Substitute and Finalize:**
    We substitute our specific function bound $M_f = D_{\mathcal{Y}}$ and the support set size $|S_1| = k_1 - 1$ into the general formula. This immediately yields the stated bound.
**Q.E.D.**
:::
#### 10.2.3 Lemma: Bound on Single-Walker ({prf:ref}`def-walker`) Error from Own Status Change
:::{prf:lemma} Bound on Single-Walker Error from Own Status Change
:label: lem-single-walker-own-status-error
Let $\mathcal{S}_1$ and $\mathcal{S}_2$ be two swarm ({prf:ref}`def-swarm-and-state-space`) states. For any walker ({prf:ref}`def-walker`) $i$ whose survival status changes ($s_{1,i} \neq s_{2,i}$), the absolute difference in its expected raw distance measurement is bounded by the diameter of the algorithmic space ({prf:ref}`def-algorithmic-space-generic`), $D_{\mathcal{Y}}$ ({prf:ref}`axiom-bounded-algorithmic-diameter`).

$$

\left| \mathbb{E}[d_i(\mathcal{S}_1)] - \mathbb{E}[d_i(\mathcal{S}_2)] \right| \le D_{\mathcal{Y}}

$$
:::
:::{prf:proof}
:label: proof-lem-single-walker-own-status-error
**Proof.**
The proof considers the two possible cases for a status change.
1.  **Case 1: Walker ({prf:ref}`def-walker`) Dies ($s_{1,i}=1 \to s_{2,i}=0$)**: The expected distance in state $\mathcal{S}_1$ is $\mathbb{E}[d_i(\mathcal{S}_1)]$, which must lie in the interval $[0, D_{\mathcal{Y}}]$. The expected distance in state $\mathcal{S}_2$ is defined to be $\mathbb{E}[d_i(\mathcal{S}_2)] = 0$. The absolute difference is therefore $|\mathbb{E}[d_i(\mathcal{S}_1)] - 0| \le D_{\mathcal{Y}}$.
2.  **Case 2: Walker ({prf:ref}`def-walker`) is Revived ($s_{1,i}=0 \to s_{2,i}=1$)**: The logic is symmetric. $\mathbb{E}[d_i(\mathcal{S}_1)] = 0$ and $\mathbb{E}[d_i(\mathcal{S}_2)] \in [0, D_{\mathcal{Y}}]$. The absolute difference is again bounded by $D_{\mathcal{Y}}$ ({prf:ref}`axiom-bounded-algorithmic ({prf:ref}`def-algorithmic-space-generic`)-diameter`).
In both cases, the bound holds.
**Q.E.D.**
:::
#### 10.2.4. Theorem: Decomposition of the Total Squared Error

:::{prf:theorem} Decomposition of the Total Squared Error
:label: thm-total-expected-distance-error-decomposition

Let $\mathcal{S}_1$ ({prf:ref}`def-alive-dead-sets`) and $\mathcal{S}_2$ be two swarm ({prf:ref}`def-swarm-and-state-space`) states. The total squared difference between their expected raw distance vectors is the sum of the squared differences over all walker ({prf:ref}`def-walker`)s. This sum can be partitioned into a sum over the set of *stable walkers*, $\mathcal{A}_{\text{stable}} = \mathcal{A}(\mathcal{S}_1) \cap \mathcal{A}(\mathcal{S}_2)$, and a sum over the set of *unstable walkers*, $\mathcal{A}_{\text{unstable}} = \mathcal{A}(\mathcal{S}_1) \Delta \mathcal{A}(\mathcal{S}_2)$.

$$
\| \mathbb{E}[\mathbf{d}(\mathcal{S}_1)] - \mathbb{E}[\mathbf{d}(\mathcal{S}_2)] \|_2^2 = \underbrace{\sum_{i \in \mathcal{A}_{\text{stable}}} |\mathbb{E}[d_i(\mathcal{S}_1)] - \mathbb{E}[d_i(\mathcal{S}_2)]|^2}_{\text{Error from Stable Walker ({prf:ref}`def-walker`)s}} + \underbrace{\sum_{i \in \mathcal{A}_{\text{unstable}}} |\mathbb{E}[d_i(\mathcal{S}_1)] - \mathbb{E}[d_i(\mathcal{S}_2)]|^2}_{\text{Error from Unstable Walkers}}

$$
:::

:::{prf:proof}
:label: proof-thm-total-expected-distance-error-decomposition
**Proof.**
This decomposition is an identity that follows directly from partitioning the set of all walker ({prf:ref}`def-walker`) indices $\{1, ..., N\}$ into two disjoint subsets: those whose survival status is the same in both swarm ({prf:ref}`def-swarm-and-state-space`)s, and those whose status changes. The total sum of squared errors over all walkers is simply the sum of the errors over these two partitions.
The set of walker ({prf:ref}`def-walker`)s whose error contribution could be non-zero is the union of the alive set ({prf:ref}`def-alive-dead-sets`)s, $\mathcal{A}(\mathcal{S}_1) \cup \mathcal{A}(\mathcal{S}_2)$. We partition this set into stable walkers, $\mathcal{A}_{\text{stable}} = \mathcal{A}(\mathcal{S}_1) \cap \mathcal{A}(\mathcal{S}_2)$, and unstable walkers, whose indices lie in the symmetric difference of the alive sets, $\mathcal{A}_{\text{unstable}} = \mathcal{A}(\mathcal{S}_1) \Delta \mathcal{A}(\mathcal{S}_2)$. For any walker **i** that is dead in both states, its expected distance is 0 in both states, so its error contribution is 0.
**Q.E.D.**
:::
#### 10.2.5 Lemma: Bound on the Total Squared Error for Unstable Walker ({prf:ref}`def-walker`)s
:::{prf:lemma} Bound on the Total Squared Error for Unstable Walkers
:label: lem-total-squared-error-unstable
Let $\mathcal{S}_1$ ({prf:ref}`def-alive-dead-sets`) and $\mathcal{S}_2$ be two swarm ({prf:ref}`def-swarm-and-state-space`) states. The total squared error in the expected raw distance from the set of unstable walker ({prf:ref}`def-walker`)s, $\mathcal{A}_{\text{unstable}}$, is bounded by the total number of status changes:

$$

\sum_{i \in \mathcal{A}_{\text{unstable}}} \big|\mathbb{E}[d_i(\mathcal{S}_1)] - \mathbb{E}[d_i(\mathcal{S}_2)]\big|^2 \le D_{\mathcal{Y}}^2 \sum_{j=1}^N (s_{1,j} - s_{2,j})^2,

$$
where $D_{\mathcal{Y}}$ ({prf:ref}`axiom-bounded-algorithmic-diameter`) is the diameter of the algorithmic space ({prf:ref}`def-algorithmic-space-generic`).
:::
:::{prf:proof}
:label: proof-lem-total-squared-error-unstable
**Proof.** For any unstable walker ({prf:ref}`def-walker`) $i$ (i.e., $s_{1,i}\neq s_{2,i}$), [](#lem-single-walker-own-status-error) gives
$|\mathbb{E}[d_i(\mathcal{S}_1)] - \mathbb{E}[d_i(\mathcal{S}_2)]| \le D_{\mathcal{Y}}$.
Squaring and summing over all unstable walker ({prf:ref}`def-walker`)s yields the stated bound, since the count
$\big|\mathcal{A}_{\text{unstable}}\big| = \sum_{j=1}^N (s_{1,j}-s_{2,j})^2$.
**Q.E.D.**
:::
#### 10.2.6. Lemma: Bound on the Total Squared Error for Stable Walker ({prf:ref}`def-walker`)s
:::{prf:lemma} Bound on the Total Squared Error for Stable Walkers
:label: lem-total-squared-error-stable
Let $\mathcal{S}_1$ and $\mathcal{S}_2$ be two swarm ({prf:ref}`def-swarm-and-state-space`) states with $|\mathcal{A}(\mathcal{S}_1)|=k_1 \ge 2$ ({prf:ref}`def-algorithmic-space-generic`) ({prf:ref}`def-alive-dead-sets`) ({prf:ref}`def-swarm-and-state-space`). The total squared error in the expected raw distance from the set of stable walker ({prf:ref}`def-walker`)s, $\mathcal{A}_{\text{stable}} = \mathcal{A}(\mathcal{S}_1) \cap \mathcal{A}(\mathcal{S}_2)$, is bounded as follows:

$$

\sum_{i \in \mathcal{A}_{\text{stable}}} |\mathbb{E}[d_i(\mathcal{S}_1)] - \mathbb{E}[d_i(\mathcal{S}_2)]|^2 \le 12 \cdot \Delta_{\text{pos}}^2(\mathcal{S}_1, \mathcal{S}_2) + \frac{8 k_1 D_{\mathcal{Y}}^2}{(k_1 - 1)^2} \cdot n_c(\mathcal{S}_1, \mathcal{S}_2)^2

$$
:::
:::{prf:proof}
:label: proof-lem-total-squared-error-stable
**Proof.**
The total error for a single stable walker ({prf:ref}`def-walker`) is first decomposed into a positional component and a structural component. The squared L2-norm of the total error over all stable walkers is then bounded by combining the established bounds for the sum of the squares of these individual components.
1.  **Decomposition of Total Stable Error:** From [](#sub-lem-stable-walker ({prf:ref}`def-walker`)-error-decomposition), the total squared error for stable walkers is bounded by twice the sum of the squared positional and structural error components:

$$

\sum_{i \in \mathcal{A}_{\text{stable}}} |\mathbb{E}[d_i(\mathcal{S}_1)] - \mathbb{E}[d_i(\mathcal{S}_2)]|^2 \le 2 \sum_{i \in \mathcal{A}_{\text{stable}}} (\Delta_{\text{pos},i})^2 + 2 \sum_{i \in \mathcal{A}_{\text{stable}}} (\Delta_{\text{struct},i})^2

$$
2.  **Bound the Positional Component:** From [](#sub-lem-stable-positional-error-bound), the total squared positional error component is bounded by:

$$

\sum_{i \in \mathcal{A}_{\text{stable}}} (\Delta_{\text{pos},i})^2 \le 6 \cdot \Delta_{\text{pos}}^2(\mathcal{S}_1, \mathcal{S}_2)

$$
3.  **Bound the Structural Component:** From [](#sub-lem-stable-structural-error-bound), the total squared structural error component is bounded by:

$$

\sum_{i \in \mathcal{A}_{\text{stable}}} (\Delta_{\text{struct},i})^2 \le \frac{4 k_1 D_{\mathcal{Y}}^2}{(k_1 - 1)^2} \cdot n_c(\mathcal{S}_1, \mathcal{S}_2)^2

$$
4.  **Combine the Bounds:** Substituting the bounds from steps 2 and 3 into the inequality from step 1 and multiplying by the factor of 2 yields the final result stated in the lemma.
**Q.E.D.**
:::
##### 10.2.6.1. Sub-Lemma: Decomposition of Stable Walker ({prf:ref}`def-walker`) Error

:::{prf:lemma} Decomposition of Stable Walker Error
:label: lem-sub-stable-walker-error-decomposition

This lemma decomposes the error for stable walkers using {prf:ref}`lem-single-walker-positional-error` and {prf:ref}`lem-single-walker-structural-error`, supporting {prf:ref}`thm-distance-operator-mean-square-continuity`.

This lemma decomposes the stable walker error using {prf:ref}`lem-single-walker-positional-error` and {prf:ref}`lem-single-walker-structural-error`.

For ({prf:ref}`def-alive-dead-sets`) ({prf:ref}`def-swarm-and-state-space`) each stable walker ({prf:ref}`def-walker`) $i \in \mathcal{A}_{\text{stable}}$, the error in its expected raw distance can be decomposed into a positional error term, $\Delta_{\text{pos},i}$, and a structural error term, $\Delta_{\text{struct},i}$.
The total squared error over the set of stable walker ({prf:ref}`def-walker`)s is bounded by twice the sum of the squared norms of these two error components:

$$

\sum_{i \in \mathcal{A}_{\text{stable}}} |\mathbb{E}[d_i(\mathcal{S}_1)] - \mathbb{E}[d_i(\mathcal{S}_2)]|^2 \le 2 \sum_{i \in \mathcal{A}_{\text{stable}}} (\Delta_{\text{pos},i})^2 + 2 \sum_{i \in \mathcal{A}_{\text{stable}}} (\Delta_{\text{struct},i})^2

$$
:::

:::{prf:proof}
:label: proof-lem-sub-stable-walker-error-decomposition
**Proof.**
1.  **Decompose Single-Walker ({prf:ref}`def-walker`) Error:** For each stable walker $i \in \mathcal{A}_{\text{stable}}$, we introduce the intermediate term $\mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} [d_{\text{alg}}(x_{2,i}, x_{2,c})]$ and apply the triangle inequality:

$$

|\mathbb{E}[d_i(\mathcal{S}_1)] - \mathbb{E}[d_i(\mathcal{S}_2)]| \le \underbrace{\left| \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\text{alg}}(x_{1,i}, x_{1,c}) \right] - \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\text{alg}}(x_{2,i}, x_{2,c}) \right] \right|}_{\Delta_{\text{pos},i}} + \underbrace{\left| \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\text{alg}}(x_{2,i}, x_{2,c}) \right] - \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_2)} \left[ d_{\text{alg}}(x_{2,i}, x_{2,c}) \right] \right|}_{\Delta_{\text{struct},i}}

$$
The term $\Delta_{\text{pos},i}$ is the error from positional change over a fixed companion set, bounded by [](#lem-single-walker ({prf:ref}`def-walker`)-positional-error). The term $\Delta_{\text{struct},i}$ is the error from structural change with fixed positions, bounded by [](#lem-single-walker-structural-error).
2.  **Bound the Squared Sum:** Using the elementary inequality $(a+b)^2 \le 2a^2 + 2b^2$, we can bound the square of the single-walker ({prf:ref}`def-walker`) error. Summing over all $i \in \mathcal{A}_{\text{stable}}$ yields the inequality stated in the lemma.
**Q.E.D.**
:::
##### 10.2.6.2. Sub-Lemma: Bounding the Positional Error Component

:::{prf:lemma} Bounding the Positional Error Component
:label: lem-sub-stable-positional-error-bound

The total squared error arising from positional changes for stable walker ({prf:ref}`def-walker`)s is bounded by the total positional displacement of all walkers in the swarm ({prf:ref}`def-swarm-and-state-space`).

$$

\sum_{i \in \mathcal{A}_{\text{stable}}} (\Delta_{\text{pos},i})^2 \le 6 \cdot \Delta_{\text{pos}}^2(\mathcal{S}_1, \mathcal{S}_2)

$$
:::

:::{prf:proof}
:label: proof-lem-sub-stable-positional-error-bound
**Proof.**
1.  **Bound the Single-Walker ({prf:ref}`def-walker`) Squared Error:** We start with the bound on $\Delta_{\text{pos},i}$ from [](#lem-single-walker-positional-error) and apply the inequality $(a+b)^2 \le 2a^2 + 2b^2$:

$$

(\Delta_{\text{pos},i})^2 \le 2 \cdot d_{\text{alg}}(x_{1,i}, x_{2,i})^2 + 2 \cdot \left( \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\text{alg}}(x_{1,c}, x_{2,c}) \right] \right)^2

$$
2.  **Apply Jensen's Inequality:** For the second term, we apply Jensen's inequality, $(\mathbb{E}[X])^2 \le \mathbb{E}[X^2]$, to move the square inside the expectation:

$$

(\Delta_{\text{pos},i})^2 \le 2 \cdot d_{\text{alg}}(x_{1,i}, x_{2,i})^2 + 2 \cdot \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\text{alg}}(x_{1,c}, x_{2,c})^2 \right]

$$
3.  **Sum Over All Stable Walker ({prf:ref}`def-walker`)s:** We sum this inequality over all $i \in \mathcal{A}_{\text{stable}}$.

$$

\sum_{i \in \mathcal{A}_{\text{stable}}} (\Delta_{\text{pos},i})^2 \le 2 \sum_{i \in \mathcal{A}_{\text{stable}}} d_{\text{alg}}(x_{1,i}, x_{2,i})^2 + 2 \sum_{i \in \mathcal{A}_{\text{stable}}} \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\text{alg}}(x_{1,c}, x_{2,c})^2 \right]

$$
4.  **Analyze the Second Term's Double Summation:** The second term is a double summation. We expand the expectation:

$$

2 \sum_{i \in \mathcal{A}_{\text{stable}}} \left( \frac{1}{|\mathcal{A}_1 \setminus \{i\}|} \sum_{c \in \mathcal{A}_1 \setminus \{i\}} d_{\text{alg}}(x_{1,c}, x_{2,c})^2 \right)

$$
Consider a specific squared distance term $d_{\text{alg}}(x_{1,j}, x_{2,j})^2$ where $j \in \mathcal{A}_1$. This term appears in the inner sum for every $i \in \mathcal{A}_{\text{stable}}$ such that $i \neq j$. The number of such appearances is $|\mathcal{A}_{\text{stable}} \setminus \{j\}|$, which is bounded above by $|\mathcal{A}_{\text{stable}}|$. The normalization factor is $\frac{1}{k_1-1}$. Therefore, the double summation is bounded by:

$$

\le \frac{2}{k_1 - 1} \sum_{i \in \mathcal{A}_{\text{stable}}} \sum_{c \in \mathcal{A}_1 \setminus \{i\}} d_{\text{alg}}(x_{1,c}, x_{2,c})^2 \le \frac{2|\mathcal{A}_{\text{stable}}|}{k_1 - 1} \sum_{j \in \mathcal{A}_1} d_{\text{alg}}(x_{1,j}, x_{2,j})^2

$$
Since $|\mathcal{A}_{\text{stable}}| \le k_1$, and for $k_1 \ge 2$, the fraction $k_1/(k_1-1) \le 2$, the entire second term from Step 3 is bounded by $4 \sum_{j \in \mathcal{A}_1} d_{\text{alg}}(x_{1,j}, x_{2,j})^2$.
5.  **Combine and Finalize:** Substituting this back into the inequality from Step 3:

$$

\sum_{i \in \mathcal{A}_{\text{stable}}} (\Delta_{\text{pos},i})^2 \le 2 \sum_{i \in \mathcal{A}_{\text{stable}}} d_{\text{alg}}(x_{1,i}, x_{2,i})^2 + 4 \sum_{j \in \mathcal{A}_1} d_{\text{alg}}(x_{1,j}, x_{2,j})^2

$$
Both sums can be bounded by the sum over all $N$ walkers, which is the definition of $\Delta_{\text{pos}}^2$:

$$

\sum_{i \in \mathcal{A}_{\text{stable}}} (\Delta_{\text{pos},i})^2 \le 6 \cdot \Delta_{\text{pos}}^2(\mathcal{S}_1, \mathcal{S}_2)

$$
**Q.E.D.**
:::
##### 10.2.6.3. Sub-Lemma: Bounding the Structural Error Component for Stable Walkers

:::{prf:lemma} Bounding the Structural Error Component for Stable Walkers
:label: lem-sub-stable-structural-error-bound

Let $\mathcal{S}_1$ and $\mathcal{S}_2$ be two swarm ({prf:ref}`def-swarm-and-state-space`) states with $|\mathcal{A}(\mathcal{S}_1)| = k_1 \ge 2$ ({prf:ref}`def-algorithmic-space-generic`) ({prf:ref}`def-alive-dead-sets`) ({prf:ref}`def-swarm-and-state-space`). Let $\mathcal{A}_{\text{stable}} = \mathcal{A}(\mathcal{S}_1) \cap \mathcal{A}(\mathcal{S}_2)$ be the set of stable walkers, and let $\Delta_{\text{struct},i}$ be the error in a single walker ({prf:ref}`def-walker`)'s expected distance due to structural change.
The total squared error arising from structural changes for stable walker ({prf:ref}`def-walker`)s is bounded by the square of the total number of status changes in the swarm.

$$

\sum_{i \in \mathcal{A}_{\text{stable}}} (\Delta_{\text{struct},i})^2 \le \frac{4 k_1 D_{\mathcal{Y}}^2}{(k_1 - 1)^2} \cdot n_c(\mathcal{S}_1, \mathcal{S}_2)^2

$$
:::

:::{prf:proof}
:label: proof-lem-sub-stable-structural-error-bound
**Proof.**
The proof proceeds by taking the established bound for the structural error of a single stable walker ({prf:ref}`def-walker`) and summing its square over all stable walkers.
1.  **Bound the Single-Walker ({prf:ref}`def-walker`) Squared Error:** We start with the bound on the structural error component for a single walker $i$, $\Delta_{\text{struct},i}$, as established in [](#lem-single-walker-structural-error). The bound is:

$$

|\Delta_{\text{struct},i}| \le \frac{2 D_{\mathcal{Y}}}{k_1 - 1} \cdot n_c(\mathcal{S}_1, \mathcal{S}_2)

$$
Squaring this expression provides a deterministic bound for the squared error of a single stable walker ({prf:ref}`def-walker`):

$$

(\Delta_{\text{struct},i})^2 \le \frac{4 D_{\mathcal{Y}}^2}{(k_1 - 1)^2} \cdot n_c(\mathcal{S}_1, \mathcal{S}_2)^2

$$
2.  **Sum Over All Stable Walker ({prf:ref}`def-walker`)s:** We sum this inequality over all stable walkers $i \in \mathcal{A}_{\text{stable}}$. Since the derived bound is identical for every stable walker, we multiply the single-walker bound by the number of stable walkers, $|\mathcal{A}_{\text{stable}}|$.

$$

\sum_{i \in \mathcal{A}_{\text{stable}}} (\Delta_{\text{struct},i})^2 \le |\mathcal{A}_{\text{stable}}| \cdot \frac{4 D_{\mathcal{Y}}^2}{(k_1 - 1)^2} \cdot n_c(\mathcal{S}_1, \mathcal{S}_2)^2

$$
3.  **Finalize:** Using the fact that the number of stable walker ({prf:ref}`def-walker`)s is bounded by the initial number of alive walkers, $|\mathcal{A}_{\text{stable}}| \le |\mathcal{A}(\mathcal{S}_1)| = k_1$, we arrive at the final bound stated in the sub-lemma.
**Q.E.D.**
:::
#### 10.2.5 Lemma: Bound on the Total Squared Error for Unstable Walker ({prf:ref}`def-walker`)s
:::{prf:proof}
:label: proof-line-2408
**Proof.**
1.  **Analyze a Single Unstable Walker ({prf:ref}`def-walker`):** Let $i$ be an unstable walker, meaning its status $s_i$ changes. From [](#lem-single-walker-own-status-error), the absolute error in its expected distance is bounded by $D_{\mathcal{Y}}$ ({prf:ref}`axiom-bounded-algorithmic ({prf:ref}`def-algorithmic-space-generic`)-diameter`). Therefore, the squared error for any single unstable walker is bounded by $D_{\mathcal{Y}}^2$.
2.  **Sum Over All Unstable Walker ({prf:ref}`def-walker`)s:** The set of unstable walkers, $\mathcal{A}_{\text{unstable}}$, is precisely the set of indices where $s_{1,i} \neq s_{2,i}$. The number of walkers in this set is $|\mathcal{A}_{\text{unstable}}| = \sum_{j=1}^N (s_{1,j} - s_{2,j})^2$, since $(s_{1,j} - s_{2,j})^2$ is 1 if the status changes and 0 otherwise.
3.  **Combine and Finalize:** The total squared error from unstable walker ({prf:ref}`def-walker`)s is the sum of their individual squared errors. Since each is bounded by $D_{\mathcal{Y}}^2$, the total sum is bounded by the number of such walkers multiplied by this bound:

$$

    \sum_{i \in \mathcal{A}_{\text{unstable}}} |\dots|^2 \le |\mathcal{A}_{\text{unstable}}| \cdot D_{\mathcal{Y}}^2 = D_{\mathcal{Y}}^2 \sum_{j=1}^N (s_{1,j} - s_{2,j})^2

$$
**Q.E.D.**
:::
#### 10.2.6 Lemma: Bound on the Total Squared Error for Stable Walkers
:::{prf:proof}
:label: proof-line-2422
**Proof.**
The total error for a single stable walker ({prf:ref}`def-walker`) is first decomposed into a positional component and a structural component. The squared L2-norm of the total error over all stable walkers is then bounded by combining the established bounds for the sum of the squares of these individual components.
1.  **Decomposition of Total Stable Error:** From [](#sub-lem-stable-walker-error-decomposition), the total squared error for stable walkers is bounded by twice the sum of the squared positional and structural error components:

$$

    \sum_{i \in \mathcal{A}_{\text{stable}}} |\mathbb{E}[d_i(\mathcal{S}_1)] - \mathbb{E}[d_i(\mathcal{S}_2)]|^2 \le 2 \sum_{i \in \mathcal{A}_{\text{stable}}} (\Delta_{\text{pos},i})^2 + 2 \sum_{i \in \mathcal{A}_{\text{stable}}} (\Delta_{\text{struct},i})^2

$$
2.  **Bound the Positional Component:** From [](#sub-lem-stable-positional-error-bound), the total squared positional error component is bounded by:

$$

    \sum_{i \in \mathcal{A}_{\text{stable}}} (\Delta_{\text{pos},i})^2 \le 6 \sum_{j=1}^N d_{\text{alg}}(x_{1,j}, x_{2,j})^2

$$
3.  **Bound the Structural Component:** From [](#sub-lem-stable-structural-error-bound), the total squared structural error component is bounded by:

$$

    \sum_{i \in \mathcal{A}_{\text{stable}}} (\Delta_{\text{struct},i})^2 \le \frac{4 k_1 D_{\mathcal{Y}}^2}{(k_1 - 1)^2} \left( \sum_{j=1}^N (s_{1,j} - s_{2,j})^2 \right)^2

$$
4.  **Combine the Bounds:** Substituting the bounds from steps 2 and 3 into the inequality from step 1 and multiplying by the factor of 2 yields the final result stated in the lemma.
**Q.E.D.**
:::
##### 10.2.6.1 Sub-Lemma: Decomposition of Stable Walker Error
:::{prf:proof}
:label: proof-line-2450
**Proof.**
1.  **Decompose Single-Walker Error:** For each stable walker ({prf:ref}`def-walker`) $i \in \mathcal{A}_{\text{stable}}$, we introduce the intermediate term $\mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} [d_{\text{alg}}(x_{2,i}, x_{2,c})]$ and apply the triangle inequality:

$$

    |\mathbb{E}[d_i(\mathcal{S}_1)] - \mathbb{E}[d_i(\mathcal{S}_2)]| \le \underbrace{\left| \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\text{alg}}(x_{1,i}, x_{1,c}) \right] - \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\text{alg}}(x_{2,i}, x_{2,c}) \right] \right|}_{\Delta_{\text{pos},i}} + \underbrace{\left| \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\text{alg}}(x_{2,i}, x_{2,c}) \right] - \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_2)} \left[ d_{\text{alg}}(x_{2,i}, x_{2,c}) \right] \right|}_{\Delta_{\text{struct},i}}

$$
The term $\Delta_{\text{pos},i}$ is the error from positional change over a fixed companion set, bounded by [](#lem-single-walker-positional-error). The term $\Delta_{\text{struct},i}$ is the error from structural change with fixed positions, bounded by [](#lem-single-walker-structural-error).
2.  **Bound the Squared Sum:** Using the elementary inequality $(a+b)^2 \le 2a^2 + 2b^2$, we can bound the square of the single-walker error. Summing over all $i \in \mathcal{A}_{\text{stable}}$ yields the inequality stated in the lemma.
**Q.E.D.**
:::
##### 10.2.6.2 Sub-Lemma: Bounding the Positional Error Component
:::{prf:proof}
:label: proof-line-2464
**Proof.**
1.  **Bound the Single-Walker Squared Error:** We start with the bound on $\Delta_{\text{pos},i}$ from [](#lem-single-walker ({prf:ref}`def-walker`)-positional-error) and apply the inequality $(a+b)^2 \le 2a^2 + 2b^2$:

$$

    (\Delta_{\text{pos},i})^2 \le 2 \cdot d_{\text{alg}}(x_{1,i}, x_{2,i})^2 + 2 \cdot \left( \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\text{alg}}(x_{1,c}, x_{2,c}) \right] \right)^2

$$
2.  **Apply Jensen's Inequality:** For the second term, we apply Jensen's inequality, $(\mathbb{E}[X])^2 \le \mathbb{E}[X^2]$, to move the square inside the expectation:

$$

    (\Delta_{\text{pos},i})^2 \le 2 \cdot d_{\text{alg}}(x_{1,i}, x_{2,i})^2 + 2 \cdot \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\text{alg}}(x_{1,c}, x_{2,c})^2 \right]

$$
3.  **Sum Over All Stable Walkers:** We sum this inequality over all $i \in \mathcal{A}_{\text{stable}}$.

$$

    \sum_{i \in \mathcal{A}_{\text{stable}}} (\Delta_{\text{pos},i})^2 \le 2 \sum_{i \in \mathcal{A}_{\text{stable}}} d_{\text{alg}}(x_{1,i}, x_{2,i})^2 + 2 \sum_{i \in \mathcal{A}_{\text{stable}}} \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\text{alg}}(x_{1,c}, x_{2,c})^2 \right]

$$
4.  **Analyze the Second Term's Double Summation:** The second term is a double summation. We expand the expectation:

$$

    2 \sum_{i \in \mathcal{A}_{\text{stable}}} \left( \frac{1}{k_1 - 1} \sum_{c \in \mathcal{A}_1 \setminus \{i\}} d_{\text{alg}}(x_{1,c}, x_{2,c})^2 \right) = \frac{2}{k_1 - 1} \sum_{i \in \mathcal{A}_{\text{stable}}} \sum_{c \in \mathcal{A}_1 \setminus \{i\}} d_{\text{alg}}(x_{1,c}, x_{2,c})^2

$$
Consider a specific squared distance term $d_{\text{alg}}(x_{1,j}, x_{2,j})^2$ where $j \in \mathcal{A}_1$. This term appears in the inner sum for every $i \in \mathcal{A}_{\text{stable}}$ such that $i \neq j$. The number of such appearances is $|\mathcal{A}_{\text{stable}} \setminus \{j\}|$, which is bounded above by $|\mathcal{A}_{\text{stable}}|$. Therefore, the double summation is bounded by:

$$

    \sum_{i \in \mathcal{A}_{\text{stable}}} \sum_{c \in \mathcal{A}_1 \setminus \{i\}} (\dots)^2 \le |\mathcal{A}_{\text{stable}}| \sum_{j \in \mathcal{A}_1} d_{\text{alg}}(x_{1,j}, x_{2,j})^2

$$
Since $|\mathcal{A}_{\text{stable}}| \le k_1$, the entire second term from Step 3 is bounded by:

$$

    \frac{2}{k_1 - 1} \cdot k_1 \sum_{j \in \mathcal{A}_1} d_{\text{alg}}(x_{1,j}, x_{2,j})^2

$$
For $k_1 \ge 2$, the fraction $k_1/(k_1-1) \le 2$. The term is therefore bounded by $4 \sum_{j \in \mathcal{A}_1} d_{\text{alg}}(x_{1,j}, x_{2,j})^2$.
5.  **Combine and Finalize:** Substituting this back into the inequality from Step 3:

$$

    \sum_{i \in \mathcal{A}_{\text{stable}}} (\Delta_{\text{pos},i})^2 \le 2 \sum_{i \in \mathcal{A}_{\text{stable}}} d_{\text{alg}}(x_{1,i}, x_{2,i})^2 + 4 \sum_{j \in \mathcal{A}_1} d_{\text{alg}}(x_{1,j}, x_{2,j})^2

$$
Both sums can be bounded by the sum over all $N$ walkers, giving the final result:

$$

    \sum_{i \in \mathcal{A}_{\text{stable}}} (\Delta_{\text{pos},i})^2 \le 6 \sum_{j=1}^N d_{\text{alg}}(x_{1,j}, x_{2,j})^2

$$
**Q.E.D.**
:::
##### 10.2.6.3 Sub-Lemma: Bounding the Structural Error Component for Stable Walkers
:::{prf:proof}
:label: proof-line-2526
**Proof.**
The proof proceeds by taking the established bound for the structural error of a single stable walker ({prf:ref}`def-walker`) and summing its square over all stable walkers in the set $\mathcal{A}_{\text{stable}}$.
1.  **Bound the Single-Walker Squared Error:** We start with the bound on the structural error component for a single walker $i$, $\Delta_{\text{struct},i}$, as established in [](#lem-single-walker-structural-error). Let $n_c = \sum_{j=1}^N (s_{1,j} - s_{2,j})^2$ be the total number of status changes. The bound from [](#lem-single-walker-structural-error) is:

$$

    |\Delta_{\text{struct},i}| \le \frac{2 D_{\mathcal{Y}}}{k_1 - 1} \cdot n_c

$$
Squaring this expression provides a deterministic bound for the squared error of a single stable walker:

$$

    (\Delta_{\text{struct},i})^2 \le \frac{4 D_{\mathcal{Y}}^2}{(k_1 - 1)^2} \cdot n_c^2

$$
2.  **Sum Over All Stable Walkers:** We sum this inequality over all stable walkers $i \in \mathcal{A}_{\text{stable}}$. Since the derived bound is identical for every stable walker and does not depend on the index $i$, we multiply the single-walker bound by the number of stable walkers, $|\mathcal{A}_{\text{stable}}|$.

$$

    \sum_{i \in \mathcal{A}_{\text{stable}}} (\Delta_{\text{struct},i})^2 \le |\mathcal{A}_{\text{stable}}| \cdot \frac{4 D_{\mathcal{Y}}^2}{(k_1 - 1)^2} \cdot n_c^2

$$
3.  **Finalize:** Using the fact that the number of stable walkers is bounded by the initial number of alive walkers, $|\mathcal{A}_{\text{stable}}| \le |\mathcal{A}(\mathcal{S}_1)| = k_1$, and substituting the definition of $n_c^2$, we arrive at the final bound stated in the sub-lemma.

$$

    \sum_{i \in \mathcal{A}_{\text{stable}}} (\Delta_{\text{struct},i})^2 \le \frac{4 k_1 D_{\mathcal{Y}}^2}{(k_1 - 1)^2} \cdot n_c^2 = \frac{4 k_1 D_{\mathcal{Y}}^2}{(k_1 - 1)^2} \left( \sum_{j=1}^N (s_{1,j} - s_{2,j})^2 \right)^2

$$
**Q.E.D.**
:::
#### 10.2.7 Bound on the Expected Raw Distance Vector Change
This theorem consolidates the error bounds from the preceding lemmas to establish a deterministic bound on the change of the *expected* raw distance vector. This bound is expressed as a function of the formal displacement components ([](#def-displacement-components)) and serves as the deterministic part of the continuity axiom for the distance operator.
:::{prf:theorem} Bound on the Expected Raw Distance Vector Change
:label: thm-expected-raw-distance-bound
Let $\mathcal{S}_1$ and $\mathcal{S}_2$ be two swarm ({prf:ref}`def-swarm-and-state-space`) states, with $|\mathcal{A}(\mathcal{S}_1)| = k_1 \ge 2$ ({prf:ref}`def-algorithmic-space-generic`) ({prf:ref}`def-alive-dead-sets`) ({prf:ref}`def-swarm-and-state-space`). Let $\mathbb{E}[\mathbf{d}(\mathcal{S})]$ be the $N$-dimensional vector of expected raw distances.
The squared Euclidean distance between the expected raw distance vectors of the two swarms is deterministically bounded by a function of the displacement component ({prf:ref}`def-displacement-components`)mathbf{d}(\mathcal{S}_1)] - \mathbb{E}[\mathbf{d}(\mathcal{S}_2)] \|_2^2 \le C_{\text{pos},d} \cdot \Delta_{\text{pos}}^2(\mathcal{S}_1, \mathcal{S}_2) + C_{\text{status},d}^{(1)} \cdot n_c(\mathcal{S}_1, \mathcal{S}_2) + C_{\text{status},d}^{(2)}(k_1) \cdot n_c^2(\mathcal{S}_1, \mathcal{S}_2)

$$
where the **Expected Distance Error Coefficients** are defined as:
*   $C_{\text{pos},d} := 12$
*   $C_{\text{status},d}^{(1)} := D_{\mathcal{Y}}^2$
*   $C_{\text{status},d}^{(2)}(k_1) := \frac{8 k_1 D_{\mathcal{Y}}^2}{(k_1 - 1)^2}$
:::
:::{prf:proof}
:label: proof-thm-expected-raw-distance-bound
**Proof.**
The proof is a direct consequence of decomposing the total error and applying the bounds established in the preceding lemmas.
1.  **Decomposition of Total Error:** Following [](#thm-total-expected-distance-error-decomposition), the total squared error is the sum of the error from the set of stable walker ({prf:ref}`def-walker`)s ($E^2_{\text{stable}}$) and the set of unstable walkers ($E^2_{\text{unstable}}$).
2.  **Bound Error Components:**
    *   The error from unstable walker ({prf:ref}`def-walker`)s, $E^2_{\text{unstable}}$, is bounded by [](#lem-total-squared-error-unstable): $E^2_{\text{unstable}} \le D_{\mathcal{Y}}^2 \cdot n_c$.
    *   The error from stable walker ({prf:ref}`def-walker`)s, $E^2_{\text{stable}}$, is bounded by [](#lem-total-squared-error-stable): $E^2_{\text{stable}} \le 12 \cdot \Delta_{\text{pos}}^2 + \frac{8 k_1 D_{\mathcal{Y}}^2}{(k_1 - 1)^2} \cdot n_c^2$.
3.  **Combine Bounds:** Summing the two bounds gives the final inequality. This theorem recasts that result by explicitly naming the coefficients for each displacement component, formalizing the bound for use in subsequent proofs.
**Q.E.D.**
:::
#### 10.2.8 Discontinuous Behavior of the Expected Raw Distance Vector at $k=1$
The continuity analysis in the preceding sections is expressly valid for the $k \geq 2$ regime. When the number of alive walker ({prf:ref}`def-walker`)s becomes one, the diversity measurement mechanism behaves in a fundamentally different, deterministic manner. The following theorem formally describes this behavior.
:::{prf:theorem} Deterministic Behavior of the Expected Raw Distance Vector at $k=1$
:label: thm-expected-raw-distance-k1
Let $\mathcal{S}$ be a swarm ({prf:ref}`def-swarm-and-state-space`) state with exactly one alive ({prf:ref}`def-alive-dead-sets`) walker ({prf:ref}`def-walker`), $|\mathcal{A}(\mathcal{S})| = 1$. The N-dimensional vector of expected raw distances, $\mathbb{E}[\mathbf{d}(\mathcal{S})]$, is deterministically the zero vector.

$$

|\mathcal{A}(\mathcal{S})| = 1 \implies \mathbb{E}[\mathbf{d}(\mathcal{S})] = \mathbf{0}

$$
**Implication for Continuity:**
Any transition between a state $\mathcal{S}_1$ with $|\mathcal{A}(\mathcal{S}_1)| \ge 2$ and a state $\mathcal{S}_2$ with $|\mathcal{A}(\mathcal{S}_2)| = 1$ induces a discontinuous change in the expected raw distance vector. The magnitude of this change is not governed by the Lipschitz bounds derived for the $k \geq 2$ regime, but is instead given by the norm of the vector in the $k \geq 2$ state:

$$

\| \mathbb{E}[\mathbf{d}(\mathcal{S}_1)] - \mathbb{E}[\mathbf{d}(\mathcal{S}_2)] \|_2^2 = \| \mathbb{E}[\mathbf{d}(\mathcal{S}_1)] \|_2^2

$$
:::
:::{prf:proof}
:label: proof-thm-expected-raw-distance-k1
**Proof.**
The proof follows directly from the definitions of the Raw Value Operator ({prf:ref}`def-raw-value-operator`) for distance and the Companion Selection Measure for the $k=1$ case. Let $\mathcal{S}$ be a swarm ({prf:ref}`def-swarm-and-state-space`) with $|\mathcal{A}(\mathcal{S})| = 1$, and let the single survivor be walker ({prf:ref}`def-walker`) $j$.
1.  **Expected Distance for the Survivor (Walker ({prf:ref}`def-walker`) **j**):**
    *   From the **Companion Selection Measure ({prf:ref}`def-companion-selection-measure`) ([](#def-companion-selection-measure))**, if a walker ({prf:ref}`def-walker`) is the only one alive, it is its own companion. Thus, the companion index is deterministically $c(j) = j$.
    *   The expected distance for walker ({prf:ref}`def-walker`) $j$ is the expectation over a single outcome:

$$

        \mathbb{E}[d_j(\mathcal{S})] = d_{\text{alg}}(x_j, x_j) = 0

$$
This holds because $d_{\text{alg}}$ is a metric, for which the distance from a point to itself is zero.
2.  **Expected Distance for Dead Walker ({prf:ref}`def-walker`)s (all **i ≠ j**):**
    *   From the definition of the **Raw Value Operator ({prf:ref}`def-raw-value-operator`) ([](#def-raw-value-operator))**, the raw value for any walker ({prf:ref}`def-walker`) that is not in the alive set ({prf:ref}`def-alive-dead-sets`) is deterministically zero.
    *   Therefore, for any dead walker ({prf:ref}`def-walker`) $i \in \mathcal{D}(\mathcal{S})$, its expected distance is $\mathbb{E}[d_i(\mathcal{S})] = 0$.
3.  **Conclusion:**
    Since the expected distance is zero for the single alive walker ({prf:ref}`def-walker`) and for all dead walkers, every component of the N-dimensional vector $\mathbb{E}[\mathbf{d}(\mathcal{S})]$ is zero. This proves that the vector is deterministically the zero vector when $k=1$.
    The implication for continuity follows directly. For a transition from $\mathcal{S}_1$ ($k_1 \ge 2$) to $\mathcal{S}_2$ ($k_2=1$), the change is $\| \mathbb{E}[\mathbf{d}(\mathcal{S}_1)] - \mathbf{0} \|_2^2$, which is not described by a continuous function of the displacement between the states but represents a discrete jump. This special case is handled by the revival dynamics of the algorithm rather than the continuity framework.
**Q.E.D.**
:::
### 10.3 Mean-Square Continuity of the Sampled Raw Distance Vector
This section serves to formally prove that the **Distance-to-Companion Measurement** operator satisfies the **Axiom of Mean-Square Continuity for Raw Values ([](#axiom-raw-value-mean-square-continuity))**. We achieve this by first demonstrating that the operator complies with the **Axiom of Bounded Measurement Variance ([](#axiom-bounded-measurement-variance))** and then using this result to derive the explicit form of its **Expected Squared Value Error Bound**, $F_{d,ms}$.
#### 10.3.1 Theorem: The Distance Operator Satisfies the Bounded Variance Axiom
First, we must prove that our specific stochastic measurement process—the distance-to-companion operator—is compliant with the foundational axiom for bounded variance.
:::{prf:theorem} The Distance Operator Satisfies the Bounded Variance Axiom
:label: thm-distance-operator-satisfies-bounded-variance-axiom
This theorem validates that {prf:ref}`def-distance-to-companion-measurement` satisfies {prf:ref}`axiom-bounded-measurement-variance`.

The **Distance-to-Companion Measurement** operator ($V=d$) satisfies the **Axiom of Bounded Measurement Variance ([](#axiom-bounded-measurement-variance))**. Its maximum measurement variance is deterministically bounded by:

$$

\kappa^2_{\text{variance}} = N \cdot D_{\mathcal{Y}}^2

$$
where $N$ is the total number of walker ({prf:ref}`def-walker`)s and $D_{\mathcal{Y}}$ ({prf:ref}`axiom-bounded-algorithmic-diameter`) is the diameter of the algorithmic space ({prf:ref}`def-algorithmic-space-generic`).
:::
:::{prf:proof}
:label: proof-thm-distance-operator-satisfies-bounded-variance-axiom
**Proof.**
The proof proceeds by bounding the variance of each component of the N-dimensional raw distance vector.
1.  **Decomposition of Total Variance:**
    The axiom requires a bound on $\mathbb{E}[\|\mathbf{d} - \mathbb{E}[\mathbf{d}]\|_2^2]$. By linearity of expectation, this is:

$$

    \mathbb{E}\left[\sum_{i=1}^N (d_i - \mathbb{E}[d_i])^2\right] = \sum_{i=1}^N \mathbb{E}[(d_i - \mathbb{E}[d_i])^2] = \sum_{i=1}^N \operatorname{Var}(d_i)

$$
2.  **Bound the Variance of a Single Component:**
    We must bound the variance, $\operatorname{Var}(d_i)$, for each walker ({prf:ref}`def-walker`) $i$.
    *   **Case 1: Dead Walker ({prf:ref}`def-walker`).** If walker $i$ is dead, its raw distance is deterministically zero ($d_i=0$). Therefore, its variance is $\operatorname{Var}(d_i) = 0$.
    *   **Case 2: Alive Walker ({prf:ref}`def-walker`).** If walker $i$ is alive, its raw distance $d_i$ is a random variable. By definition, any distance measurement in the algorithmic space is bounded on the interval $[0, D_{\mathcal{Y}}]$. For any random variable $X$ bounded on an interval, its variance is bounded by $\operatorname{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 \le \mathbb{E}[X^2]$. Since $d_i \in [0, D_{\mathcal{Y}}]$, we have $d_i^2 \in [0, D_{\mathcal{Y}}^2]$. The expectation is therefore bounded by $\mathbb{E}[d_i^2] \le D_{\mathcal{Y}}^2$. Thus, for any alive walker, $\operatorname{Var}(d_i) \le D_{\mathcal{Y}}^2$.
3.  **Sum Over All Walker ({prf:ref}`def-walker`)s:**
    The total variance is the sum of the individual variances. Since each of the $N$ terms is bounded above by $D_{\mathcal{Y}}^2$, the sum is bounded by:

$$

    \sum_{i=1}^N \operatorname{Var}(d_i) \le N \cdot D_{\mathcal{Y}}^2

$$
This provides a uniform bound that holds for any swarm ({prf:ref}`def-swarm-and-state-space`) state $\mathcal{S}$. The axiom is therefore satisfied with $\kappa^2_{\text{variance}} = N \cdot D_{\mathcal{Y}}^2$.
**Q.E.D.**
:::
#### 10.3.2 Theorem: Mean-Square Continuity of the Distance Operator
With the variance of the distance operator now axiomatically bounded, we can establish its mean-square continuity. This theorem provides the deterministic function that bounds the expected squared error, which is the critical input for the rest of the stability analysis.
:::{prf:theorem} Mean-Square Continuity of the Distance Operator
:label: thm-distance-operator-mean-square-continuity
The **Distance-to-Companion Measurement** operator ($V=d$) is mean-square continuous for transitions in the $k \geq 2$ regime. For any two swarm ({prf:ref}`def-swarm-and-state-space`) states $\mathcal{S}_1$ and $\mathcal{S}_2$ with $|\mathcal{A}(\mathcal{S}_1)|=k_1 \ge 2$, the expected squared Euclidean distance between the sampled raw distance vectors is deterministically bounded by the function $F_{d,ms}$:

$$

\mathbb{E}[\|\mathbf{d}(\mathcal{S}_1) - \mathbf{d}(\mathcal{S}_2)\|_2^2] \le F_{d,ms}(\mathcal{S}_1, \mathcal{S}_2)

$$
where the **Expected Squared Distance Error Bound** is defined as:

$$

\boxed{
F_{d,ms}(\mathcal{S}_1, \mathcal{S}_2) := 6 N D_{\mathcal{Y}}^2 + 3 \left( C_{\text{pos},d} \cdot \Delta_{\text{pos}}^2 + C_{\text{status},d}^{(1)} \cdot n_c + C_{\text{status},d}^{(2)}(k_1) \cdot n_c^2 \right)
}

$$
and the coefficients $C_{\dots,d}$ are the deterministic **Expected Distance Error Coefficients** from [](#thm-distance-operator-mean-square-continuity).
With the explicit derivation of this function, we have formally proven that the Distance-to-Companion operator is a valid raw value ({prf:ref}`def-raw-value-operator`) operator that satisfies the **Axiom of Mean-Square Continuity for Raw Values**. This function will now be used as a direct input to the continuity analysis of the standardization operator ({prf:ref}`def-standardization-operator-n-dimensional`).
:::
:::{prf:proof}
:label: proof-thm-distance-operator-mean-square-continuity
**Proof.**
The proof bounds the total expected squared error by decomposing it into a stochastic variance component and a deterministic mean component. Let $\mathbf{d}_1 = \mathbf{d}(\mathcal{S}_1)$ and $\mathbf{d}_2 = \mathbf{d}(\mathcal{S}_2)$.
1.  **Decomposition of Total Error:**
    We introduce the expectation vectors $\mathbb{E}[\mathbf{d}_1]$ and $\mathbb{E}[\mathbf{d}_2]$ and use the inequality $\|A+B+C\|_2^2 \le 3(\|A\|_2^2 + \|B\|_2^2 + \|C\|_2^2)$.

$$

    \|\mathbf{d}_1 - \mathbf{d}_2\|_2^2 = \|(\mathbf{d}_1 - \mathbb{E}[\mathbf{d}_1]) + (\mathbb{E}[\mathbf{d}_1] - \mathbb{E}[\mathbf{d}_2]) - (\mathbf{d}_2 - \mathbb{E}[\mathbf{d}_2])\|_2^2

$$
$$
    \le 3\|\mathbf{d}_1 - \mathbb{E}[\mathbf{d}_1]\|_2^2 + 3\|\mathbb{E}[\mathbf{d}_1] - \mathbb{E}[\mathbf{d}_2]\|_2^2 + 3\|\mathbf{d}_2 - \mathbb{E}[\mathbf{d}_2]\|_2^2

    $$
2.  **Take the Expectation:**
    We take the expectation of both sides. By linearity of expectation, this gives:

$$

    \mathbb{E}[\|\mathbf{d}_1 - \mathbf{d}_2\|_2^2] \le 3\mathbb{E}[\|\mathbf{d}_1 - \mathbb{E}[\mathbf{d}_1]\|_2^2] + 3\mathbb{E}[\|\mathbb{E}[\mathbf{d}_1] - \mathbb{E}[\mathbf{d}_2]\|_2^2] + 3\mathbb{E}[\|\mathbf{d}_2 - \mathbb{E}[\mathbf{d}_2]\|_2^2]

$$
3.  **Bound the Components:**
    *   **Stochastic Variance Terms:** The first and third terms are bounded by the **Axiom of Bounded Measurement Variance**, which we have shown is satisfied by the distance operator in [](#thm-distance-operator-satisfies-bounded-variance-axiom) with $\kappa^2_{\text{variance}} = N D_{\mathcal{Y}}^2$. Therefore:
        *   $\mathbb{E}[\|\mathbf{d}_1 - \mathbb{E}[\mathbf{d}_1]\|_2^2] \le N D_{\mathcal{Y}}^2$
        *   $\mathbb{E}[\|\mathbf{d}_2 - \mathbb{E}[\mathbf{d}_2]\|_2^2] \le N D_{\mathcal{Y}}^2$
    *   **Deterministic Mean Term:** The middle term involves the squared norm of a deterministic vector difference, so the expectation has no effect. This term is bounded by the analysis in Section 10.3. From [](#thm-distance-operator-mean-square-continuity), we have:

$$

        \|\mathbb{E}[\mathbf{d}_1] - \mathbb{E}[\mathbf{d}_2]\|_2^2 \le C_{\text{pos},d} \cdot \Delta_{\text{pos}}^2 + C_{\text{status},d}^{(1)} \cdot n_c + C_{\text{status},d}^{(2)}(k_1) \cdot n_c^2

$$
4.  **Combine the Bounds:**
    Substituting the bounds from Step 3 into the inequality from Step 2 yields the final result:

$$

    \mathbb{E}[\|\mathbf{d}_1 - \mathbf{d}_2\|_2^2] \le 3(N D_{\mathcal{Y}}^2) + 3(\text{Bound from Thm 10.3.2}) + 3(N D_{\mathcal{Y}}^2)

$$
This simplifies to the expression for $F_{d,ms}(\mathcal{S}_1, \mathcal{S}_2)$ as stated in the theorem.
**Q.E.D.**
:::
## 12. Standardization pipeline
The core of the algorithm's decision-making process relies on transforming raw measurements into a common, standardized scale. This is handled by the standardization pipeline, a composite operator that maps a full swarm state $S$ to a corresponding N-dimensional standardized vector (a "Z-score" vector). The output is defined to be zero for any walker that is not in the alive set ({prf:ref}`def-alive-dead-sets`).
The stability of the entire algorithm is critically dependent on the continuity properties of this pipeline. This section provides a rigorous, self-contained proof of its probabilistic continuity, valid for any aggregation operator satisfying the axiomatic requirements of [](#def-swarm-aggregation-operator-axiomatic) and any raw value operator ({prf:ref}`def-raw-value-operator`) satisfying the continuity axiom of [](#axiom-raw-value-mean-square-continuity).
:::{admonition} Notation freeze: $\sigma$ vs. $\sigma'$ (patched)
:class: note
- $\sigma$: noise scale on $\mathcal X$ used by perturbation kernels (and $\delta$ for cloning kernels). It never denotes a statistical deviation in this document.
- $\sigma'$: standard deviation associated with the aggregation measure on $\mathcal Y$ for the raw values.
- $\sigma\'_{\text{reg}}$: the patched version of $\sigma'$ used in the standardization operator; depends on $\varepsilon_{\text{std}}>0$ and satisfies the explicit global Lipschitz bound proved in [](#lem-sigma-reg-derivative-bounds). Define $\sigma'_{\min,\text{bound}}:=\sqrt{\kappa_{\text{var,min}}+\varepsilon_{\text{std}}^2}$.
This convention avoids symbol collisions and freezes the patched normalization used throughout.
:::
### 11.1 Definition of the standardization pipeline
#### 11.1.1 N-Dimensional Standardization Operator
:::{prf:definition} N-Dimensional Standardization Operator
:label: def-standardization-operator-n-dimensional
The **N-Dimensional Standardization Operator ({prf:ref}`def-standardization-operator-n-dimensional`)**, denoted $z$, is a function that maps a swarm ({prf:ref}`def-swarm-and-state-space`) state $S$ to an N-dimensional vector, parameterized by a choice of a raw value ({prf:ref}`def-raw-value-operator`) operator and an aggregation operator.
**Signature:** $z: \Sigma_N \times (\text{Raw Value Operator ({prf:ref}`def-raw-value-operator`)}) \times (\text{Swarm Aggregation Operator ({prf:ref}`def-swarm-aggregation-operator-axiomatic`)}) \to \mathbb{R}^N$
**Inputs:**
*   The current swarm state, $S_t$.
*   A **Raw Value Operator ({prf:ref}`def-raw-value-operator`)**, $V$ (per [](#def-raw-value-operator)).
*   A **Swarm Aggregation Operator ({prf:ref}`def-swarm-aggregation-operator-axiomatic`)**, $M$ (e.g., $R_{\text{agg}}$ or $M_D$, per [](#def-swarm-aggregation-operator-axiomatic)).
*   All relevant implicit parameters ($\varepsilon_{\text{std}}$).
**Operation:**
The operator computes the output vector $z = z(S_t, V, M)$ as follows:
1.  **Generate Raw Values:**
    a. Let $\mathcal{A}(S_t)$ be the set of alive walker ({prf:ref}`def-walker`)s. Let $k = |\mathcal{A}(S_t)|$. If $k=0$, return the zero vector and terminate.
    b. Generate a single stochastic sample of the raw value vector by drawing $v \sim V(S_t)$. Let $v_A$ be the k-dimensional sub-vector corresponding to the alive set ({prf:ref}`def-alive-dead-sets`).
2.  **Aggregate and Measure Statistics:**
    a. Form the **Swarm Aggregation Measure**: $\mu_v = M(S_t; v_A)$.
    b. Measure the statistical properties $(\mu_A, \sigma'_A)$ from $\mu_v$ (per Def. 11.1.2), using the provided $\varepsilon_{\text{std}}$ and the regularized standard deviation $\sigma\'_{\text{reg}}$.
3.  **Standardize and Assemble N-Dimensional Vector:**
    a. Initialize an N-dimensional zero vector, $z_{\text{out}} \leftarrow 0$.
    b. For each walker ({prf:ref}`def-walker`) $i \in \mathcal{A}(S_t)$:
        *   Compute its Z-score: $z_i := (v_i - \mu_A) / \sigma'_A$.
        *   Set the $i$-th component of the output vector: $z_{\text{out}}[i] := z_i$.
**Output:** The full N-dimensional standardized vector $z_{\text{out}}$.
:::
#### 11.1.2 Statistical Properties Measurement via the Regularized Standard Deviation Function
The first step is to distill the vector of raw measurements from the alive set ({prf:ref}`def-alive-dead-sets`) into a few key statistics. This is achieved by calculating the moments of the **Swarm Aggregation Measure**. The definition of the standard deviation is modified to use a smoothed function, which is a critical and mandatory component for securing a global Lipschitz control on the denominator and, ultimately, the Hölder continuity bounds for the full standardization operator ({prf:ref}`def-standardization-operator-n-dimensional`).
:::{prf:definition} Statistical Properties Measurement
:label: def-statistical-properties-measurement
Let $\mathbf{v}_{\mathcal{A}}$ ({prf:ref}`def-swarm-and-state-space`) be the vector of raw scalar values for the alive set ({prf:ref}`def-alive-dead-sets`) $\mathcal{A}_t$. Let $\mu_{\mathbf{v}} = M(\mathcal{S}_t; \mathbf{v}_{\mathcal{A}})$ be the **Swarm Aggregation Measure** for these values. The **Statistical Properties Measurement** extracts the effective mean and a smoothed, regularized standard deviation from this measure:
*   **Mean:** $\mu_{\mathcal{A}} := \mathbb{E}[\mu_{\mathbf{v}}]$
*   **Regularized Standard Deviation:** $\sigma'_{\mathcal{A}} := \sigma'_{\text{reg}}(\operatorname{Var}[\mu_{\mathbf{v}}])$
where $\sigma'_{\text{reg}}: \mathbb{R}_{\ge 0} \to \mathbb{R}_{>0}$ is the **Regularized Standard Deviation**. This $C^\infty$ replacement for the square-root prevents pathological sensitivity near zero variance while maintaining smooth behavior everywhere. It is defined as:

$$
\sigma'_{\text{reg}}(V) := \sqrt{V + \sigma'^2_{\min}}

$$

where $\sigma'_{\min} := \sqrt{\kappa_{\text{var,min}} + \varepsilon_{\text{std}}^2} > 0$ is the **regularization parameter** combining the variance floor threshold $\kappa_{\text{var,min}} > 0$ and the numerical stability parameter $\varepsilon_{\text{std}} > 0$.
**Properties:**
1. **C^∞ Regularity:** The function is infinitely differentiable on $[0, \infty)$ as a composition of smooth functions.
2. **Positive Lower Bound:** $\sigma'_{\text{reg}}(V) \ge \sigma'_{\min} > 0$ for all $V \ge 0$, preventing division by zero.
3. **Asymptotic Behavior:** For large $V \gg \sigma'^2_{\min}$, the regularized function closely approximates the natural square root: $\sigma'_{\text{reg}}(V) \approx \sqrt{V} + \frac{\sigma'^2_{\min}}{2\sqrt{V}}$.
4. **Monotonicity:** The function is strictly increasing, with $\sigma'_{\text{reg}}(0) = \sigma'_{\min}$ and $\lim_{V \to \infty} \sigma'_{\text{reg}}(V) = \infty$.

This regularized standardization operator ({prf:ref}`def-standardization-operator-n-dimensional`) is applied in {prf:ref}`02_euclidean_gas` for the patched standardization step that produces standardized reward and distance scores.
::{prf:lemma} Derivative Bounds for Regularized Standard Deviation
:label: lem-sigma-reg-derivative-bounds
The regularized standard deviation $\sigma'_{	ext{reg}}(V) = \sqrt{V + \sigma'^2_{\min}}$ has explicit derivative bounds for all orders. For the first three derivatives:

$$
\left|(\sigma'_{	ext{reg}})'(V)
ight| = rac{1}{2\sqrt{V + \sigma'^2_{\min}}} \le rac{1}{2\sigma'_{\min}} =: L_{\sigma'_{	ext{reg}}}

$$

$$
\left|(\sigma'_{	ext{reg}})''(V)
ight| = rac{1}{4(V + \sigma'^2_{\min})^{3/2}} \le rac{1}{4\sigma'^3_{\min}} =: L_{\sigma''_{	ext{reg}}}

$$

$$
\left|(\sigma'_{	ext{reg}})'''(V)
ight| = rac{3}{8(V + \sigma'^2_{\min})^{5/2}} \le rac{3}{8\sigma'^5_{\min}} =: L_{\sigma'''_{	ext{reg}}}

$$

General form: For the $n$-th derivative with $n \ge 1$,

$$
\left|(\sigma'_{	ext{reg}})^{(n)}(V)
ight| \le rac{(2n-1)!!}{2^n \sigma'^{(2n-1)}_{\min}}

$$

where $(2n-1)!! = 1 \cdot 3 \cdot 5 \cdots (2n-1)$ is the double factorial.

Referenced by {prf:ref}`def-fragile-gas-algorithm`.
:::
:::{prf:proof}
:label: proof-lem-sigma-reg-derivative-bounds
Direct computation of derivatives of $\sigma'_{	ext{reg}}(V) = (V + \sigma'^2_{\min})^{1/2}$:

$$
(\sigma'_{	ext{reg}})'(V) = rac{1}{2}(V + \sigma'^2_{\min})^{-1/2}

$$

$$
(\sigma'_{	ext{reg}})''(V) = -rac{1}{4}(V + \sigma'^2_{\min})^{-3/2}

$$

$$
(\sigma'_{	ext{reg}})'''(V) = rac{3}{8}(V + \sigma'^2_{\min})^{-5/2}

$$

Since $V \ge 0$, the maximum magnitude of each derivative occurs at $V = 0$, yielding the stated bounds. The general form follows from the pattern of alternating signs and double factorials in the $n$-th derivative of $(V + \sigma'^2_{\min})^{1/2}$.
**Q.E.D.**
:::
#### 11.1.3 Continuity of Statistical Properties
The stability of the standardization operator ({prf:ref}`def-standardization-operator-n-dimensional`) hinges on the continuity of the measured mean ($\mu_{\mathcal{A}}$) and the regularized standard deviation ($\sigma'_{\mathcal{A}}$). These properties are not fundamental axioms themselves but are instead consequences of the axiomatic properties of the chosen **Swarm Aggregation Operator ({prf:ref}`def-swarm-aggregation-operator-axiomatic`)** $M$ ([](#def-swarm-aggregation-operator-axiomatic)) and the Lipschitz continuity of the new **Regularized Standard Deviation Function** ([](#def-statistical-properties-measurement)). The following lemmas formally derive the continuity bounds for $\mu_{\mathcal{A}}$ and $\sigma'_{\mathcal{A}}$ with respect to both changes in the raw value vector and changes in the swarm's structure.
:::{prf:lemma} Value Continuity of Statistical Properties
:label: lem-stats-value-continuity
LetLipschitz ({prf:ref}`axiom-reward-regularity`)a fixed swarm ({prf:ref}`def-swarm-and-state-space`) state with alive set ({prf:ref}`def-alive-dead-sets`) $\mathcal{A}$ of size $k = |\mathcal{A}| \geq 1$. Let $\mathbf{v}_1$ anLipschitz ({prf:ref}`axiom-reward-regularity`)e two raw value ({prf:ref}`def-raw-value-operator`) with components bounded by $V_{\max}$. The mean $\mu(\mathcal{S}, \mathbf{v})$ and regularized standard deviation $\sigma'(\mathcal{S}, \mathbf{v})$ are Lipschitz continuous with respect to the raw value vector $\mathbf{v}$.

$$
|\mu(\mathcal{S}, \mathbf{v}_1) - \mu(\mathcal{S}, \mathbf{v}_2)| \le L_{\mu,M}(\mathcal{S}) \cdot \|\mathbf{v}_1 - \mathbf{v}_2\|_2

$$

$$
|\sigma'(\mathcal{S}, \mathbf{v}_1) - \sigma'(\mathcal{S}, \mathbf{v}_2)| \le L_{\sigma',M}(\mathcal{S}) \cdot \|\mathbf{v}_1 - \mathbf{v}_2\|_2

$$

where $L_{\mu,M}$ is the axiomatic value Lipschitz function for the mean from [](#def-swarm ({prf:ref}`def-swarm-and-state-space`)-aggregation-operator-axiomatic) (explicit expressions for the empirical aggregator ({prf:ref}`lem-empirical-aggregator-properties`) appear in [](#lem-empirical-aggregator-properties)), and $L_{\sigma',M}$ is the derived Lipschitz constant for the regularized standard deviation, given by:

$$
\boxed{
L_{\sigma',M}(\mathcal{S}) := L_{\sigma\'_{\text{reg}}} \cdot \left( L_{m_2,M}(\mathcal{S}) + 2V_{\max}L_{\mu,M}(\mathcal{S}) \right)
}

$$

and $L_{\sigma\'_{\text{reg}}} = \frac{1}{2\sigma'_{\min}}$ is the finite, global Lipschitz constant of the Regularized Standard Deviation Function from [](#lem-sigma-reg-derivative-bounds).

This value continuity lemma is applied in {prf:ref}`02_euclidean_gas` for bounding standardization error with respect to reward and distance value changes.
:::
:::{prf:proof}
:label: proof-lem-stats-value-continuity
**Proof.**
The bound for the mean $\mu$ is a direct application of the axiom in [](#def-swarm ({prf:ref}`def-swarm-and-state-space`)-aggregation-operator-axiomatic). The bound for $\sigma'$ is derived by composition. $\sigma'(\mathcal{S}, \mathbf{v})$ is the composition of the variance function $\text{Var}(\mathbf{v}) = m_2(\mathcal{S}, \mathbf{v}) - \mu(\mathcal{S}, \mathbf{v})^2$ and the smoothed function $\sigma\'_{\text{reg}}(V)$.
1.  **Lipschitz Constant of $\sigma\'_{\text{reg}}(V)$:** The function $\sigma\'_{\text{reg}}(V) = \sqrt{V + \sigma'^2_{\min}}$ is infinitely differentiable. Its first derivative is $(\sigma\'_{\text{reg}})'(V) = \frac{1}{2\sqrt{V + \sigma'^2_{\min}}}$, which is maximized at $V = 0$. Therefore, its global Lipschitz constant is $L_{\sigma\'_{\text{reg}}} = \frac{1}{2\sigma'_{\min}}$, a finite, positive constant.
2.  **Lipschitz Constant of the Variance:** The change in variance is $|\text{Var}(\mathbf{v}_1) - \text{Var}(\mathbf{v}_2)| = |(m_2(\mathbf{v}_1) - \mu(\mathbf{v}_1)^2)- (m_2(\mathbf{v}_2) - \mu(\mathbf{v}_2)^2)|$. By the triangle inequality, this is $\leq |m_2(\mathbf{v}_1) - m_2(\mathbf{v}_2)| + |\mu(\mathbf{v}_1)^2 - \mu(\mathbf{v}_2)^2|$.
    *   The first term is bounded by $L_{m_2,M}(\mathcal{S}) \|\mathbf{v}_1-\mathbf{v}_2\|_2$.
    *   The second term, $|\mu_1-\mu_2||\mu_1+\mu_2|$, is bounded by $(L_{\mu,M}(\mathcal{S})\|\mathbf{v}_1-\mathbf{v}_2\|_2)(2V_{\max})$.
    *   Thus, the Lipschitz constant for the variance, $L_{\text{Var}}$, is bounded by $L_{m_2,M}(\mathcal{S}) + 2V_{\max} L_{\mu,M}(\mathcal{S})$.
3.  **Chain Rule for Lipschitz Functions:** The Lipschitz constant of the composition is bounded by the product of the individual Lipschitz constants, $L(\sigma\'_{\text{reg}} \circ \text{Var}) \le L_{\sigma\'_{\text{reg}}} \cdot L_{\text{Var}}$, which yields the expression for $L_{\sigma',M}$.
**Q.E.D.**
:::
:::{prf:lemma} Structural Continuity of Statistical Properties
:label: lem-stats-structural-continuity
Lraw value ({prf:ref}`def-raw-value-operator`) a fixed raw value vector. Let $\mathcal{S}_1$ and $\mathcal{S}_2$ be two swarm ({prf:ref}`def-swarm-and-state-space`) states. The mean $\mu(\mathcal{S}, \mathbf{v})$ and regularized standard deviation $\sigma'(\mathcal{S}, \mathbf{v})$ are continuous with respect to changes in the swarm structure.

$$
|\mu(\mathcal{S}_1, \mathbf{v}) - \mu(\mathcal{S}_2, \mathbf{v})| \le L_{\mu,S}(\mathcal{S}_1, \mathcal{S}_2) \cdot \|\mathbf{s}_1 - \mathbf{s}_2\|_2^2

$$

$$
|\sigma'(\mathcal{S}_1, \mathbf{v}) - \sigma'(\mathcal{S}_2, \mathbf{v})| \le L_{\sigma',S}(\mathcal{S}_1, \mathcal{S}_2) \cdot \|\mathbf{s}_1 - \mathbf{s}_2\|_2^2

$$

where $L_{\mu,S}$ is the axiomatic structural continuity function for the mean from [](#def-swarm ({prf:ref}`def-swarm-and-state-space`)-aggregation-operator-axiomatic) (see [](#lem-empirical-aggregator-properties) for the empirical constants), and $L_{\sigma',S}$ is the derived structural continuity function for the regularized standard deviation, given by:

$$
\boxed{
L_{\sigma',S}(\mathcal{S}_1, \mathcal{S}_2) := L_{\sigma\'_{\text{reg}}} \cdot \left( L_{m_2,S}(\mathcal{S}_1, \mathcal{S}_2) + 2V_{\max}L_{\mu,S}(\mathcal{S}_1, \mathcal{S}_2) \right)
}

$$

This structural continuity lemma is applied in {prf:ref}`02_euclidean_gas` for analyzing standardization error with respect to walker ({prf:ref}`def-walker`) status changes.
:::
:::{prf:proof}
:label: proof-lem-stats-structural-continuity
**Proof.**
The proof is identical in structure to that of [](#lem-stats-value-continuity), but it uses the structural continuity functions ($L_{\mu,S}$, $L_{m_2,S}$) from the aggregator axiom instead of the value-based Lipschitz constants. The change in variance due to structure is first shown to be bounded by $(L_{m_2,S}(\mathcal{S}_1, \mathcal{S}_2) + 2V_{\max} L_{\mu,S}(\mathcal{S}_1, \mathcal{S}_2)) \|\mathbf{s}_1-\mathbf{s}_2\|_2^2$. This is then composed with the globally Lipschitz function $\sigma\'_{\text{reg}}(\cdot)$ (with Lipschitz constant $L_{\sigma\'_{\text{reg}}}$), yielding the final result for $L_{\sigma',S}$.
**Q.E.D.**
:::
#### 11.1.4 Theorem: General Bound on the Norm of the Standardized Vector
A key property of the standardization process is that the magnitude of the resulting standardized vector is algebraically bounded, regardless of the specific aggregation operator used, provided the operator produces a mean within the range of the input values. The following theorem establishes a universal bound for the squared L2-norm of this vector. This general result is crucial for obtaining robust continuity bounds for the full operator pipeline.
:::{prf:theorem} General Bound on the Norm of the Standardized Vector
:label: thm-z-score-norm-bound
Let $\mathbf{v} = (v_iraw value ({prf:ref}`def-raw-value-operator`)A}}$ be a $k$-dimensional vector of raw values from an alive set ({prf:ref}`def-alive-dead-sets`) $\mathcal{A}$ of size $k=|\mathcal{A}| \ge 1$. The raw value ({prf:ref}`def-raw-value-operator`)nded such that $|v_i| \le V_{\max}$. Let the statistical properties $(\mu_{\mathcal{A}}, \sigma'_{\mathcal{A}})$ be calculated using any valid **Swarm ({prf:ref}`def-swarm-and-state-space`) Aggregation Operator** $M$ that guarantees the mean is bounded by the values, i.e., $|\mu_{\mathcal{A}}| \le V_{\max}$.
Let $\mathbf{z}$ be the corresponding $k$-dimensional standardized vector, where each component is $z_i = (v_i - \mu_{\mathcal{A}}) / \sigma'_{\mathcal{A}}$ and the regularized standard deviation is $\sigma'_{\mathcal{A}} = \sigma\'_{\text{reg}}(\operatorname{Var}[\mu_{\mathbf{v}}])$ from [](#def-statistical-properties-measurement). Denote the minimal value of this map by $\sigma'_{\min\,\text{bound}} := \sqrt{\kappa_{\text{var,min}} + \varepsilon_{\mathrm{std}}^2}$.
The squared Euclidean norm of the standardized vector $\mathbf{z}$ is strictly bounded by a constant that depends on the number of alive walker ({prf:ref}`def-walker`)s and the global parameters:

$$
\|\mathbf{z}\|_2^2 \le k \left( \frac{2V_{\max}}{\varepsilon_{\mathrm{std}}} \right)^2

$$

This universal bound on standardized vector norms is applied in {prf:ref}`02_euclidean_gas` for bounding the magnitude of standardized reward and distance scores in error analysis.
:::
:::{prf:proof}
:label: proof-thm-z-score-norm-bound
**Proof.**
The proof proceeds by first establishing a uniform bound on the magnitude of any single component of the standardized vector and then summing the squares of these bounds.
1.  **Bound a Single Standardized Component:**
    The squared Euclidean norm of the standardized vector $\mathbf{z}$ is the sum of its squared components, $\|\mathbf{z}\|_2^2 = \sum_{i \in \mathcal{A}} z_i^2$. We first bound the absolute value of a single component, $|z_i|$.

$$
|z_i| = \left| \frac{v_i - \mu_{\mathcal{A}}}{\sigma'_{\mathcal{A}}} \right| = \frac{|v_i - \mu_{\mathcal{A}}|}{|\sigma'_{\mathcal{A}}|}

$$

2.  **Bound the Numerator and Denominator:**
    *   **Numerator:** Using the triangle inequality, the numerator is bounded by the sum of the absolute values of its terms: $|v_i - \mu_{\mathcal{A}}| \le |v_i| + |\mu_{\mathcal{A}}|$. By the problem's preconditions, the raw values are bounded by $|v_i| \le V_{\max}$. For any aggregation operator that is a convex combination of its inputs (such as the empirical mean), the resulting mean $\mu_{\mathcal{A}}$ will also be bounded by $V_{\max}$. We assume this standard property holds, giving $|\mu_{\mathcal{A}}| \le V_{\max}$. Therefore, the numerator is bounded by:

$$
|v_i - \mu_{\mathcal{A}}| \le V_{\max} + V_{\max} = 2V_{\max}

$$

*   **Denominator:** The regularized standard deviation obeys the floor $\sigma'_{\mathcal{A}} \ge \sigma'_{\min\,\text{bound}}$ because the cubic patch is constant on $[0,\kappa_{\text{var,min}}]$ and nondecreasing thereafter. In particular, the denominator is strictly bounded below by this positive constant:

$$
|\sigma'_{\mathcal{A}}| \ge \sigma'_{\min\,\text{bound}}

$$

3.  **Combine for Component-wise Bound:**
    Combining the bounds for the numerator and denominator gives a uniform bound for the magnitude of any single standardized score:

$$
|z_i| \le \frac{2V_{\max}}{\sigma'_{\min\,\text{bound}}}

$$

4.  **Sum Over All Components:**
    The squared L2-norm is the sum of the squares of these components over the $k$ walker ({prf:ref}`def-walker`)s in the alive set ({prf:ref}`def-alive-dead-sets`) $\mathcal{A}$.

$$
\|\mathbf{z}\|_2^2 = \sum_{i \in \mathcal{A}} z_i^2 \le \sum_{i \in \mathcal{A}} \left( \frac{2V_{\max}}{\sigma'_{\min\,\text{bound}}} \right)^2

$$

Since the bound is the same for all $k$ components, we have:

$$
\|\mathbf{z}\|_2^2 \le k \left( \frac{2V_{\max}}{\sigma'_{\min\,\text{bound}}} \right)^2

$$

This provides a general bound on the norm of the standardized vector that is valid for any compliant aggregation operator.
**Q.E.D.**
:::
#### 11.1.5 Asymptotic Behavior of Moment Continuity
The axiomatic structural growth exponents ($p_{\mu,S}, p_{m_2,S}$) of an aggregation operator determine the asymptotic behavior of the continuity for the derived statistical moments. The following theorem establishes how these base exponents propagate to the regularized standard deviation function, a critical step in analyzing the system's stability for large swarms.
:::{prf:theorem} Asymptotic Behavior of the Structural Continuity for the Regularized Standard Deviation
:label: thm-asymptotic-std-dev-structural-continuity
Let the chosen swarm ({prf:ref}`def-swarm-and-state-space`) aggregation operatorLipschitz ({prf:ref}`axiom-reward-regularity`)al growth exponents $p_{\mu,S}$ and $p_{m_2,S}$ for its mean and second moment, respectively, as defined in [](#def-swarm-aggregation-operator-axiomatic) ({prf:ref}`def-alive-dead-sets`) ({prf:ref}`def-swarm-and-state-space`) ({prf:ref}`axiom-reward-regularity`). Let $L_{\sigma',S}(\mathcal{S})$ be the structural Lipschitz function for the regularized standard deviation, as derived in [](#lem-stats-structural-continuity).
The asymptotic behavior of this function for large swarm ({prf:ref}`def-swarm-and-state-space`) size $k = |\mathcal{A}(\mathcal{S})|$ is determined by the larger of the two structural growth exponents. Let the worst-case exponent be:

$$
p_{\text{worst-case}} := \max(p_{\mu,S}, p_{m_2,S})

$$

Then, for large $k$, the structural Lipschitz function for the standard deviation is governed by this worst-case exponent:

$$
L_{\sigma',S}(k) \propto k^{p_{\text{worst-case}}}

$$

:::
:::{prf:proof}
:label: proof-thm-asymptotic-std-dev-structural-continuity
**Proof.**
The proof proceeds by analyzing the asymptotic form of the bound for the structural Lipschitz constant of the regularized standard deviation, $L_{\sigma',S}$, which was established in [](#lem-stats-structural-continuity).
1.  **Recall the Bound for $L_{\sigma',S}$:**
    From [](#lem-stats-structural-continuity), the structural Lipschitz constant is bounded by:

$$
L_{\sigma',S}(\mathcal{S}) \le \frac{L_{m_2,S}(\mathcal{S}) + 2V_{\max}L_{\mu,S}(\mathcal{S})}{2\varepsilon_{\mathrm{std}}}

$$

2.  **Analyze the Asymptotic Behavior of the Numerator:**
    We analyze the behavior of the numerator for a large number of alive walker ({prf:ref}`def-walker`)s, $k = |\mathcal{A}(\mathcal{S})|$. By the axiomatic definition of the structural growth exponents ([](#def-swarm-aggregation-operator-axiomatic)), the structural Lipschitz functions have the following asymptotic forms:
    *   $L_{\mu,S}(k) \propto k^{p_{\mu,S}}$
    *   $L_{m_2,S}(k) \propto k^{p_{m_2,S}}$
    The numerator is therefore a sum of two terms with power-law growth:

$$
L_{m_2,S}(k) + 2V_{\max}L_{\mu,S}(k) \propto k^{p_{m_2,S}} + C \cdot k^{p_{\mu,S}}

$$

where $C = 2V_{\max}$ is a constant.
3.  **Identify the Dominant Term:**
    In the limit of large $k$, the behavior of a sum of power-law terms is dominated by the term with the largest exponent. Therefore, the asymptotic behavior of the numerator is proportional to $k$ raised to the power of the maximum of the two exponents.

$$
\text{Numerator}(k) \propto k^{\max(p_{\mu,S}, p_{m_2,S})} = k^{p_{\text{worst-case}}}

$$

4.  **Conclusion:**
    The denominator, $2\varepsilon_{\mathrm{std}}$, is a constant that does not depend on the swarm ({prf:ref}`def-swarm-and-state-space`) size $k$. The asymptotic behavior of the entire expression for $L_{\sigma',S}(k)$ is therefore determined solely by the behavior of its numerator. This gives the final result:

$$
L_{\sigma',S}(k) \propto k^{p_{\text{worst-case}}}

$$

**Q.E.D.**
:::
### 11.2. Mean-Square Continuity of the Standardization Operator ({prf:ref}`def-standardization-operator-n-dimensional`)
The analysis of the **N-Dimensional Standardization Operator ({prf:ref}`def-standardization-operator-n-dimensional`)**, $z(S, V, M)$, is central to the framework's stability. While the patched definition of the operator using the Regularized Standard Deviation Function ([](#def-statistical-properties-measurement)) also satisfies a stronger deterministic Lipschitz continuity property (proven in Section 11.3), the analysis of its **mean-square continuity** is preserved here. This is for two primary reasons: first, the mean-square framework provides a more detailed, component-wise analysis of error propagation from different sources (value vs. structure); second, the resulting bounds on the *average* error are often tighter and more representative of the system's typical behavior in non-degenerate regimes than the bounds derived from a worst-case deterministic analysis.
The following sections provide a rigorous, self-contained proof of the operator's mean-square continuity. The proof is valid for any aggregation operator satisfying the axiomatic requirements of [](#def-swarm-aggregation-operator-axiomatic) and any raw value operator ({prf:ref}`def-raw-value-operator`) that is proven to be mean-square continuous (e.g., the distance operator from [](#thm-distance-operator-mean-square-continuity)).
The strategy is to decompose the total expected squared error into its two fundamental sources:
1.  **Value-Induced Error:** The error resulting from the change in the raw value vector ($v_1 \to v_2$) while holding the swarm's structure constant.
2.  **Structure-Induced Error:** The error resulting from the change in the swarm's structure ($S_1 \to S_2$) for a given raw value vector.
By bounding the expectation of these two components separately, we establish a unified and robust mean-square continuity bound for the entire operator.
#### 11.2.1. Theorem: Decomposition of Mean-Square Standardization Error

:::{prf:theorem} Decomposition of Mean-Square Standardization Error
:label: thm-standardization-operator-unified-mean-square-continuity

Let $S_1$ and $S_2$ be two swarm ({prf:ref}`def-swarm-and-state-space`) states. Let the standardizraw value ({prf:ref}`def-raw-value-operator`)rf:ref}`def-standardization-operator-n-dimensional`) $z$ use a raw value operator $V$ and a swarm aggregation operator $M$. Let $z_1 = z(S_1, V, M)$ and $z_2 = z(S_2, V, M)$ be the corresponding standardized vectors resulting from the full stochastic process.
The expected squared Euclidean distance between the output vectors $z_1$ and $z_2$ is bounded by the sum of two fundamental error components:

$$
\mathbb{E}[\| \mathbf{z}_1 - \mathbf{z}_2 \|_2^2] \le 2 \cdot E_{V,ms}^2(\mathcal{S}_1, \mathcal{S}_2) + 2 \cdot E_{S,ms}^2(\mathcal{S}_1, \mathcal{S}_2)

$$

where the error components are formally defined in the following sections.
:::

:::{prf:proof}
:label: proof-thm-standardization-operator-unified-mean-square-continuity
**Proof.**
The proof follows from decomposing the total error using an intermediate vector and then taking the expectation. The intermediate vector is $z_{\text{inter}} := z(\mathcal{S}_1, \mathbf{v}_2, M)$, which uses the second swarm ({prf:ref}`def-swarm-and-state-space`)'s raw values with the first swarm's structure.
The total squared error is bounded using the inequality $\|a+b\|_2^2 \leq 2(\|a\|_2^2 + \|b\|_2^2)$:

$$
\| \mathbf{z}_1 - \mathbf{z}_2 \|_2^2 = \| (\mathbf{z}_1 - \mathbf{z}_{\text{inter}}) + (\mathbf{z}_{\text{inter}} - \mathbf{z}_2) \|_2^2 \le 2 \| \mathbf{z}_1 - \mathbf{z}_{\text{inter}} \|_2^2 + 2 \| \mathbf{z}_{\text{inter}} - \mathbf{z}_2 \|_2^2

$$

The first term, $\|z_1 - z_{\text{inter}}\|_2^2$, is the squared **value error**, as it arises from the change $v_1 \to v_2$ for a fixed structure $S_1$. The second term, $\|z_{\text{inter}} - z_2\|_2^2$, is the squared **structural error**, as it arises from the change $S_1 \to S_2$ for a fixed value vector $v_2$.
Taking the expectation of both sides of the inequality over all sources of randomness and applying linearity gives the stated result.
**Q.E.D.**
:::
##### 11.2.1.1. The Expected Squared Value Error ($E^2_{V,ms}$)

:::{prf:definition} The Expected Squared Value Error
:label: def-expected-squared-value-error

The **Expected Squared Value Error**, $E^2_{V,ms}(\mathcal{S}_1, \mathcal{S}_2)$, bounds the component of error that arises from the change in the underlying probability distribution of the raw value ({prf:ref}`def-raw-value-operator`) vector (from $V(\mathcal{S}_1)$ to $V(\mathcal{S}_2)$), while holding the swarm ({prf:ref}`def-swarm-and-state-space`)'s structural context for the standardization fixed at $\mathcal{S}_1$.
It is defined as:

$$
E_{V,ms}^2(\mathcal{S}_1, \mathcal{S}_2) := \mathbb{E}[\| \mathbf{z}(\mathcal{S}_1, \mathbf{v}_1, M) - \mathbf{z}(\mathcal{S}_1, \mathbf{v}_2, M) \|_2^2]

$$

where the expectation is taken over the joint distribution of the raw value vectors $\mathbf{v}_1 \sim V(\mathcal{S}_1)$ and $\mathbf{v}_2 \sim V(\mathcal{S}_2)$. This term measures the propagation of error from the input measurement's distribution to the output of the standardization operator ({prf:ref}`def-standardization-operator-n-dimensional`), under a fixed structural context. Its explicit bound is derived in [](#thm-standardization-value-error-mean-square).
:::

##### 11.2.1.2. The Expected Squared Structural Error ($E^2_{S,ms}$)

:::{prf:definition} The Expected Squared Structural Error
:label: def-expected-squared-structural-error

The **Expected Squared Structural Error**, $E^2_{S,ms}(\mathcal{S}_1, \mathcal{S}_2)$, bounds the expected error in the standardization operator ({prf:ref}`def-standardization-operator-n-dimensional`)'s output arising from the change in the swarm ({prf:ref}`def-swarm-and-state-space`) structure from $\mathcal{S}_1$ to $\mathcal{S}_2$, evaluated using the second swarm's raw value ({prf:ref}`def-raw-value-operator`) vector $\mathbf{v}_2$. It is defined as:

$$
E_{S,ms}^2(\mathcal{S}_1, \mathcal{S}_2) := \mathbb{E}[\| \mathbf{z}(\mathcal{S}_1, \mathbf{v}_2, M) - \mathbf{z}(\mathcal{S}_2, \mathbf{v}_2, M) \|_2^2]

$$

where the expectation is taken over the distribution of the raw value vector $\mathbf{v}_2 \sim V(\mathcal{S}_2)$. Its explicit bound is derived in [](#thm-standardization-structural-error-mean-square).
:::

#### 11.2.2. Bounding the Expected Squared Value Error ($E^2_{V,ms}$)
This section provides a rigorous, self-contained proof for the bound on the **Expected Squared Value Error**, $E^2_{V,ms}$. This term quantifies the component of the standardization operator ({prf:ref}`def-standardization-operator-n-dimensional`)'s error that arises exclusively from the change in the underlying distribution of the raw value vector, while the swarm's structure is held constant.
The central result is the following theorem, which establishes that the expected output error of the standardization pipeline is bounded by the expected input error from the raw value operator ({prf:ref}`def-raw-value-operator`).
:::{prf:theorem} Bounding the Expected Squared Value Error
:label: thm-standardization-value-error-mean-square
Let $S_1$ be a fixed swarm ({prf:ref}`def-swarm-and-state-space`) state. Let $V$ be a raw value operator that is mean-square continuous, such that $\mathbb{E}[\|\mathbf{v}_1 - \mathbf{v}_2\|_2^2] \le F_{V,ms}(\mathcal{S}_1, \mathcal{S}_2)$ for some deterministic bounding function $F_{V,ms}$.
The expected squared value error is bounded as follows:

$$
E_{V,ms}^2(\mathcal{S}_1, \mathcal{S}_2) \le C_{V,\text{total}}(\mathcal{S}_1) \cdot F_{V,ms}(\mathcal{S}_1, \mathcal{S}_2)

$$

where $C_{V,\text{total}}(\mathcal{S}_1)$ is the **Total Value Error Coefficient**, a deterministic constant derived from the axiomatic properties of the aggregation operator and the global parameters, as formally defined in [](#def-lipschitz-value-error-coefficients).

Proof provided in {prf:ref}`proof-thm-standardization-value-error-mean-square`.
:::
The proof of this theorem requires a careful algebraic decomposition of the total error vector into three distinct and manageable components. The subsequent subsections will state and prove a deterministic bound for each of these three components. These results are then assembled in the final proof of the main theorem.
##### 11.2.2.1. Sub-Lemma: Algebraic Decomposition of the Value Error

:::{prf:lemma} Algebraic Decomposition of the Value Error
:label: lem-sub-value-error-decomposition

Let $\mathcal{S}$ be a fixed swarm ({prf:ref}`def-swarm ({prf:ref}`def-swarm-and-state-space`)-and-stateraw value ({prf:ref}`def-raw-value-operator`)h alive set ({prf:ref}`def-alive-dead-sets`) $\mathcal{A}$ of size $k$. Let $\mathbf{v}_1$raw value ({prf:ref}`def-raw-value-operator`)$ be two raw value vectors for the alive set. Let $(\mu_1, \sigma'_1)$ and $(\mu_2, \sigma'_2)$ be the corresponding statistical properties, and let $\mathbf{z}_1$ and $\mathbf{z}_2$ be the corresponding standardized vectors.
The total value error vector, $\Delta\mathbf{z} = \mathbf{z}_1 - \mathbf{z}_2$, can be expressed as the sum of three components:

$$
\Delta\mathbf{z} = \Delta_{\text{direct}} + \Delta_{\text{mean}} + \Delta_{\text{fluc}}

$$

where:
1.  **The Direct Shift ($\Delta_{\text{direct}}$):** The error from the change in the raw value vector itself, scaled by the initial standard deviation.

$$
\Delta_{\text{direct}} := \frac{\mathbf{v}_1 - \mathbf{v}_2}{\sigma'_1}

$$

2.  **The Mean Shift ($\Delta_mean$):** The error from the change in the aggregator's computed mean, applied uniformly to all walker ({prf:ref}`def-walker`)s.

$$
\Delta_{\text{mean}} := \frac{\mu_2 - \mu_1}{\sigma'_1} \cdot \mathbf{1}

$$

where $**1**$ is a k-dimensional vector of ones.
3.  **The Statistical Fluctuation ($\Delta_fluc$):** The error from the change in the aggregator's computed standard deviation, which rescales the second standardized vector.

$$
\Delta_{\text{fluc}} := \mathbf{z}_2 \cdot \frac{\sigma'_2 - \sigma'_1}{\sigma'_1}

$$

Furthermore, the total squared error is bounded by three times the sum of the squared norms of these components:

$$
\|\Delta\mathbf{z}\|_2^2 \le 3\left( \|\Delta_{\text{direct}}\|_2^2 + \|\Delta_{\text{mean}}\|_2^2 + \|\Delta_{\text{fluc}}\|_2^2 \right)

$$

:::

:::{prf:proof}
:label: proof-line-3101
**Proof.**
The proof of the decomposition is a direct algebraic manipulation.
1.  **Start with the Definition of the Error.**
    The total error is $\Delta\mathbf{z} = \mathbf{z}_1 - \mathbf{z}_2 = \frac{\mathbf{v}_1 - \mu_1}{\sigma'_1} - \frac{\mathbf{v}_2 - \mu_2}{\sigma'_2}$.
2.  **Decomposition.**
    We add and subtract terms to isolate the desired components.

$$
\Delta\mathbf{z} = \frac{\mathbf{v}_1 - \mu_1}{\sigma'_1} - \frac{\mathbf{v}_2 - \mu_2}{\sigma'_1} + \frac{\mathbf{v}_2 - \mu_2}{\sigma'_1} - \frac{\mathbf{v}_2 - \mu_2}{\sigma'_2}

$$

$$
= \left( \frac{\mathbf{v}_1 - \mathbf{v}_2}{\sigma'_1} \right) + \left( \frac{\mu_2 - \mu_1}{\sigma'_1} \cdot \mathbf{1} \right) + \left( \frac{\mathbf{v}_2 - \mu_2}{\sigma'_1} - \frac{\mathbf{v}_2 - \mu_2}{\sigma'_2} \right)

$$

The final term can be rewritten by factoring out $(v_2 - \mu_2)$:

$$
= \Delta_{\text{direct}} + \Delta_{\text{mean}} + (\mathbf{v}_2 - \mu_2) \left(\frac{1}{\sigma'_1} - \frac{1}{\sigma'_2}\right) = \Delta_{\text{direct}} + \Delta_{\text{mean}} + \frac{\mathbf{v}_2 - \mu_2}{\sigma'_2} \frac{\sigma'_2 - \sigma'_1}{\sigma'_1}

$$

Recognizing that $(v_2 - \mu_2) / \sigma'_2$ is $z_2$, this matches the definition of $\Delta_fluc$.
3.  **Bound on the Squared Norm.**
    The bound on the total squared norm follows directly from the triangle inequality and the elementary inequality $(a+b+c)^2 \leq 3(a^2+b^2+c^2)$.
**Q.E.D.**
:::
##### 11.2.2.2. Sub-Lemma: Bounding the Direct Shift Error Component

:::{prf:lemma} Bounding the Direct Shift Error Component
:labelraw value ({prf:ref}`def-raw-value-operator`)alue-shift-bound

Let $\mathcal{S}$ be a fixed swarm ({prf:ref}`def-swarm-and-state-space`raw value ({prf:ref}`def-raw-value-operator`)hbf{v}_1$ and $\mathbf{v}_2$ be two raw value vectors for the alive set ({prf:ref}`def-alive-dead-sets`). The squared Euclidean norm of the direct shift error component, $\Delta_{\text{direct}} = (\mathbf{v}_1 - \mathbf{v}_2) / \sigma'_1$, is bounded as follows:

$$
\|\Delta_{\text{direct}}\|_2^2 \le \frac{1}{\big(\sigma'_{\min,\text{bound}}\big)^2} \cdot \|\mathbf{v}_1 - \mathbf{v}_2\|_2^2

$$

where $\sigma'_{\min,\text{bound}} := \sqrt{\kappa_{\text{var,min}}+\varepsilon_{\text{std}}^2}$ is the uniform lower bound from the regularized standard deviation.
:::

:::{prf:proof}
:label: proof-lem-sub-direct-value-shift-bound
**Proof.**
The proof is a direct application of the definition of $\Delta_{\text{direct}}$ and the lower bound on $\sigma'_1$. The squared norm is $(1/(\sigma'_1)^2)\|\mathbf v_1 - \mathbf v_2\|_2^2$. From [](#def-statistical-properties-measurement), the regularized standard deviation obeys $\sigma'_1\ge \sigma'_{\min,\text{bound}}$, hence $1/(\sigma'_1)^2 \le 1/(\sigma'_{\min,\text{bound}})^2$.
**Q.E.D.**
:::
##### 11.2.2.3. Sub-Lemma: Bounding the Mean Shift Error Component

:::{prf:lemma} Boundiraw value ({prf:ref}`def-raw-value-operator`)Error Component
:label: lem-sub-mean-shift-bound

Let $\mathcalraw value ({prf:ref}`def-raw-value-operator`)arm ({prf:ref}`def-swarm-and-state-space`) state with alive ({prf:ref}`def-alive-dead-sets`) set ({prf:ref}`def-aliveLipschiraw value ({prf:ref}`def-raw-value-operator`)m-reward-regularity`)hcal{A}$ of size **k**. Let $\mathbf{v}_1$ and $\mathbf{v}_2$ be two raw value vectors. The squared Euclidean norm of the mean shift error component, $\Delta_{\text{mean}} = ((\mu_2 - \mu_1) / \sigma'_1) \cLipschitz ({prf:ref}`axiom-reward-regularity`)s bounded as follows:

$$
\|\Delta_{\text{mean}}\|_2^2 \le \frac{k \cdot (L_{\mu,M}(\mathcal{S}))^2}{\big(\sigma'_{\min,\text{bound}}\big)^2} \cdot \|\mathbf{v}_1 - \mathbf{v}_2\|_2^2

$$

where $L_{\mu,M}(S)$ is the axiomatic **Value Lipschitz Function** for the aggregator's mean.
:::

:::{prf:proof}
:label: proof-lem-sub-mean-shift-bound
**Proof.**
The squared norm is $k \cdot (\mu_2 - \mu_1)^2 / (\sigma'_1)^2$. From the aggregator axiom ([](#def-swarm ({prf:ref}`def-swarm-and-state-space`)-aggregation-operator-axiomatic)), $(\mu_2 - \mu_1)^2 \leq (L_{\mu,M}(S))^2 \|v_1 - v_2\|_2^2$. Combining this with the lower bound on $\sigma'_1$ gives the final result.
**Q.E.D.**
:::
##### 11.2.2.4. Sub-Lemma: Bounding the Statistical Fluctuation Error Component

:::{prf:lemma} Bounding the Statistical Fluctuation Error Component
:label: lem-sub-statistical-fluctuation-bound

Let $\mathcal{S}$ ({prf:ref}`def-swarm-and-state-space`) be a fixedraw value ({prf:ref}`def-raw-value-operator`)alive set ({prf:ref}`def-alive-dead-sets`) $\mathcal{A}$ of size **k**. Let $\mathbf{v}_1$ and $\mathbf{v}_2$ be two raw value ({prf:ref}`def-raw-value-operator`) vectors with components bounded by $V_{\max}$. The squared Euclidean norm of the statistical fluctuation error component, $\Delta_{\text{fluc}} = \mathbf{z}_2 \cdot ((\sigma'_2 - \sigma'_1) / \sigma'_1)$, is bounded as follows:

$$
\|\Delta_{\text{fluc}}\|_2^2 \le k \left( \frac{2V_{\max}}{\sigma'_{\min,\text{bound}}} \right)^2 \left( \frac{L_{\sigma',M}(\mathcal{S})}{\sigma'_{\min,\text{bound}}} \right)^2 \cdot \|\mathbf{v}_1 - \mathbf{v}_2\|_2^2

$$

where $L_{\sigma',M}(S)$ is the derived Lipschitz constant for the regularized standard deviation from [](#lem-stats-value-continuity).
:::

:::{prf:proof}
:label: proof-lem-sub-statistical-fluctuation-bound
**Proof.**
The squared norm is $\|z_2\|_2^2 \cdot (\sigma'_2 - \sigma'_1)^2 / (\sigma'_1)^2$. We bound each term:
- From [](#thm-z-score-norm-bound), $\|z_2\|_2^2 \leq k\,(2V_{\max}/\sigma'_{\min,\text{bound}})^2$.
- From [](#lem-stats-value-continuity), $(\sigma'_2 - \sigma'_1)^2 \leq (L_{\sigma',M}(S))^2 \|v_1 - v_2\|_2^2$.
- The term $1/(\sigma'_1)^2$ is bounded by $1/(\sigma'_{\min,\text{bound}})^2$.
Combining these three bounds yields the final result.
**Q.E.D.**
:::
##### 11.2.2.5. Definition: Value Error Coefficients
:::{prf:definition} Value Error Coefficients
:label: def-value-error-coefficients

Let $\mathcal{S}$ be a fixed swarm ({prf:ref}`def-swarm-and-state-space`) state with alive set ({prf:ref}`def-alive-dead-sets`) $\mathcal{A}$ of size **k**, and let **M** be the chosen **Swarm Aggregation Operator**. The coefficients for the value error bounds are defined as follows:

1.  **The Direct Shift Coefficient ($C_V,direct$):**

$$
C_{V,\text{direct}} := \frac{1}{\sigma'^2_{\min,\text{bound}}}

$$

2.  **The Mean Shift Coefficient ($C_V,\mu(S)$):**

$$
C_{V,\mu}(\mathcal{S}) := \frac{k \cdot (L_{\mu,M}(\mathcal{S}))^2}{\sigma'^2_{\min,\text{bound}}}

$$

3.  **The Statistical Fluctuation Coefficient ($C_V,\sigma(S)$):**

$$
C_{V,\sigma}(\mathcal{S}) := k \left( \frac{2V_{\max}}{\sigma'_{\min,\text{bound}}} \right)^2 \left( \frac{L_{\sigma',M}(\mathcal{S})}{\sigma'_{\min,\text{bound}}} \right)^2

$$

4.  **The Total Value Error Coefficient ($C_V,total(S)$):** The composite coefficient that bounds the total squared error.

$$
C_{V,\text{total}}(\mathcal{S}) := 3 \cdot \left( C_{V,\text{direct}} + C_{V,\mu}(\mathcal{S}) + C_{V,\sigma}(\mathcal{S}) \right)

$$

where $L_{\mu,M}(S)$ and $L_{\sigma',M}(S)$ are the value Lipschitz functions for the aggregator's mean and regularized standard deviation, respectively.
:::
##### 11.2.2.6. Proof of Theorem 11.2.2
:label: proof-thm-standardization-value-error-mean-square
:::{prf:proof} of {prf:ref}`thm-standardization-value-error-mean-square`
:label: proof-proof-thm-standardization-value-error-mean-square
Let $S_1$ be a fixed swarm ({prf:ref}`def-swarm-and-state-space`) state. Let $V$ be a raw value ({prf:ref}`def-raw-value-operator`) operator that is mean-square continuous, such that $\mathbb{E}[\|\mathbf{v}_1 - \mathbf{v}_2\|_2^2] \le F_{V,ms}(\mathcal{S}_1, \mathcal{S}_2)$ for some deterministic bounding function $F_{V,ms}$.
The expected squared value error is bounded as follows:

$$
E_{V,ms}^2(\mathcal{S}_1, \mathcal{S}_2) \le C_{V,\text{total}}(\mathcal{S}_1) \cdot F_{V,ms}(\mathcal{S}_1, \mathcal{S}_2)

$$

where $C_{V,\text{total}}(\mathcal{S}_1)$ is the **Total Value Error Coefficient** from [](#def-lipschitz-value-error-coefficients).
:::
:::{prf:proof}
:label: proof-proof-thm-standardization-value-error-mean-square-2
**Proof.**
1.  **Start with the Decomposed Error Bound.**
    From [](#sub-lem-value-error-decomposition), we have a deterministic bound on the squared error for any specific realization of $v_1$ and $v_2$:

$$
\|\mathbf{z}_1 - \mathbf{z}_2\|_2^2 \le 3\left( \|\Delta_{\text{direct}}\|_2^2 + \|\Delta_{\text{mean}}\|_2^2 + \|\Delta_{\text{fluc}}\|_2^2 \right)

$$

2.  **Substitute Deterministic Component Bounds.**
    We substitute the deterministic bounds for each component from the preceding sub-lemmas, which all relate the component error to $\|v_1 - v_2\|_2^2$. Factoring out this term and using the definitions from [](#def-lipschitz-value-error-coefficients) gives:

$$
\|\mathbf{z}_1 - \mathbf{z}_2\|_2^2 \le C_{V,\text{total}}(\mathcal{S}_1) \cdot \|\mathbf{v}_1 - \mathbf{v}_2\|_2^2

$$

3.  **Take the Expectation.**
    The expected squared value error is the expectation of the left-hand side. We take the expectation of both sides. Since $C_{V,total}(S_1)$ is a deterministic constant for a fixed state $S_1$:

$$
\mathbb{E}[\|\mathbf{z}_1 - \mathbf{z}_2\|_2^2] \le C_{V,\text{total}}(\mathcal{S}_1) \cdot \mathbb{E}[\|\mathbf{v}_1 - \mathbf{v}_2\|_2^2]

$$

4.  **Apply the Mean-Square Continuity Axiom for Raw Values.**
    By axiom ([](#axiom-raw-value-mean-square-continuity)), $E[\|v_1 - v_2\|_2^2]$ is bounded by $F_{V,ms}$. Substituting this gives the final result.
**Q.E.D.**
:::
#### 11.2.3. Bounding the Expected Squared Structural Error ($E^2_{S,ms}$)
This section provides a rigorous, self-contained proof for the bound on the **Expected Squared Structural Error**, $E^2_{S,ms}$. This term quantifies the component of the standardization operator ({prf:ref}`def-standardization-operator-n-dimensional`)'s error that arises exclusively from the change in the swarm's structure (i.e., the set of alive walkers), for a fixed underlying raw value vector.
:::{prf:theorem} Bounding the Expected Squared Structural Error
:label: thm-standardization-structural-error-mean-square
This theorem bounds the structural error component of {prf:ref}`def-standardization-operator-n-dimensional`, quantifying how status changes affect standardization.

The expected squared structural error is bounded deterministically by a function of the number of status changes, $n_c$.

$$
E_{S,ms}^2(\mathcal{S}_1, \mathcal{S}_2) \le C_{S,\text{direct}} \cdot n_c(\mathcal{S}_1, \mathcal{S}_2) + C_{S,\text{indirect}}(\mathcal{S}_1, \mathcal{S}_2) \cdot n_c(\mathcal{S}_1, \mathcal{S}_2)^2

$$

where $C_{S,\text{direct}}$ and $C_{S,\text{indirect}}$ are the **Structural Error Coefficients**, deterministic constants derived from the axiomatic properties of the aggregation operator and the global parameters, as formally defined in [](#def-structural-error-coefficients).
:::
The proof of this theorem requires an algebraic decomposition of the total structural error into two distinct components: a "direct" error from walker ({prf:ref}`def-walker`)s appearing or disappearing from the alive set ({prf:ref}`def-alive-dead-sets`), and an "indirect" error from the resulting change in the statistical moments that affects all other walkers.
##### 11.2.3.1. Sub-Lemma: Algebraic Decomposition of the Structural Error

:::{prf:lemma} Algebraic Decomposition of the Structural Error
:label: lem-sub-structural-error-decomposition

Let $\mathbf{v}$ be a fixed raw value ({prf:ref}`def-raw-value-operator`) vector. Let $\mathcal{S}_1$ and $\mathcal{S}_2$ be two swarm ({prf:ref}`def-swarm-and-state-space`) states with alive set ({prf:ref}`def-alive-dead-sets`)s $\mathcal{A}_1$ and $\mathcal{A}_2$. Let $\mathbf{z}_1 = \mathbf{z}(\mathcal{S}_1, \mathbf{v})$ and $\mathbf{z}_2 = \mathbf{z}(\mathcal{S}_2, \mathbf{v})$ be the corresponding N-dimensional standardized vectors.
The total structural error vector, $\Delta\mathbf{z} = \mathbf{z}_1 - \mathbf{z}_2$, can be expressed as the sum of two orthogonal components, and its squared norm is the sum of the squared norms of the components:

$$
\|\Delta\mathbf{z}\|_2^2 = \|\Delta_{\text{direct}}\|_2^2 + \|\Delta_{\text{indirect}}\|_2^2

$$

where:
1.  **The Direct Error ($\Delta_{\text{direct}}$):** The error vector whose non-zero components correspond to walker ({prf:ref}`def-walker`)s whose status changes.
2.  **The Indirect Error ($\Delta_{\text{indirect}}$):** The error vector whose non-zero components correspond to walker ({prf:ref}`def-walker`)s whose status remains the same.
:::

:::{prf:proof}
:label: proof-lem-sub-structural-error-decomposition
**Proof.**
The proof follows from partitioning the sum of squared errors over the N walker ({prf:ref}`def-walker`) indices into a sum over walkers whose status changes and a sum over walkers whose status is stable. These two sets of indices are disjoint. The two corresponding error vectors therefore have disjoint support, are orthogonal, and the squared norm of their sum is the sum of their squared norms.
**Q.E.D.**
:::
##### 11.2.3.2. Sub-Lemma: Bounding the Direct Structural Error Component

:::{prf:lemma} Bounding the Direct Structural Error Component
raw value ({prf:ref}`def-raw-value-operator`)rect-structural-error

This lemma bounds the direct component of {prf:ref}`def-expected-squared-structural-error`.

Let $\mathbf{v}$ be a fixed raw value vector with components bounded by $V_{\max}$. The squared Euclidean norm of the direct structural error component, $\|\Delta_{\text{direct}}\|^2$, is bounded by the number of status changes $n_c$.

$$
\|\Delta_{\text{direct}}\|_2^2 \le \left( \frac{4V_{\max}^2}{\sigma'^2_{\min,\text{bound}}} \right) n_c

$$

:::

:::{prf:proof}
:label: proof-lem-sub-direct-structural-error
**Proof.**
The direct error vector has $n_c$ non-zero components. For each such component **i**, one of $z_{1,i}$ or $z_{2,i}$ is zero, and the other is a valid Z-score. From [](#thm-z-score-norm-bound), any single Z-score is bounded by $|z_j| \leq 2V_{\max}/\sigma'_{\min,\text{bound}}$. The squared error for component **i** is thus bounded by $(2V_{\max}/\sigma'_{\min,\text{bound}})^2$. Summing this bound over the $n_c$ unstable walker ({prf:ref}`def-walker`)s gives the final result.
**Q.E.D.**
:::
##### 11.2.3.3. Sub-Lemma: Bounding the Indirect Structural Error Component

:::{prf:lemma} Bounding the Indirect Structural Error Component
:label: lem-sub-indirect-structural-error

Let $\mathbf{v}$ be a fixed raw value ({prf:ref}`def-raw-value-operator`) vector. Let $\mathcal{S}_1$ and $\mathcal{S}_2$ be two swarm ({prf:ref}`def-swarm-and-state-space`) states. The squared Euclidean norm of the indirect structural error component, $\|\Delta_{\text{indirect}}\|^2$, is bounded as follows:

$$
\|\Delta_{\text{indirect}}\|_2^2 \le C_{S,\text{indirect}}(\mathcal{S}_1, \mathcal{S}_2) \cdot n_c(\mathcal{S}_1, \mathcal{S}_2)^2

$$

where $C_{S,indirect}$ is the **Total Indirect Structural Error Coefficient**.
:::

:::{prf:proof}
:label: proof-lem-sub-indirect-structural-error
**Proof.**
The proof combines the algebraic error decomposition with the deterministic bounds for each component. From [](#sub-lem-structural-error-decomposition), $\|\Deltaz\|^2 = \|\Delta_{\text{direct}}\|^2 + \|\Delta_{\text{indirect}}\|^2$. We substitute the deterministic bounds from [](#sub-lem-direct-structural-error) and [](#sub-lem-indirect-structural-error). This gives a deterministic upper bound on the squared error for any realization of $v_2$:

$$
\|\mathbf{z}(\mathcal{S}_1, \mathbf{v}_2) - \mathbf{z}(\mathcal{S}_2, \mathbf{v}_2)\|_2^2 \le C_{S,\text{direct}} \cdot n_c + C_{S,\text{indirect}}(\mathcal{S}_1, \mathcal{S}_2) \cdot n_c^2

$$

The expected squared structural error is the expectation of the left-hand side. Since the right-hand side is a deterministic constant that does not depend on the random variable $v_2$, taking the expectation of both sides yields the final theorem.
**Q.E.D.**
:::
#### 11.2.4. General Asymptotic Behavior of the Total Standardization Error
This theorem consolidates the results from the preceding sections to establish the final asymptotic scaling law for the standardization operator ({prf:ref}`def-standardization-operator-n-dimensional`)'s continuity. This result is the cornerstone for understanding the algorithm's behavior and fundamental limitations, particularly for large swarms.
##### 11.2.4.1. Theorem: General Asymptotic Scaling of Mean-Square Standardization Error

:::{prf:theorem} General Asymptotic Scaling of Mean-Square Standardization Error
:label: thm-general-asymptotic-scaling-mean-square

The total **expected** squared error of the N-Dimensional Standardization Operator ({prf:ref}`def-standardization-operator-n-dimensional`), $\mathbb{E}[\| \mathbf{z}_1 - \mathbf{z}_2 \|_2^2]$, is bounded by the sum of the expected squared value error ($E^2_{V,ms}$) and the expected squared structural error ($E^2_{S,ms}$). Its asymptotic behavior for a large initial swarm ({prf:ref}`def-swarm-and-state-space`) size, $k_1 = |\mathcal{A}(\mathcal{S}_1)|$, is the sum of the asymptotic behaviors of these two distinct error sources:

$$
\mathbb{E}[\| \mathbf{z}_1 - \mathbf{z}_2 \|_2^2] \in O(E_{V,ms}^2(k_1, \varepsilon_{\mathrm{std}})) + O(E_{S,ms}^2(k_1, \varepsilon_{\mathrm{std}}))

$$

The specific scaling of these components is determined by the user's choices for the Raw Value ({prf:ref}`def-raw-value-operator`) Operator and Swarm ({prf:ref}`def-swarm-and-state-space`) Aggregation Operator via their axiomatic properties:
1.  **Value Error Scaling:**

$$
E_{V,ms}^2 \in O\left( \frac{k_1 \cdot (L_{m_2,M}(k_1))^2}{\sigma'^2_{\min,\text{bound}}} \cdot F_{V,ms}(k_1) \right) + O\left( \frac{k_1 \cdot (L_{m_2,M}(k_1))^2 L_{\sigma\'_{\text{reg}}}^2}{\sigma'^4_{\min,\text{bound}}} \cdot F_{V,ms}(k_1) \right)

$$

2.  **Structural Error Scaling:**

$$
E_{S,ms}^2 \in O\left(\frac{n_c}{\sigma'^2_{\min,\text{bound}}}\right) + O\left(\frac{k_1^{1+2p_{\text{worst-case}}} \cdot n_c^2 L_{\sigma\'_{\text{reg}}}^2}{\sigma'^4_{\min,\text{bound}}}\right)

$$

:::

##### 11.2.4.2. Benchmark Case Analysis: Empirical Aggregator and Distance-to-Companion Measurement
We instantiate the general asymptotic result for the most common and fundamental configuration to reveal the algorithm's practical stability limits.
*   **Choice of Swarm ({prf:ref}`def-swarm-and-state-space`) Aggregation Operator:** The **Empirical Measure Aggregator**. From [](#lem-empirical-aggregator-properties), this aggregator is **Structurally Stable** with $p_{\text{worst-case}} = -1$. Its value continuity function for the second moment scales as $L_{m2,M}(k_1) \propto k_1^{-1/2}$.
*   **Choice of Raw Value Operator ({prf:ref}`def-raw-value-operator`):** The **Distance-to-Companion Measurement**. From [](#thm-distance-operator-mean-square-continuity), its bound $F_{d,ms}$ is asymptotically constant with respect to $k_1$, so $F_{d,ms}(k_1) \in O(1)$.
**Asymptotic Analysis:**
**A. Value Error Component ($E^2_{V,ms}$):**
Substituting the benchmark scaling into the general formula:

$$
E_{V,ms}^2 \in O\left( \frac{1}{\sigma'^2_{\min,\text{bound}}} + \frac{L_{\sigma\'_{\text{reg}}}^2}{\sigma'^4_{\min,\text{bound}}} \right)

$$

The value error is constant with respect to the number of alive walker ({prf:ref}`def-walker`)s $k_1$.
**B. Structural Error Component ($E^2_{S,ms}$):**
With $p_{\text{worst-case}} = -1$:

$$
E_{S,ms}^2 \in O\left(\frac{n_c}{\sigma'^2_{\min,\text{bound}}}\right) + O\left(\frac{n_c^2 L_{\sigma\'_{\text{reg}}}^2}{k_1\sigma'^4_{\min,\text{bound}}}\right)

$$

**Conclusion for Benchmark Case:**
For this benchmark configuration, the total expected squared error for the standardization operator ({prf:ref}`def-standardization-operator-n-dimensional`) has the following asymptotic scaling for large $k_1$:

$$
\boxed{
\mathbb{E}[\| \mathbf{z}_1 - \mathbf{z}_2 \|_2^2] \in O\!\left(\frac{1}{\sigma'^2_{\min,\text{bound}}} + \frac{L_{\sigma\'_{\text{reg}}}^2}{\sigma'^4_{\min,\text{bound}}}\right) + O\!\left(\frac{n_c}{\sigma'^2_{\min,\text{bound}}}\right) + O\!\left(\frac{n_c^2 L_{\sigma\'_{\text{reg}}}^2}{k_1\sigma'^4_{\min,\text{bound}}}\right)
}

$$

##### 11.2.4.3. Implications and Interpretation
This result reveals two distinct operational regimes:
1.  **Regime 1: Normal Operation (Low Attrition).**
    If the number of status changes $n_c$ is small, the structural error terms are bounded or vanish for large $k_1$. The system's stability is dominated by the value error, which is **constant with respect to swarm size** and scales with $1/\sigma'^2_{\min,\text{bound}}$ and $L_{\sigma\'_{\text{reg}}}^2/\sigma'^4_{\min,\text{bound}}$. When the variance floor is much smaller than the smoothing parameter ($\kappa_{\text{var,min}} \ll \varepsilon_{\text{std}}^2$), this reduces to the familiar $O(\varepsilon_{\mathrm{std}}^{-6})$ sensitivity.
2.  **Regime 2: Catastrophic Collapse.**
    If a significant fraction of the swarm dies, such that $n_c \propto k_1$, then the total error **grows linearly with the initial swarm size** with coefficients proportional to $L_{\sigma\'_{\text{reg}}}^2/\sigma'^4_{\min,\text{bound}}$. In the $\kappa_{\text{var,min}} \ll \varepsilon_{\text{std}}^2$ regime this again matches the $O(k_1\varepsilon_{\mathrm{std}}^{-6})$ scaling.
### 11.3 Deterministic Lipschitz Continuity of the Patched Standardization Operator ({prf:ref}`def-standardization-operator-n-dimensional`)
The introduction of the **Regularized Standard Deviation Function** ([](#def-statistical-properties-measurement)) ($\sigma\'_{\text{reg}}$) in Section 11.1.2 provides a critical stability guarantee that is stronger than mean-square continuity. By ensuring the denominator of the standardization formula is a globally Lipschitz function of the raw value variance, the pathological sensitivity near zero-variance states is eliminated. This, in turn, enables a deterministic, worst-case **global continuity with a Lipschitz–Hölder modulus** for the entire N-Dimensional Standardization Operator ({prf:ref}`def-standardization-operator-n-dimensional`).
This property is a non-negotiable prerequisite for certain powerful long-term convergence results, such as those derived from Feynman-Kac particle system theory. The following sections provide a rigorous, self-contained proof of this property. The strategy is to deterministically decompose the total error vector, $\Deltaz = z(S_1, v_1, M) - z(S_2, v_2, M)$, into a series of manageable components and to bound the L2-norm of each component by a term proportional to the N-Particle Displacement Metric ({prf:ref}`def-n-particle-displacement-metric`) and the L2-norm of the raw value difference.
#### 11.3.1 Theorem: Decomposition of the Total Standardization Error
To establish the joint Lipschitz continuity of the standardization operator ({prf:ref}`def-standardization-operator-n-dimensional`) with respect to both the swarm state **S** and the raw value vector **v**, we first decompose the total squared error into two distinct components: a **Value Error** arising from the change in the raw value vector for a fixed swarm structure, and a **Structural Error** arising from the change in the swarm structure for a fixed raw value vector.
:::{prf:theorem} Decomposition of the Total Standardization Error
:label: thm-deterministic-error-decomposition
Let $z(S, v, M)$ be the N-Dimensional Standardization Operator ({prf:ref}`def-standardization-operator-n-dimensional`) ({prf:rraw value ({prf:ref}`def-raw-value-operator`)ation-operator-n-dimensional`). Let $S_1$ and $S_2$ be two swarm ({prf:ref}`def-swarm-and-state-space`) states, and let $v_1$ and $v_2$ be two corresponding N-dimensional raw value vectors. Let the output standardized vectors be $z_1 = z(S_1, v_1, M)$ and $z_2 = z(S_2, v_2, M)$.
The total squared Euclidean error between the output vectors is bounded by the sum of two fundamental error components:

$$
\|\mathbf{z}_1 - \mathbf{z}_2\|_2^2 \le 2 \cdot E_{V}^2(\mathcal{S}_1; \mathbf{v}_1, \mathbf{v}_2) + 2 \cdot E_{S}^2(\mathcal{S}_1, \mathcal{S}_2; \mathbf{v}_2)

$$

where the error components are defined as:
1.  **The Squared Value Error ($E_V^2$):** The deterministic squared error arising from the change in the raw value vector (from $v_1$ to $v_2$) while holding the swarm ({prf:ref}`def-swarm-and-state-space`)'s structure fixed at $S_1$.

$$
    E_{V}^2(\mathcal{S}_1; \mathbf{v}_1, \mathbf{v}_2) := \| \mathbf{z}(\mathcal{S}_1, \mathbf{v}_1, M) - \mathbf{z}(\mathcal{S}_1, \mathbf{v}_2, M) \|_2^2

    $$
2.  **The Squared Structural Error ($E_S^2$):** The deterministic squared error arising from the change in the swarm ({prf:ref}`def-swarm-and-state-space`)'s structure (from $S_1$ to $S_2$) while using the fixed raw value vector $v_2$.

$$

    E_{S}^2(\mathcal{S}_1, \mathcal{S}_2; \mathbf{v}_2) := \| \mathbf{z}(\mathcal{S}_1, \mathbf{v}_2, M) - \mathbf{z}(\mathcal{S}_2, \mathbf{v}_2, M) \|_2^2

$$
:::
:::{prf:proof}
:label: proof-thm-deterministic-error-decomposition
**Proof.**
The proof follows from decomposing the total error using an intermediate vector and then applying the triangle inequality. Let the intermediate vector be $z_{\text{in}}ter := z(S_1, v_2, M)$, which uses the second raw value ({prf:ref}`def-raw-value-operator`) vector with the first swarm ({prf:ref}`def-swarm-and-state-space`)'s structure.
The total error vector is $z_1 - z_2 = (z_1 - z_{\text{in}}ter) + (z_{\text{in}}ter - z_2)$.
The total squared error is bounded using the elementary inequality $\|A+B\|_2^2 \leq 2(\|A\|_2^2 + \|B\|_2^2)$:

$$

\| \mathbf{z}_1 - \mathbf{z}_2 \|_2^2 \le 2 \| \mathbf{z}_1 - \mathbf{z}_{\text{inter}} \|_2^2 + 2 \| \mathbf{z}_{\text{inter}} - \mathbf{z}_2 \|_2^2

$$
The first term on the right-hand side is the squared Value Error, $E_V^2$, as it arises from the change $v_1 → v_2$ for a fixed structure $S_1$. The second term is the squared Structural Error, $E_S^2$, as it arises from the change $S_1 → S_2$ for a fixed value vector $v_2$. This completes the decomposition.
**Q.E.D.**
:::
#### 11.3.2 Sub-Lemma: Algebraic Decomposition of the Value Error
To bound the squared value error, $E_V^2$, we first perform a purely algebraic decomposition of the error vector $\Deltaz = z(S, v_1, M) - z(S, v_2, M)$ for a fixed swarm ({prf:ref}`def-swarm-and-state-space`) state **S**. This decomposition isolates the different sources of error: the direct change in the raw values, the change in the computed mean, and the change in the computed standard deviation.
:::{prf:lemma} Algebraic Decomposition of the Value Error
:labraw value ({prf:ref}`def-raw-value-operator`)itz-value-error-decomposition
Let **S** be a fixed swarm ({prf:ref}`def-swarm-and-state-space`) state with alive set ({prf:ref}`def-alive-dead-sets`) **A** of size **k**. Let $v_1$ and $v_2$ be two raw value vectors for the alive set. Let $(\mu_1, \sigma'_1)$ and $(\mu_2, \sigma'_2)$ be the corresponding statistical properties, and let $z_1$ and $z_2$ be the corresponding standardized vectors.
The total value error vector, $\Deltaz = z_1 - z_2$, can be expressed as the sum of three components:

$$

\Delta\mathbf{z} = \Delta_{\text{direct}} + \Delta_{\text{mean}} + \Delta_{\text{denom}}

$$
where:
1.  **The Direct Shift ($\Delta_{\text{direct}}$):** The error from the change in the raw value vector itself, scaled by the initial standard deviation.

$$

    \Delta_{\text{direct}} := \frac{\mathbf{v}_1 - \mathbf{v}_2}{\sigma'_1}

$$
2.  **The Mean Shift ($\Delta_mean$):** The error from the change in the aggregator's computed mean, applied uniformly to all walker ({prf:ref}`def-walker`)s.

$$

    \Delta_{\text{mean}} := \frac{\mu_2 - \mu_1}{\sigma'_1} \cdot \mathbf{1}

$$
where $**1**$ is a k-dimensional vector of ones.
3.  **The Denominator Shift ($\Delta_denom$):** The error from the change in the regularized standard deviation, which rescales the second standardized vector.

$$

    \Delta_{\text{denom}} := \mathbf{z}_2 \cdot \frac{\sigma'_2 - \sigma'_1}{\sigma'_1}

$$
Furthermore, the total squared error is bounded by three times the sum of the squared norms of these components:

$$

\|\Delta\mathbf{z}\|_2^2 \le 3\left( \|\Delta_{\text{direct}}\|_2^2 + \|\Delta_{\text{mean}}\|_2^2 + \|\Delta_{\text{denom}}\|_2^2 \right)

$$
:::
:::{prf:proof}
:label: proof-line-3447
**Proof.**
The proof of the decomposition is a direct algebraic manipulation.
1.  **Start with the Definition of the Error.**
    The total error is $\Deltaz = z_1 - z_2 = (v_1 - \mu_1) / \sigma'_1 - (v_2 - \mu_2) / \sigma'_2$.
2.  **Decomposition.**
    We add and subtract terms to isolate the desired components.

$$

    \Delta\mathbf{z} = \frac{\mathbf{v}_1 - \mu_1}{\sigma'_1} - \frac{\mathbf{v}_2 - \mu_2}{\sigma'_1} + \frac{\mathbf{v}_2 - \mu_2}{\sigma'_1} - \frac{\mathbf{v}_2 - \mu_2}{\sigma'_2}

$$
$$

    = \left( \frac{\mathbf{v}_1 - \mathbf{v}_2}{\sigma'_1} \right) + \left( \frac{\mu_2 - \mu_1}{\sigma'_1} \cdot \mathbf{1} \right) + \left( \frac{\mathbf{v}_2 - \mu_2}{\sigma'_1} - \frac{\mathbf{v}_2 - \mu_2}{\sigma'_2} \right)

    $$
The final term can be rewritten by factoring out $(v_2 - \mu_2)$:

$$
    = \Delta_{\text{direct}} + \Delta_{\text{mean}} + (\mathbf{v}_2 - \mu_2) \left(\frac{1}{\sigma'_1} - \frac{1}{\sigma'_2}\right) = \Delta_{\text{direct}} + \Delta_{\text{mean}} + \frac{\mathbf{v}_2 - \mu_2}{\sigma'_2} \frac{\sigma'_2 - \sigma'_1}{\sigma'_1}

$$

Recognizing that $(v_2 - \mu_2) / \sigma'_2$ is $z_2$, this matches the definition of $\Delta_denom$.
3.  **Bound on the Squared Norm.**
    The bound on the total squared norm follows directly from the triangle inequality and the elementary inequality $(a+b+c)^2 \leq 3(a^2+b^2+c^2)$.
**Q.E.D.**
:::
#### 11.3.3 Theorem: Bounding the Squared Value Error
With the algebraic decomposition in place, we can now establish a deterministic bound for the Squared Value Error, $E_V^2$, in terms of the squared norm of the raw value difference. This theorem demonstrates that the standardization operator ({prf:ref}`def-standardization-operator-n-dimensional`) is Lipschitz continuous with respect to its raw value vector input for a fixed swarm structure.
:::{prf:theorem} Bounding the Squared Value Error
:label: thm-lipschitz-value-error-bound
Let **S** be a fixed swarm ({prf:ref}`def-swarm-and-state-space`) state. Let $v_1$ and $v_2$ be lipschitz ({prf:ref}`axiom-reward-regularity`)ors. The squared value error, $E_V^2(S; v_1, v_2) = \|z(S, v_1, M) - z(S, v_2, M)\|_2^2$, is deterministically bounded as follows:

$$
E_{V}^2(\mathcal{S}; \mathbf{v}_1, \mathbf{v}_2) \le C_{V,\teraw value ({prf:ref}`def-raw-value-operator`)l{S}) \cdot \|\mathblipschitz ({prf:ref}`axiom-reward-regularity`)}_2\|_2^2

$$

where $C_{V,total}(S)$ is the **Total Value Error Coefficient**, a deterministic, finite constant that depends on the state **S** but not on the raw value vectors, as formally defined in the subsequent section.
:::
:::{prf:proof}
:label: proof-thm-lipschitz-value-error-bound
**Proof.**
The proof proceeds by bounding the squared L2-norm of each of the three components from the algebraic decomposition in [](#sub-lem-lipschitz-value-error-decomposition) and then summing them.
1.  **Bound the Direct Shift Component ($\Delta_{\text{direct}}$):**
    The squared norm is $\|(v_1 - v_2) / \sigma'_1\|_2^2 = (1/(\sigma'_1)^2)\|v_1 - v_2\|_2^2$. From the definition of the Regularized Standard Deviation Function ([](#def-statistical-properties-measurement)), the denominator $\sigma'_1$ is always bounded below by $\sigma'_{\min\,\text{bound}}$. Therefore, $1/(\sigma'_1)^2 \le 1/\sigma'^2_{\min\,\text{bound}}$. This gives:

$$
    \|\Delta_{\text{direct}}\|_2^2 \le \frac{1}{\sigma'^2_{\min\,\text{bound}}} \cdot \|\mathbf{v}_1 - \mathbf{v}_2\|_2^2

$$

2.  **Bound the Mean Shift Component ($\Delta_mean$):**
    The squared norm is $k \cdot (\mu_2 - \mu_1)^2 / (\sigma'_1)^2$. Using the axiomatic value continuity of the mean ($|\mu_2 - \mu_1| \leq L_{\mu,M}(S) \|v_1 - v_2\|_2$), this is bounded by:

$$
    \|\Delta_{\text{mean}}\|_2^2 \le \frac{k \cdot (L_{\mu,M}(\mathcal{S}))^2}{\sigma'^2_{\min\,\text{bound}}} \cdot \|\mathbf{v}_1 - \mathbf{v}_2\|_2^2

$$

3.  **Bound the Denominator Shift Component ($\Delta_denom$):**
    The squared norm is $\|z_2\|_2^2 \cdot (\sigma'_2 - \sigma'_1)^2 / (\sigma'_1)^2$. We bound each term:
    *   From [](#thm-z-score-norm-bound), $\|z_2\|_2^2 \leq k\big(2V_{\max}/\sigma'_{\min\,\text{bound}}\big)^2$.
    *   From the proven value continuity of the smoothed standard deviation ([](#lem-stats-value-continuity)), $(\sigma'_2 - \sigma'_1)^2 \leq (L_{\sigma',M}(S))^2 \|v_1 - v_2\|_2^2$.
    *   The term $1/(\sigma'_1)^2$ is bounded by $1/\sigma'^2_{\min\,\text{bound}}$.
    Combining these gives a bound of the form $C \cdot \|v_1 - v_2\|_2^2$ for this component as well.
4.  **Combine the Bounds:**
    Substituting the bounds for each of the three components into the inequality from [](#sub-lem-lipschitz-value-error-decomposition) ($\|\Deltaz\|_2^2 \leq 3(\|\Delta_{\text{direct}}\|_2^2 + ...)$), and factoring out the common term $\|v_1 - v_2\|_2^2$, yields the final result. The sum of the coefficients for each component, multiplied by 3, constitutes the **Total Value Error Coefficient**, $C_{V,total}(S)$. Since all constituent parts are finite for a given state **S**, $C_{V,total}(S)$ is a finite constant.
**Q.E.D.**
:::
#### 11.3.4 Definition: Value Error Coefficients
To formalize the result of the preceding theorem and provide modular components for the final proof, we explicitly define the coefficients used in the bound for the Squared Value Error. These coefficients are deterministic functions of a fixed swarm ({prf:ref}`def-swarm-and-state-space`) state **S**.
:::{prf:definition} Value Error Coefficients
:label: def-lipschitz-value-error-coefficients
Let **S** be a fixed swarm ({prf:ref}`def-swarm-and-state-space`) state with alive set ({prf:ref}`def-alive-dead-sets`) **A** of size **k**, and let **M** be the chosen **Swarm Aggregation Operator**. Let

$$
\sigma'_{\min\,\text{bound}} := \sqrt{\kappa_{\text{var,min}} + \varepsilon_{\text{std}}^2}

$$

be the uniform lower bound on the regularized standard deviation. The coefficients for the value error bounds are defined as follows:
1.  **The Direct Shift Coefficient ($C_V,direct$):**

$$
    C_{V,\text{direct}} := \frac{1}{\sigma'^2_{\min\,\text{bound}}}

$$

2.  **The Mean Shift Coefficient ($C_V,\mu(S)$):**

$$
    C_{V,\mu}(\mathcal{S}) := \frac{k \cdot (L_{\mu,M}(\mathcal{S}))^2}{\sigma'^2_{\min\,\text{bound}}}

$$

3.  **The Denominator Shift Coefficient ($C_V,\sigma(S)$):**

$$
    C_{V,\sigma}(\mathcal{S}) := k \left( \frac{2V_{\max}}{\sigma'_{\min\,\text{bound}}} \right)^2 \left( \frac{L_{\sigma',M}(\mathcal{S})}{\sigma'_{\min\,\text{bound}}} \right)^2

$$

4.  **The Total Value Error Coefficient ($C_V,total(S)$):** The composite coefficient that bounds the total squared error.

$$
    C_{V,\text{total}}(\mathcal{S}) := 3 \cdot \left( C_{V,\text{direct}} + C_{V,\mu}(\mathcal{S}) + C_{V,\sigma}(\mathcal{S}) \right)

$$

where $L_{\mu,M}(S)$ and $L_{\sigma',M}(S)$ are the value Lipschitz functions for the aggregator's mean and regularized standard deviation, respectively, as defined in [](#lem-stats-value-continuity).
:::
#### 11.3.5 Theorem: Bounding the Squared Structural Error
We now turn to bounding the Squared Structural Error, $E_S^2$, which arises from the change in the swarm's structure (the set of alive walker ({prf:ref}`def-walker`)s) for a fixed raw value vector. This theorem demonstrates that the standardization operator's continuity with respect to structural changes is not strictly Lipschitz, but has a non-linear, Hölder-type component.
:::{prf:theorem} Bounding the Squared Structural Error
:label: thm-lipschitz-structural-error-bound
Let **v** be a fixed raw value ({prf:ref}`def-raw-value-operator`) vector. Let $S_1$ and $S_2$ be two swarm ({prf:ref}`def-swarm-and-state-space`) states. The squared structural error, $E_S^2(S_1, S_2; v) = \|z(S_1, v, M) - z(S_2, v, M)\|_2^2$, is deterministically bounded as follows:

$$
E_{S}^2(\mathcal{S}_1, \mathcal{S}_2; \mathbf{v}) \le C_{S,\text{direct}} \cdot n_c(\mathcal{S}_1, \mathcal{S}_2) + C_{S,\text{indirect}}(\mathcal{S}_1, \mathcal{S}_2) \cdot n_c(\mathcal{S}_1, \mathcal{S}_2)^2

$$

where $C_{S,direct}$ and $C_{S,indirect}(S_1, S_2)$ are the **Structural Error Coefficients**, which are deterministic, finite coefficients formally defined in the subsequent section. The presence of the $n_c^2$ term confirms that the error is not linearly proportional to the number of status changes.
:::
:::{prf:proof}
:label: proof-thm-lipschitz-structural-error-bound
**Proof.**
The proof proceeds by decomposing the total structural error vector $\Deltaz = z(S_1, v) - z(S_2, v)$ into two orthogonal components: a "direct" error from walker ({prf:ref}`def-walker`)s whose status changes, and an "indirect" error affecting walkers whose status is stable.
1.  **Decomposition of Structural Error:** The N-dimensional error vector $\Deltaz$ is partitioned based on walker ({prf:ref}`def-walker`) indices. The squared norm is the sum of the squared norms over these disjoint sets:

$$
    \|\Delta\mathbf{z}\|_2^2 = \|\Delta_{\text{direct}}\|_2^2 + \|\Delta_{\text{indirect}}\|_2^2

$$

*   $\Delta_{\text{direct}}$ has non-zero components only for indices **i** where $s_{1,i} ≠ s_{2,i}$.
    *   $\Delta_{\text{indirect}}$ has non-zero components only for indices **i** where $s_{1,i} = s_{2,i} = 1$.
2.  **Bound the Direct Error Component ($\Delta_{\text{direct}}$):**
    This component has $n_c$ non-zero terms. For each such term **i**, one of $z_{1,i}$ or $z_{2,i}$ is zero. The other is a valid Z-score, whose magnitude is bounded by $|z_j| \leq 2V_{\max} / \sigma'_{\min\,\text{bound}}$. The squared error for this component is thus bounded by $(2V_{\max} / \sigma'_{\min\,\text{bound}})^2$. Summing over all $n_c$ unstable walker ({prf:ref}`def-walker`)s gives a bound that is linear in $n_c$:

$$
    \|\Delta_{\text{direct}}\|_2^2 \le \left( \frac{2V_{\max}}{\sigma'_{\min\,\text{bound}}} \right)^2 n_c(\mathcal{S}_1, \mathcal{S}_2)

$$

3.  **Bound the Indirect Error Component ($\Delta_{\text{indirect}}$):**
    For the $k_{\text{stable}}$ walker ({prf:ref}`def-walker`)s that are alive in both swarms, the error $z_{1,i} - z_{2,i}$ is decomposed into a mean-shift part and a denominator-shift part. Using $\|a+b\|^2 \leq 2(\|a\|^2 + \|b\|^2)$, we bound the sum of these errors over all stable walkers.
    *   The mean shift error is bounded by $2k_{\text{stable}} \cdot ((\mu_1 - \mu_2)/\sigma'_1)^2$. Using the structural continuity of the mean ($|\mu_1-\mu_2|^2 \leq (L_{\mu,S})^2 (n_c)^2$), this term is bounded by an expression proportional to $n_c^2$.
    *   The denominator shift error is bounded by $2\|z_1\|^2((\sigma'_1-\sigma'_2)/\sigma'_2)^2$. Using the structural continuity of $\sigma'$ ($|\sigma'_1-\sigma'_2|^2 \leq (L_{\sigma',S})^2 (n_c)^2$), this term is also bounded by an expression proportional to $n_c^2$.
    The sum of these two terms gives a total bound for the indirect error that is quadratic in $n_c$.
4.  **Combine the Bounds:**
    Summing the bounds for the direct (linear in $n_c$) and indirect (quadratic in $n_c$) components gives the final bound as stated in the theorem.
**Q.E.D.**
:::
#### 11.3.6 Definition: Structural Error Coefficients
To formalize the result of the preceding theorem, we explicitly define the coefficients used in the bound for the Squared Structural Error. These coefficients are deterministic functions of the two swarm ({prf:ref}`def-swarm-and-state-space`) states, $\mathcal{S}_1$ and $\mathcal{S}_2$.
:::{prf:definition} Structural Error Coefficients
:label: def-lipschitz-structural-error-coefficients
Let $\mathcal{S}_1$ and $\mathcal{S}_2$ be two swarm ({prf:ref}`def-swarm-and-state-space`) states with alive set ({prf:ref}`def-alive-dead-sets`)s $\mathcal{A}_1$ and $\mathcal{A}_2$, of sizes $k_1:=|\mathcal{A}_1|$ and $k_2:=|\mathcal{A}_2|$. Let $k_{\text{stable}}:=|\mathcal{A}_1\cap\mathcal{A}_2|$. Let

$$
\sigma'_{\min\,\text{bound}} := \sqrt{\kappa_{\text{var,min}} + \varepsilon_{\text{std}}^2}

$$

be a uniform lower bound on the regularized standard deviation. The coefficients for the structural error bounds are defined as follows:
1.  **The Direct Structural Error Coefficient ($C_{S,\text{direct}}$):** The coefficient of the term linear in $n_c$.

$$
    C_{S,\text{direct}} := \left( \frac{2V_{\max}}{\sigma'_{\min\,\text{bound}}} \right)^2

$$

2.  **The Indirect Structural Error Coefficient ($C_{S,\text{indirect}}(\mathcal{S}_1, \mathcal{S}_2)$):** The coefficient of the term quadratic in $n_c$. This coefficient bounds the error for the stable walker ({prf:ref}`def-walker`)s.

$$
    C_{S,\text{indirect}}(\mathcal{S}_1, \mathcal{S}_2) := 2 k_{\text{stable}} \frac{(L_{\mu,S}(\mathcal{S}_1, \mathcal{S}_2))^2}{\sigma'^{2}_{\min\,\text{bound}}} + 2 k_1 \left(\frac{2V_{\max}}{\sigma'_{\min\,\text{bound}}}\right)^2 \frac{(L_{\sigma',S}(\mathcal{S}_1, \mathcal{S}_2))^2}{\sigma'^{2}_{\min\,\text{bound}}}

$$

where $L_{\mu,S}$ and $L_{\sigma',S}$ are the structural continuity functions for the aggregator's mean and regularized standard deviation, as defined in [](#lem-stats-structural-continuity).
:::
#### 11.3.7 Theorem: Global Continuity of the Patched Standardization Operator ({prf:ref}`def-standardization-operator-n-dimensional`)
By combining the bounds for the value error and the structural error, we can now state the final deterministic continuity property of the patched standardization operator ({prf:ref}`def-standardization-operator-n-dimensional`). The operator is not globally Lipschitz, but it is jointly continuous with a well-defined Lipschitz-Hölder structure.
:::{prf:theorem} Global Continuity of the Patched Standardization Operator
:label: thm-global-continuity-patched-standardization
Let $z(\mathcal{S}, v, M)$ be the N-Dimensional Standardization Operator ({prf:ref}`def-standardization-operator-n-dimensional`) using thraw value ({prf:ref}`def-raw-value-operator`)andard Deviation Function** ([](#def-statistical-properties-measurement)). Let $\mathcal{S}_1$ and $\mathcal{S}_2$ be two swarm ({prf:ref}`def-swarm-and-state-space`) states, and let $\mathbf{v}_1$ and $\mathbf{v}_2$ be two corresponding N-dimensional raw value vectors.
The squared Euclidean error between the output standardized vectors, $\|z(\mathcal{S}_1, \mathbf{v}_1, M) - z(\mathcal{S}_2, \mathbf{v}_2, M)\|_2^2$, is deterministically bounded by a function of the swarm ({prf:ref}`def-swarm-and-state-space`) displacement and the raw value difference:

$$
\|\mathbf{z}_1 - \mathbf{z}_2\|_2^2 \le 2 C_{V,\text{total}}(\mathcal{S}_1) \cdot \|\mathbf{v}_1 - \mathbf{v}_2\|_2^2 + 2 C_{S,\text{direct}} \cdot n_c(\mathcal{S}_1, \mathcal{S}_2) + 2 C_{S,\text{indirect}}(\mathcal{S}_1, \mathcal{S}_2) \cdot n_c(\mathcal{S}_1, \mathcal{S}_2)^2

$$

where $C_{V,\text{total}}$, $C_{S,\text{direct}}$, and $C_{S,\text{indirect}}$ are the finite, deterministic coefficients defined in [](#def-lipschitz-value-error-coefficients) and [](#def-lipschitz-structural-error-coefficients).
:::
:::{prf:proof}
:label: proof-thm-global-continuity-patched-standardization
**Proof.**
The proof is a direct assembly of the bounds derived in the preceding theorems of this section.
1.  **Decomposition of Total Error:** From [](#thm-deterministic-error-decomposition), the total squared error is bounded by the sum of the squared value error and the squared structural error:

$$
    \|\mathbf{z}_1 - \mathbf{z}_2\|_2^2 \le 2 E_{V}^2(\mathcal{S}_1; \mathbf{v}_1, \mathbf{v}_2) + 2 E_{S}^2(\mathcal{S}_1, \mathcal{S}_2; \mathbf{v}_2)

$$

2.  **Substitute the Value Error Bound:** From [](#thm-lipschitz-value-error-bound), the squared value error is bounded by:

$$
    E_{V}^2(\mathcal{S}_1; \mathbf{v}_1, \mathbf{v}_2) \le C_{V,\text{total}}(\mathcal{S}_1) \cdot \|\mathbf{v}_1 - \mathbf{v}_2\|_2^2

$$

3.  **Substitute the Structural Error Bound:** From [](#thm-lipschitz-structural-error-bound), the squared structural error is bounded by:

$$
    E_{S}^2(\mathcal{S}_1, \mathcal{S}_2; \mathbf{v}_2) \le C_{S,\text{direct}} \cdot n_c(\mathcal{S}_1, \mathcal{S}_2) + C_{S,\text{indirect}}(\mathcal{S}_1, \mathcal{S}_2) \cdot n_c(\mathcal{S}_1, \mathcal{S}_2)^2

$$

4.  **Combine the Bounds:** Substituting the bounds from steps 2 and 3 into the decomposition from step 1 yields the final inequality as stated in the theorem. This provides a complete, deterministic, worst-case bound on the operator's output error.
**Q.E.D.**
:::
## 13. Fitness potential operator
This section defines the sequence of deterministic operators that transform the raw measurement vectors (rewards and distances) into a final, N-dimensional fitness potential vector. These operators are executed after the stochastic measurement stage and are fixed for the remainder of the cloning decision process.
### 12.1 Rescaled Potential Operator for the Alive Set ({prf:ref}`def-alive-dead-sets`)
:::{prf:definition} Rescaled Potential Operator for the Alive Set
:label: def-alive-set-potential-operator
The **Rescaled Potential Operator for the Alive Set**, denoted $V_{\text{op},\mathcal{A}}$, is a deterministic function that maps the raw reward and distance vectors of an alive set ({prf:ref}`def-alive-dead-sets`) of size $k=|\mathcal{A}_t|$ to a vector of fitness potentials for that same set.
**Signature:** $V_{\text{op},\mathcal{A}}: \Sigma_N \times \mathbb{R}^k \times \mathbb{R}^k \to \mathbb{R}^k$
**Inputs:**
*   The current swarm ({prf:ref}`def-swarm-and-state-space`) state, $\mathcal{S}_t$ (used for the aggregation operators).
*   The raw reward vector for the alive set ({prf:ref}`def-alive-dead-sets`), $\mathbf{r} = (r_j)_{j \in \mathcal{A}_t}$.
*   The raw distance vector for the alive set ({prf:ref}`def-alive-dead-sets`), $\mathbf{d} = (d_j)_{j \in \mathcal{A}_t}$.
*   All relevant algorithmic parameters ($\eta, \varepsilon_{\mathrm{std}}, z_{\max}, R_{agg}, M_D, \alpha, \beta$).
**Operation:**
The operator computes the output vector $\mathbf{V}_{\mathcal{A}} = (V_i)_{i \in \mathcal{A}_t}$ as follows:
1.  **Standardize Raw Values (patched z‑score):** The **N-Dimensional Standardization Operator ({prf:ref}`def-standardization-operator-n-dimensional`)** (Def. 11.1.1) is applied independently to each raw vector using the regularized standard deviation $\sigma\'_{\text{reg}}$.
    *   Compute reward Z‑scores: $\mathbf{z_r} := z(\mathcal{S}_t, \mathbf{r}, R_{agg}, \varepsilon_{\mathrm{std}})$.
    *   Compute distance Z-scores: $\mathbf{z_d} := z(\mathcal{S}_t, \mathbf{d}, M_D, \varepsilon_{\mathrm{std}})$.
2.  **Compute Potentials:** For each walker ({prf:ref}`def-walker`) $i \in \mathcal{A}_t$:
    a.  Apply the **Smooth Piecewise Rescale Function ({prf:ref}`def-axiom-rescale-function`)** ($g_A$) and add the lower bound $\eta$ to create the rescaled components from the Z-scores $z_{i,r}$ and $z_{i,d}$:
        *   $r'_i := g_A(z_{i,r}) + \eta$
        *   $d'_i := g_A(z_{i,d}) + \eta$
    b.  Combine the components to get the final fitness potential for that walker ({prf:ref}`def-walker`):

$$
    V_i := (d'_i)^{\beta} \cdot (r'_i)^{\alpha} \quad \text{for } i \in \mathcal{A}_t

$$

**Output:** The operator returns the $k$-dimensional vector $\mathbf{V}_{\mathcal{A}} = (V_i)_{i \in \mathcal{A}_t}$.
:::
:::{prf:definition} Swarm Potential Assembly Operator
:label: def-swarm-potential-assembly-operator
The **Swarm Potential Assembly Operator**, denoted $A_{\text{pot}}$, is a deterministic function that maps the potential vector of the alive set ({prf:ref}`def-alive-dead-sets`) to the full N-dimensional fitness potential vector for the entire swarm.
**Signature:** $A_{\text{pot}}: \Sigma_N \times \mathbb{R}^{|\mathcal{A}_t|} \to \mathbb{R}^N$
**Inputs:**
*   The current swarm state ({prf:ref}`def-swarm-and-state-space`), $\mathcal{S}_t = (w_{t,i})_{i=1}^N$.
*   The potential vector for the alive set ({prf:ref}`def-alive-dead-sets`), $\mathbf{V}_{\mathcal{A}} = (V_j)_{j \in \mathcal{A}_t}$, as computed by the *Rescaled Potential Operator for the Alive Set*.
**Operation:**
The operator computes the N-dimensional output vector $\mathbf{V}_{\text{fit}} = (V_{\text{fit},i})_{i=1}^N$ as follows:
1.  Initialize an N-dimensional zero vector, $\mathbf{V}_{\text{fit}} \leftarrow \mathbf{0}$.
2.  For each walker ({prf:ref}`def-walker`) $j \in \mathcal{A}(\mathcal{S}_t)$:
    *   Let $V_j$ be the corresponding value from the input vector $\mathbf{V}_{\mathcal{A}}$.
    *   Set the $j$-th component of the output vector: $V_{\text{fit},j} := V_j$.
**Output:** The full N-dimensional fitness potential vector $\mathbf{V}_{\text{fit}}$.
:::
### 12.2. Mean-Square Continuity of the Fitness Potential Operator
The Fitness Potential Operator is the result of a multi-stage composition of functions, including the stochastic measurement operators. Its continuity is therefore analyzed in a probabilistic sense. This section proves that the operator is **mean-square continuous**, which provides a bound on the *average* squared error between the output potential vectors of two input swarms. This property is the foundation for analyzing the long-term stability and ergodicity of the system. A subsequent section will establish a stronger, deterministic continuity property required for certain convergence theorems.
The proof is built upon two key properties of the potential function: its boundedness and its Lipschitz continuity with respect to its inputs.
#### 12.2.1. Lemma: Boundedness of the Fitness Potential

:::{prf:lemma} Boundedness of the Fitness Potential
:label: lem-potential-boundedness

For any alive walker ({prf:ref}`def-walker`) $i$, its fitness potential $V_i$ is strictly positive and uniformly bounded. That is, there exist finite, state-independent constants $V_{\text{pot,min}}$ and $V_{\text{pot,max}}$ such that:

$$
0 < V_{\text{pot,min}} \le V_i \le V_{\text{pot,max}} < \infty

$$

where the bounds are defined in terms of the global algorithmic parameters:
*   $V_{\text{pot,min}} := \eta^{\alpha+\beta}$
*   $V_{\text{pot,max}} := (g_{A,\max} + \eta)^{\alpha+\beta}$
*   $g_{A,\max} := \log(1 + z_{\max}) + 1$
:::

:::{prf:proof}
:label: proof-lem-potential-boundedness
**Proof.**
The proof follows from the definition of the potential function and the properties of the rescale function ({prf:ref}`def-axiom-rescale-function`).
1.  **Bound the Rescaled Components.**
    The fitness potential is $V_i = (g_A(z_{d,i}) + \eta)^{\beta} \cdot (g_A(z_{r,i}) + \eta)^{\alpha}$.
    From the analysis of the **Smooth Piecewise Rescale Function ({prf:ref}`def-axiom-rescale-function`)** in Section 8.2, for any real input $z$, the function $g_A(z)$ is bounded on the interval $(0, g_{A,\max}]$.
    Therefore, the rescaled components $r'_i = g_A(z_{r,i}) + \eta$ and $d'_i = g_A(z_{d,i}) + \eta$ are bounded on the interval $(\eta, g_{A,\max} + \eta]$. Since $\eta > 0$, these components are always strictly positive.
2.  **Combine for Final Bounds.**
    Since $\alpha, \beta \geq 0$, the potential $V_i$ is bounded by raising these component bounds to the appropriate powers.
    *   **Lower Bound:** $V_i \geq (\eta)^\beta \cdot (\eta)^\alpha = \eta^{(\alpha+\beta)} =: V_{\text{pot,min}}$.
    *   **Upper Bound:** $V_i \leq (g_{A,\max} + \eta)^\beta \cdot (g_{A,\max} + \eta)^\alpha = (g_{A,\max} + \eta)^{(\alpha+\beta)} =: V_{\text{pot,max}}$.
This completes the proof.
**Q.E.D.**
:::
#### 12.2.2. Lemma: Lipschitz Continuity of the Fitness Potential Function

:::{prf:lemma} Lipschitz Continuity of the Fitness Potential Function
:label: lem-component-potential-lipschitz

This lemma establishes Lipschitz continuity of the fitneLipschitz ({prf:ref}`axiom-reward-regularity`)ion, building on {prf:ref}`thm-rescale-function-lipschitz` and the compositional structure of {prf:ref}`def-alive-set-potential-operator`.

Let the component-wise potential function be defined as $FLipschitz ({prf:ref}`axiom-reward-regularity`)(z_d) + \eta)^{\beta} \cdot (g_A(z_r) + \eta)^{\alpha}$. This function is Lipschitz continuous with respect to its Z-score inputs. For any two pairs of Z-scores $(z_{r1}, z_{d1})$ and $(z_{r2}, z_{d2})$:

$$
|F(z_{r1}, z_{d1}) - F(z_{r2}, z_{d2})| \le L_{F,r}|z_{r1} - z_{r2}| + L_{F,d}|z_{d1} - z_{d2}|

$$

where the Lipschitz constants $L_{F,r}$ and $L_{F,d}$ are finite, state-independent constants.
:::

:::{prf:proof}
:label: proof-lem-component-potential-lipschitz
**Proof.**
The proof proceeds by bounding the partial derivatives of $F$ with respect to its inputs, $z_r$ and $z_d$.
1.  **Partial Derivative with respect to $z_r$:**

$$
\frac{\partial F}{\partial z_r} = (g_A(z_d) + \eta)^{\beta} \cdot \left[ \alpha (g_A(z_r) + \eta)^{\alpha-1} \cdot g'_A(z_r) \right]

$$

We bound the absolute value of each term in this product:
    *   $|(g_A(z_d) + \eta)^\beta| \leq (g_{A,\max} + \eta)^\beta$.
    *   $|\alpha| = \alpha$.
    *   $|(g_A(z_r) + \eta)^{(\alpha-1)}|$: If $\alpha \geq 1$, this is bounded by $(g_{A,\max} + \eta)^{(\alpha-1)}$. If $\alpha < 1$, this is $1/(g_A(z_r)+\eta)^{(1-\alpha)}$, which is bounded by $1/\eta^{(1-\alpha)}$. In both cases, this term is uniformly bounded.
    *   $|g'_A(z_r)| \leq L_{g_A}$ from [](#thm-rescale-function-lipschitz).
    Since each term in the product is uniformly bounded by a finite constant, the partial derivative $\partial F/\partial z_r$ is uniformly bounded. Let this bound be $L_{F,r}$.
2.  **Partial Derivative with respect to $z_d$:**
    The argument is symmetric to the one above, yielding a uniform bound $L_{F,d}$.
3.  **Conclusion:**
    Since the partial derivatives are uniformly bounded, the function $F$ is Lipschitz continuous, and the total change is bounded by the sum of the changes along each dimension, weighted by the corresponding Lipschitz constants.
**Q.E.D.**
:::
#### 12.2.4. Sub-Lemma: Bounding the Expected Error from Unstable Walker ({prf:ref}`def-walker`)s
:::{prf:lemma} Bounding the Expected Error from Unstable Walkers
:label: lem-sub-potential-unstable-error-mean-square

This lemma bounds the error contribution from unstable walkers in {prf:ref}`def-alive ({prf:ref}`def-alive-dead-sets`)-set-potential-operator`.

The expected squared error component from walker ({prf:ref}`def-walker`)s changing their survival status is bounded deterministically by the number of status changes.

$$
E_{\text{unstable,ms}}^2(\mathcal{S}_1, \mathcal{S}_2) := \mathbb{E}\left[\sum_{i \in \mathcal{A}_{\text{unstable}}} |V_{1,i} - V_{2,i}|^2\right] \le V_{\text{pot,max}}^2 \cdot n_c(\mathcal{S}_1, \mathcal{S}_2)

$$

:::
:::{prf:proof}
:label: proof-lem-sub-potential-unstable-error-mean-square
**Proof.**
For ({prf:ref}`def-alive-dead-sets`) ({prf:ref}`def-swarm-and-state-space`) any walker ({prf:ref}`def-walker`) $i$ in the unstable set $\mathcal{A}_{\text{unstable}}$, its survival status changes. This means one of $V_{1,i}$ or $V_{2,i}$ is zero, while the other is a non-zero potential. From [](#lem-potential-boundedness), any non-zero potential is bounded by $V_{\text{pot,max}}$. Thus, the squared difference $|V_{1,i} - V_{2,i}|^2$ is deterministically bounded by $V_{\text{pot,max}}^2$.
The total squared error from this set is therefore bounded by the number of unstable walker ({prf:ref}`def-walker`)s ($n_c$) multiplied by this bound: $V_{\text{pot,max}}^2 \cdot n_c(\mathcal{S}_1, \mathcal{S}_2)$. Since this bound is a deterministic constant, its expectation is the constant itself.
**Q.E.D.**
:::
#### 12.2.5. Sub-Lemma: Bounding the Expected Error from Stable Walker ({prf:ref}`def-walker`)s

:::{prf:lemma} Bounding the Expected Error from Stable Walkers
:label: lem-sub-potential-stable-error-mean-square

This lemma bounds the stable walker error by combining {prf:ref}`lem-component-potential-lipschitz` with the standardization continuity from {prf:ref}`thm-standardization-operator-unified-mean-square-continuity`.

The expected squared error component from walker ({prf:ref}`def-walker`)s that remain alive ({prf:ref}`def-alive-dead-sets`) in both states ($\mathcal{A}_{\text{stable}} = \mathcal{A}(\mathcal{S}_1) \cap \mathcal{A}(\mathcal{S}_2)$), denoted $E^2_{\text{stable,ms}}$, is bounded in terms of the mean-square continuity of the underlying standardization pipelines.

$$
E_{\text{stable,ms}}^2(\mathcal{S}_1, \mathcal{S}_2) \le 2L_{F,r}^2 \cdot \mathbb{E}[\|\Delta\mathbf{z}_r\|_2^2] + 2L_{F,d}^2 \cdot \mathbb{E}[\|\Delta\mathbf{z}_d\|_2^2]

$$

where:
*   $L_{F,r}$ and $L_{F,d}$ are the component-wise Lipschitz constants for the potential function from [](#lem-component-potential-lipschitz).
*   $\mathbb{E}[\|\Delta\mathbf{z}_r\|_2^2]$ and $\mathbb{E}[\|\Delta\mathbf{z}_d\|_2^2]$ are the total expected squared error bounds for the **reward standardization pipeline** and **distance standardization pipeline**, respectively. These bounds are given by **[](#thm-standardization-operator-unified-mean-square-continuity)**.
:::

:::{prf:proof}
:label: proof-lem-sub-potential-stable-error-mean-square
**Proof.**
The proof proceeds by applying the Lipschitz continuity of the fitness potential function and then taking the expectation.
1.  **Bound the Single-Walker ({prf:ref}`def-walker`) Error:**
    For ({prf:ref}`def-alive-dead-sets`) any stable walker ({prf:ref}`def-walker`) $i \in \mathcal{A}_{\text{stable}}$, its fitness potential $V_i$ is a function of its reward Z-score $z_{r,i}$ and its distance Z-score $z_{d,i}$. From the Lipschitz continuity of the component-wise potential function ([](#lem-component-potential-lipschitz)) and the inequality $(a+b)^2 \leq 2a^2 + 2b^2$, we can bound the squared error for this single walker:

$$
|V_{1,i} - V_{2,i}|^2 \le \left(L_{F,r}|\Delta z_{r,i}| + L_{F,d}|\Delta z_{d,i}|\right)^2 \le 2L_{F,r}^2|\Delta z_{r,i}|^2 + 2L_{F,d}^2|\Delta z_{d,i}|^2

$$

where $\Delta z_{r,i}$ and $\Delta z_{d,i}$ are the changes in the $i$-th components of the reward and distance standardized vectors, respectively.
2.  **Sum Over All Stable Walker ({prf:ref}`def-walker`)s:**
    The total squared error for the stable set is the sum of the individual squared errors. The sum over the stable subset is less than or equal to the sum over all $N$ walker ({prf:ref}`def-walker`)s, which is the full squared L2-norm of the error vectors:

$$
\sum_{i \in \mathcal{A}_{\text{stable}}} |V_{1,i} - V_{2,i}|^2 \le 2L_{F,r}^2 \|\Delta\mathbf{z}_r\|_2^2 + 2L_{F,d}^2 \|\Delta\mathbf{z}_d\|_2^2

$$

3.  **Take the Expectation:**
    We take the expectation of both sides of the inequality. By linearity of expectation, we get:

$$
E_{\text{stable,ms}}^2 = \mathbb{E}\left[\sum_{i \in \mathcal{A}_{\text{stable}}} |V_{1,i} - V_{2,i}|^2\right] \le 2L_{F,r}^2 \mathbb{E}[\|\Delta\mathbf{z}_r\|_2^2] + 2L_{F,d}^2 \mathbb{E}[\|\Delta\mathbf{z}_d\|_2^2]

$$

The terms on the right are precisely the mean-square error bounds for the standardization pipelines, which are functions of the input displacement and are derived in Section 11.2.
**Q.E.D.**
:::
### 12.3 Deterministic Continuity of the Fitness Potential Operator
While mean-square continuity is sufficient for analyzing the system's average behavior and proving ergodicity, certain powerful convergence theorems (such as those from Feynman-Kac theory) require a stronger, deterministic, worst-case guarantee on the system's interaction potential. This section establishes this stronger property.
By leveraging the deterministic Lipschitz-Hölder continuity of the patched standardization operator ({prf:ref}`def-standardization-operator-n-dimensional`) (proven in Section 11.3), we prove that the composite **Fitness Potential Operator** is also deterministically continuous. This result is a non-negotiable prerequisite for the Feynman-Kac convergence analysis presented in the `04_convergence.md` document.
#### 12.3.1 Theorem: Deterministic Continuity of the Fitness Potential Operator

:::{prf:theorem} Deterministic Continuity of the Fitness Potential Operator
:label: thm-deterministic-potential-continuity

Let the Fitness Potential Operator $V_{\text{op}}$ be constructed using the patched **N-Dimensional Standardization Operator ({prf:ref}`def-standardization-operator-n-dimensional`)** ({prf:ref}`def-algorithmic-space-generic`) ({prf:ref}`def-standardization-operator-n-dimensional`). Let $(\mathcal{S}_1, \mathbf{v}_{r1}, \mathbf{v}_{d1})$ and $(\mathcal{S}_2, \mathbf{v}_{r2}, \mathbf{v}_{d2})$ be two sets of inputs, consisting of swarm ({prf:ref}`def-swarm-and-state-space`) states and their corresponding raw reward and distance vectors. Let $\mathbf{V}_1$ and $\mathbf{V}_2$ be the resulting N-dimensional fitness potential vectors.
The squared Euclidean error between the output potential vectors is deterministically bounded by a function of the swarm ({prf:ref}`def-swarm-and-state-space`) displacement and the raw value ({prf:ref}`def-raw-value-operator`) differences:

$$
\|\mathbf{V}_1 - \mathbf{V}_2\|_2^2 \le F_{\text{pot,det}}(\mathcal{S}_1, \mathcal{S}_2, \mathbf{v}_{r1}, \mathbf{v}_{r2}, \mathbf{v}_{d1}, \mathbf{v}_{d2})

$$

where $F_{\text{pot,det}}$ is a deterministic bounding function that is jointly continuous in its arguments and vanishes as $d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1, \mathcal{S}_2) \to 0$, $\|\mathbf{v}_{r1} - \mathbf{v}_{r2}\|_2 \to 0$, and $\|\mathbf{v}_{d1} - \mathbf{v}_{d2}\|_2 \to 0$.
:::

#### 12.3.2 Proof of Deterministic Continuity for the Fitness Potential Operator
:label: proof-deterministic-potential-continuity
:::{prf:proof}
:label: proof-proof-deterministic-potential-continuity
**Proof.**
The proof proceeds by deterministically decomposing the total error and applying the established continuity properties of the constituent operators.
1.  **Decomposition of Total Error:** The total squared error is decomposed into contributions from unstable walker ({prf:ref}`def-walker`)s (whose status changes) and stable walkers.

$$
    \|\mathbf{V}_1 - \mathbf{V}_2\|_2^2 = \sum_{i \in \mathcal{A}_{\text{unstable}}} |V_{1,i} - V_{2,i}|^2 + \sum_{i \in \mathcal{A}_{\text{stable}}} |V_{1,i} - V_{2,i}|^2

$$

2.  **Bound the Error from Unstable Walker ({prf:ref}`def-walker`)s:**
    The error from the $n_c$ unstable walker ({prf:ref}`def-walker`)s is bounded deterministically. Since one potential is zero and the other is bounded by $V_{\text{pot,max}}$ ([](#lem-potential-boundedness)), this component is bounded by:

$$
    \sum_{i \in \mathcal{A}_{\text{unstable}}} |V_{1,i} - V_{2,i}|^2 \le V_{\text{pot,max}}^2 \cdot n_c(\mathcal{S}_1, \mathcal{S}_2)

$$

3.  **Bound the Error from Stable Walker ({prf:ref}`def-walker`)s:**
    For stable walker ({prf:ref}`def-walker`)s, the potential $V_i$ is a composite function of the standardized vectors for rewards and distance: $V_i = F(z_{r,i}, z_{d,i})$. As shown in [](#lem-component-potential-lipschitz), the function $F$ is globally Lipschitz continuous with respect to its Z-score inputs. The total squared error for the stable set is therefore bounded by a linear combination of the squared errors of the underlying standardization pipelines:

$$
    \sum_{i \in \mathcal{A}_{\text{stable}}} |V_{1,i} - V_{2,i}|^2 \le 2L_{F,r}^2 \|\mathbf{z}_{r,1} - \mathbf{z}_{r,2}\|_2^2 + 2L_{F,d}^2 \|\mathbf{z}_{d,1} - \mathbf{z}_{d,2}\|_2^2

$$

where the constants $L_{F,r}$ and $L_{F,d}$ are from [](#lem-component-potential-lipschitz).
4.  **Apply the Deterministic Bound for Standardization:**
    We now substitute the deterministic bound from [](#thm-global-continuity-patched-standardization) for both the reward and distance standardization pipelines. For $*\in\{r,d\}$ we obtain

$$
    \|\mathbf{z}_{*,1} - \mathbf{z}_{*,2}\|_2^2 \le 2 C_{V,\text{total}}^{(*)}\cdot \|\Delta\mathbf{v}_*\|_2^2 + 2 C_{S,\text{direct}}^{(*)} \cdot n_c + 2 C_{S,\text{indirect}}^{(*)}(\mathcal{S}_1,\mathcal{S}_2) \cdot n_c^2,

$$

where $C_{V,\text{total}}^{(*)}$ is defined in [](#def-lipschitz-value-error-coefficients) and $C_{S,\text{direct}}^{(*)}$, $C_{S,\text{indirect}}^{(*)}$ are from [](#def-lipschitz-structural-error-coefficients). The dependence on the swarm ({prf:ref}`def-swarm-and-state-space`) states is entirely through these deterministic coefficients.
5.  **Assemble the Final Bound `F_pot,det`:**
    Combining the bounds from steps 2–4 yields the final deterministic function $F_{\text{pot,det}}$. It is a sum of terms proportional to $\|\Delta\mathbf{v}_r\|^2$, $\|\Delta\mathbf{v}_d\|^2$, $n_c$, and $n_c^2$, with coefficients obtained by collecting $V_{\text{pot,max}}$, $L_{F,*}$, and the standardization constants $C_{V,\text{total}}^{(*)}$, $C_{S,\text{direct}}^{(*)}$, $C_{S,\text{indirect}}^{(*)}$. Since each constituent coefficient is finite by definition, $F_{\text{pot,det}}$ is a well-defined, continuous bound on the deterministic error. This completes the proof.
**Q.E.D.**
:::
#### 12.3.3 Corollary: Pipeline Continuity Under Margin-Based Status Stability
:::{prf:corollary}
:label: cor-pipeline-continuity-margin-stability

Under {prf:ref}`axiom-margin-stability`, the deterministic bound from {prf:ref}`thm-deterministic-potential-continuity` simplifies significantly, with the unstable term vanishing for small input perturbations.

Assume the **Axiom of Margin-Based Status Stability** ([](#axiom-margin-stability)). Then for all inputs
$(\mathcal{S}_1, \mathbf{v}_{r1}, \mathbf{v}_{d1})$ and $(\mathcal{S}_2, \mathbf{v}_{r2}, \mathbf{v}_{d2})$,
the deterministic bound $F_{\text{pot,det}}$ in [](#thm-deterministic-potential-continuity) satisfies

$$
F_{\text{pot,det}}(\mathcal{S}_1, \mathcal{S}_2, \mathbf{v}_{r1}, \mathbf{v}_{r2}, \mathbf{v}_{d1}, \mathbf{v}_{d2})
\;\xrightarrow[(d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1,\mathcal{S}_2),\,\|\Delta\mathbf{v}_r\|,\,\|\Delta\mathbf{v}_d\|)\to 0]{}\;0.

$$

Moreover, for $d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1,\mathcal{S}_2)^2\le r_{\mathrm{status}}$ we have $n_c=0$ and the unstable term vanishes exactly; the remaining terms are controlled by the deterministic continuity of the patched standardization operator ({prf:ref}`def-standardization-operator-n-dimensional`) and the Lipschitz continuity of the potential map $F ({prf:ref}`def-perturbation-operator`)$.
:::
## 14. The Perturbation Operator ({prf:ref}`def-perturbation-operator`)
The perturbation stage applies a random displacement to each walker in the swarm, representing an exploration step. This is a purely stochastic operator  ({prf:ref}`def-perturbation-operator`)that only affects walker positions.
### ({prf:ref}`def-perturbation-operator`) 13.1 Definition: Perturbation Operator
:::{prf:definition} Pertu ({prf:ref}`def-perturbation-operator`)rbation Operator
:label: def-perturbation-operator
The **Perturbation Operator ({prf:ref}`def-perturbation-operator`)**, denoted $\Psi_{\text{pert}}: \Sigma_N \to \mathcal{P}(\Sigma_N)$, maps an input swarm ({prf:ref}`def-swarm-and-state-space`) $\mathcal{S}_{\text{in}}$ to a distribution over swarms where only the positions have been updated.
For each walker ({prf:ref}`def-walker`) $i$, its output state $w_{\text{out},i} = (x_{\text{out},i}, s_{\text{out},i})$ is determined as follows:
1.  Its output position is sampled from the **Perturbation Measure ({prf:ref}`def-perturbation-measure`)**:

$$
x_{\text{out},i} \sim \mathcal{P}_\sigma(x_{\text{in},i}, \cdot)

$$

2.  Its status remains unchanged from the input: $s_{\text{out},i} = s_{\text{in},i}$.
The operator is the product measure of these N independent processes.
:::
:::{admonition} Randomness discipline for perturbation
:class: note
For each walker ({prf:ref}`def-walker`) $i$, sample the perturbation noise $\xi_i$ independently from the chosen noise law (Gaussian, uniform ball, heat kernel, …), using a per‑walker PRNG stream. If the noise scale $\sigma$ is adapted from $\mathcal S_t$, the mapping $\sigma(\mathcal S_t)$ is deterministic; no shared, additional randomness is introduced at this stage. This enforces within‑step independence required by Assumption A.
:::
### 13.2 Continuity of the Perturbation Operator ({prf:ref}`def-perturbation-operator`)
The Perturbation Operator ({prf:ref}`def-perturbation-operator`) is a purely stochastic operator that updates the positions of all walker ({prf:ref}`def-walker`)s in the swarm according to the **Perturbation Measure**. This section establishes the probabilistic continuity of this operator with respect to the **N-Particle Displacement ({prf:ref}`def-n-particle-displacement-metric`) Metric ($d_{\text{Disp},\mathcal{Y}}$)**. The analysis will show that for two input swarms that are close to each other, the resulting output swarms are also close with high probability. The proof relies on a fundamental axiomatic property of the chosen noise measure and projection map, which bounds the expected displacement in the algorithmic space. We will use this axiom to construct a high-probability bound on the total output displacement by decomposing the error into its positional and status components.
#### 13.2.1 Axiomatic Requirement: Bounded Second Moment of Perturbation
For the continuity of the Perturbation Operator ({prf:ref}`def-perturbation-operator`) to be well-defined, the random displacement it introduces must be statistically bounded. The user's choice of **Perturbation Measure ({prf:ref}`def-perturbation-measure`)** and **Projection Map** must satisfy an axiom that provides a uniform bound on the mean of the squared displacement in the algorithmic space.
:::{prf:axiom} Axiom of Bounded Second Moment of Perturbation
:label: def-axiom-bounded-second-moment-perturbation
This axiom constrains the {prf:ref}`def-perturbation-measure` and ensures bounded behavior in the {prf:ref}`def-algorithmic-space-generic`.

*   **Core Assumption:** The expectation of the squared displacement caused by the **Perturbation Measure ({prf:ref}`def-perturbation-measure`)**, after being projected into the **Algorithmic Space ({prf:ref}`def-algorithmic-space-generic`)**, is uniformly bounded across all possible starting positions. This ensures that, on average, walker ({prf:ref}`def-walker`)s do not experience infinite displacement.
*   **Axiomatic Parameter:** The user of this framework must provide one non-negative constant derived from their choice of operators:
    1.  **$M_{\text{pert}}^2$ (The Maximum Expected Squared Displacement):** An upper bound on the expectation of the squared displacement.
*   **Condition:** For any starting position $x_{\text{in}} \in \mathcal{X}$, let the random variable for the squared displacement be $Y := d_{\mathcal{Y}}(\varphi(x_{\text{out}}), \varphi(x_{\text{in}}))^2$ where $x_{\text{out}} \sim \mathcal{P}_\sigma(x_{\text{in}}, \cdot)$. The constant must satisfy:

$$

M_{\text{pert}}^2 \ge \sup_{x_{\text{in}} \in \mathcal{X}} \mathbb{E}[Y]

$$
*   **Framework Application:** This axiom provides a uniform bound on the *mean* of the random displacement. The *fluctuations* around this mean are bounded via **McDiarmid’s inequality** for functions of independent inputs (Assumption A), applied to the average of per‑walker ({prf:ref}`def-walker`) squared displacements. Bounded differences ({prf:ref}`thm-mcdiarmids-inequality`) hold with constants $c_i=D_{\mathcal{Y}}^2/N$ since each term lies in $[0,D_{\mathcal{Y}}^2]$ (finite diameter). No further variance assumptions are required. See Boucheron–Lugosi–Massart (Appendix B).
*   **Failure Mode Analysis:** If this axiom is violated (i.e., if the supremum is infinite), walker ({prf:ref}`def-walker`)s could be displaced by an infinite amount on average, making the operator's behavior unpredictable and breaking the continuity guarantees.
:::
#### 13.2.2. Bounding the Output Positional Displacement
The first step is to establish an algebraic bound on the positional displacement between the two output swarms. This bound decomposes the total output displacement into the initial input displacement and the random displacement introduced by the perturbation operator ({prf:ref}`def-perturbation-operator`) acting on each swarm ({prf:ref}`def-swarm-and-state-space`).
:::{prf:lemma} Bounding the Output Positional Displacement
:label: lem-sub-perturbation-positional-bound-reproof
This lemma analyzes the output displacement of the {prf:ref}`def-perturbation-operator`.

Let $\mathcal{S}_1, \mathcal{S}_2$ be two input swarm ({prf:ref}`def-swarm-and-state-space`)s, and let $\mathcal{S}'_1, \mathcal{S}'_2$ be the corresponding output swarms after applying the Perturbation Operator ({prf:ref}`def-perturbation-operator`). The total squared positional displacement between the output swarms, $\Delta_{\text{pos}}^2(\mathcal{S}'_1, \mathcal{S}'_2)$, is bounded as follows:

$$

\Delta_{\text{pos}}^2(\mathcal{S}'_1, \mathcal{S}'_2) \le 3\Delta_{\text{pos}}^2(\mathcal{S}_1, \mathcal{S}_2) + 3\Delta_{\text{pert}}^2(\mathcal{S}_1) + 3\Delta_{\text{pert}}^2(\mathcal{S}_2)

$$
where $\Delta_{\text{pert}}^2(\mathcal{S})$ is the **Total Perturbation-Induced Displacement** from [](#def-perturbation-fluctuation-bounds-reproof).
:::
:::{prf:proof}
:label: proof-lem-sub-perturbation-positional-bound-reproof

**Proof.**
For ({prf:ref}`def-algorithmic-space-generic`) ({prf:ref}`def-swarm-and-state-space`) any walker ({prf:ref}`def-walker`) $i$, by applying the triangle inequality to the distance $d_{\mathcal{Y}}(\varphi(x'_{1,i}), \varphi(x'_{2,i}))$ using the input positions as intermediate points, and then using the inequality $(a+b+c)^2 \le 3(a^2 + b^2 + c^2)$, we get the following bound on the squared distance for the $i$-th walker:

$$

d_{\mathcal{Y}}(\varphi(x'_{1,i}), \varphi(x'_{2,i}))^2 \le 3 d_{\mathcal{Y}}(\varphi(x'_{1,i}), \varphi(x_{1,i}))^2 + 3 d_{\mathcal{Y}}(\varphi(x_{1,i}), \varphi(x_{2,i}))^2 + 3 d_{\mathcal{Y}}(\varphi(x_{2,i}), \varphi(x'_{2,i}))^2

$$
Summing this inequality over all $N$ walker ({prf:ref}`def-walker`)s and recognizing that the sum of the middle terms is $\Delta_{\text{pos}}^2(\mathcal{S}_1, \mathcal{S}_2)$ and the sums of the outer terms are the definitions of $\Delta_{\text{pert}}^2(\mathcal{S}_1)$ and $\Delta_{\text{pert}}^2(\mathcal{S}_2)$ yields the stated result. This decomposition is a purely algebraic consequence of the triangle inequality and holds regardless of any statistical dependency.
**Q.E.D.**
:::
#### 13.2.3. Concentration Inequality for Total Perturbation-Induced Displacement
The algebraic bound from the previous section shows that the final positional displacement depends on the random variable $\Delta_{\text{pert}}^2(\mathcal{S})$. To establish a probabilistic continuity bound, we must find a high-probability bound for this random variable. The term $\Delta_{\text{pert}}^2(\mathcal{S})$ is a sum of the **N** random squared displacements of each walker ({prf:ref}`def-walker`), which are statistically independent for a fixed initial state $\mathcal{S}$. This allows us to apply a concentration inequality for functions of independent random variables to bound its deviation from its expected value. We will use McDiarmid's Inequality.
##### 13.2.3.0. Inputs and Lipschitz constants for McDiarmid
For clarity, we analyze the normalized functional

$$

f_{\text{avg}}\;:=\;\frac{1}{N}\,\Delta_{\text{pert}}^2(\mathcal{S}_{\text{in}})
\;=\; \frac{1}{N}\sum_{i=1}^N d_{\mathcal{Y}}\!\big(\varphi(x'_{\text{out},i}),\varphi(x_{\text{in},i})\big)^2.

$$
Under [](#axiom-bounded-algorithmic-diameter), each term is bounded in $[0, D_{\mathcal{Y}}^2]$. Changing only the $i$‑th random input can change $f_{\text{avg}}$ by at most $c_i = D_{\mathcal{Y}}^2/N$. Assumption A supplies the required independe ({prf:ref}`thm-mcdiarmids-inequality`)nce.
:::{prf:lemma} Bounded differences ({prf:ref}`thm-mcdiarmids-inequality`) for $f_{\text{avg}}$
:label ({prf:ref}`thm-mcdiarmids-inequality`): lem-bounded-differences-favg

This le ({prf:ref}`thm-mcdiarmids-inequality`)mma establishes the bounded differences condition for the perturbation displacement functional, enabling application of {prf:ref}`thm-mcdiarmids-inequality` to obtain probabilistic continuity of {prf:ref}`def-perturbation-operator`.

Under [](#axiom-bounded-algorithmic-diameter), for the normalized functional $f_{\text{avg}}$ defined above, the McDiarmid bounded‑difference constants may be taken as $c_i=D_{\mathcal{Y}}^2/N ({prf:ref}`thm-mcdiarmids-inequality`)$ for all $i$.
:::
##### 13.2.3.1. McDiarmid's Inequality (Bounded Differences Inequality)
:::{prf:theorem} McDiarmid's Inequality (Bounded Differences Inequality) (Boucheron–Lugosi–Massart)
:label: thm-mcdiarmids-inequality
This is a standard concentration inequality from probability theory, used to bound the deviation of {prf:ref}`def-perturbation-operator` from its expected behavior.

Let $X_1, X_2, \dots, X_N$ be a set of independent random variables. Let $f$ be a function of these variables, $f(X_1, \dots, X_N)$, that satisfies the **bounded differences property**. This means that for each variable $i \in \{1, \dots, N\}$, there exists a constant $c_i$ such that if only the $i$-th variable is changed, the function's value cannot change by more than $c_i$:

$$

\sup_{x_1, \dots, x_N, x'_i} |f(x_1, \dots, x_i, \dots, x_N) - f(x_1, \dots, x'_i, \dots, x_N)| \le c_i

$$
Then for any $t > 0$, the probability that the function's value deviates from its expected value by more than $t$ is bounded by:

$$

P(|f(X_1, \dots, X_N) - \mathbb{E}[f(X_1, \dots, X_N)]| \ge t) \le 2\exp\left(\frac{-2t^2}{\sum_{i=1}^N c_i^2}\right)

$$
:::
##### 13.2.3.2. Probabilistic Bound on Total Perturbation-Induced Displacement
:::{prf:lemma} Probabilistic Bound on Total Perturbation-Induced Displacement
:label: lem-sub-probabilistic-bound-perturbation-displacement-reproof
Let $\mathcal{S}_{\text{in}}$ be an input swarm ({prf:ref}`def-swarm-and-state-space`). Assume the **Axiom of Bounded Second Moment of Perturbation** ([](#def-axiom-bounded-second-moment-perturbation)) holds. Then for any probability of failure $\delta' \in (0, 1)$, the **Total Perturbation-Induced Displacement** is bounded with probability at least $1-\delta'$:

$$

\Delta_{\text{pert}}^2(\mathcal{S}_{\text{in}}) \le B_M(N) + B_S(N, \delta')

$$
where $B_M(N)$ is the **Mean Displacement Bound** and $B_S(N, \delta')$ is the **Stochastic Fluctuation Bound**, as defined in the subsequent section.
:::
:::{prf:proof}
:label: proof-lem-sub-probabilistic-bound-perturbation-displacement-reproof

**Proof.**
The proof proceeds by applying McDiarmid's Inequality ({prf:ref}`thm-mcdiarmids-inequality`) to the function that computes the total perturbation-induced displacement.
1.  **Define the Function and Independent Variables.**
    *   **Independent Variables:** The perturbation of the N-particle swarm ({prf:ref}`def-swarm-and-state-space`) is the result of **N** independent random choices made by the perturbation measure for each walker ({prf:ref}`def-walker`).
    *   **Function:** The function we wish to bound is the **Total Perturbation-Induced Displacement**, **f**, which is a function of these **N** independent random choices for a fixed initial state $\mathcal{S}_{\text{in}}$.
2.  **Prove the Bounded Differences Property.**
    We apply McDiarmid to $f_{\text{avg}}$. Changing only the **i**-th random outcome only affects the **i**-th summand. Since each summand is in $[0, D_{\mathcal{Y}}^2]$, the bounded differences constants are $c_i = D_{\mathcal{Y}}^2/N$ for all $i$.
3.  **Apply McDiarmid's Inequality and Solve for the Bound.**
    The sum of squares is $\sum_{i=1}^N c_i^2 = N\,(D_{\mathcal{Y}}^2/N)^2 = D_{\mathcal{Y}}^4/N$. McDiarmid yields, for $t>0$,

$$

    \mathbb{P}\big( |f_{\text{avg}} - \mathbb{E}[f_{\text{avg}}]| \ge t \big) \le 2\exp\!\left(\!-\,\frac{2N t^2}{ D_{\mathcal{Y}}^4}\right).

$$
Setting the failure probability to $\delta'$ and solving for $t$ gives the stochastic fluctuation bound for the average,
    $\displaystyle B_{S,\text{avg}}(N,\delta') = D_{\mathcal{Y}}^2 \sqrt{\tfrac{1}{2N}\ln\!(\tfrac{2}{\delta'})}$. Multiplying back by $N$ recovers the bound for $\Delta_{\text{pert}}^2$ used below.
4.  **Combine with Axiomatic Mean Bound.**
    The expected total displacement $E[\Delta_pert^2]$ is bounded by the **Mean Displacement Bound**, $B_M(N)$. Combining these gives the final high-probability bound.
**Q.E.D.**
:::
#### 13.2.4. Definition: Perturbation Fluctuation Bounds
This section formally defines the two components that bound the total random displacement introduced by the Perturbation Operator ({prf:ref}`def-perturbation-operator`). These terms are direct consequences of the preceding analysis and are functions of the foundational parameters of the system.
:::{prf:definition} Perturbation Fluctuation Bounds
:label: def-perturbation-fluctuation-bounds-reproof
The total random displacement introduced by the Perturbation Operator ({prf:ref}`def-perturbation-operator`) is bounded by the sum of two components: a deterministic bound on its mean and a probabilistic bound on its fluctuations.
1.  **The Mean Displacement Bound ($B_M(N)$):** A deterministic upper bound on the total expected squared displacement for a swarm ({prf:ref}`def-swarm-and-state-space`) of size N. It is derived from the **Axiom of Bounded Second Moment of Perturbation** ([](#def-axiom-bounded-second-moment-perturbation)).

$$

    B_M(N) := N \cdot M_{\text{pert}}^2

$$
2.  **The Stochastic Fluctuation Bound ($B_S(N, \delta')$):** A high-probability bound on the deviation of the total squared displacement from its mean, derived from McDiarmid's inequality in [](#sub-lem-probabilistic-bound-perturbation-displacement-reproof). For a given failure probability $\delta' \in (0, 1)$, it is defined as:

$$

    B_S(N, \delta') := D_{\mathcal{Y}}^2 \sqrt{\frac{N}{2} \ln\left(\frac{2}{\delta'}\right)}

$$
where $D_{\mathcal{Y}}$ ({prf:ref}`axiom-bounded-algorithmic-diameter`) is the diameter of the algorithmic space ({prf:ref}`def-algorithmic-space-generic`), a foundational geometric parameter.
:::
#### 13.2.5. Synthesis: The Full Probabilistic Continuity Bound for the Perturbation Operator ({prf:ref}`def-perturbation-operator`)
This theorem assembles the preceding results to provide the final, rigorous probabilistic continuity bound for the Perturbation Operator ({prf:ref}`def-perturbation-operator`).
:::{prf:theorem} Probabilistic Continuity of the Perturbation Operator
:label: thm-perturbation-operator-continuity-reproof
Let $\mathcal{S}_1$ ({prf:ref}`def-algorithmic-space-generic`) and $\mathcal{S}_2$ be two input swarm ({prf:ref}`def-swarm-and-state-space`)s. Let the output swarms be generated by independent applications of the Perturbation Operator ({prf:ref}`def-perturbation-operator`): $\mathcal{S}'_1 \sim \Psi_{\text{pert}}(\mathcal{S}_1, \cdot)$ and $\mathcal{S}'_2 \sim \Psi_{\text{pert}}(\mathcal{S}_2, \cdot)$.
Assume the chosen **Perturbation Measure ({prf:ref}`def-perturbation-measure`)** satisfies the **Axiom of Bounded Second Moment of Perturbation ([](#def-axiom-bounded-second-moment-perturbation))**.
Then for any probability of failure $\delta \in (0, 1)$, the squared **N-Particle Displacement ({prf:ref}`def-n-particle-displacement-metric`) Metric** between the two output swarms is bounded with probability at least $1-\delta$ by:

$$

d_{\text{Disp},\mathcal{Y}}(\mathcal{S}'_1, \mathcal{S}'_2)^2 \le 3 \frac{\Delta_{\text{pos}}^2(\mathcal{S}_1, \mathcal{S}_2)}{N} + \lambda_{\mathrm{status}} \frac{n_c(\mathcal{S}_1, \mathcal{S}_2)}{N} + \frac{6}{N} \left( B_M(N) + B_S(N, \delta/2) \right)

$$
:::
:::{prf:proof}
:label: proof-thm-perturbation-operator-continuity-reproof

**Proof.**
The proof constructs a high-probability bound for the output displacement metric ({prf:ref}`def-n-particle-displacement-metric`) by composing the algebraic and probabilistic bounds from the preceding lemmas.
1.  **Decomposition of the Output Metric.**
    The squared N-Particle Displacement Metric ({prf:ref}`def-n-particle-displacement-metric`) for the output swarm ({prf:ref}`def-swarm-and-state-space`)s is:

$$

    d_{\text{Disp},\mathcal{Y}}(\mathcal{S}'_1, \mathcal{S}'_2)^2 = \frac{1}{N}\Delta_{\text{pos}}^2(\mathcal{S}'_1, \mathcal{S}'_2) + \frac{\lambda_{\mathrm{status}}}{N} n_c(\mathcal{S}'_1, \mathcal{S}'_2)

$$
2.  **Bound the Components.**
    The Perturbation Operator ({prf:ref}`def-perturbation-operator`) does not alter the survival status of any walker ({prf:ref}`def-walker`), so the status change term is deterministic. The output positional displacement is bounded by [](#sub-lem-perturbation-positional-bound-reproof).
3.  **Construct a Probabilistic Bound.**
    The random variables $\Delta_{\text{pert}}^2(\mathcal{S}_1)$ and $\Delta_{\text{pert}}^2(\mathcal{S}_2)$ are independent. We use the **union bound** to establish a simultaneous high-probability bound for both terms, allocating a failure probability of $\delta' = \delta/2$ to each. From [](#sub-lem-probabilistic-bound-perturbation-displacement-reproof), both bounds hold simultaneously with probability at least $1-\delta$.
4.  **Combine All Bounds.**
    Substituting the deterministic bound for the status component and the high-probability bound for the positional component back into the metric definition gives the final result as stated in the theorem.
**Q.E.D.**
:::
:::{admonition} Scope and assumptions
:class: note
This concentration bound relies on Assumption A (in‑step independence) and on the with‑replacement sampling policy in §7.1.1. Implementations that use within‑step dependence (e.g., systematic resampling with a shared uniform) violate these assumptions; in that case use a dependent bounded‑differences inequality (see Appendix) instead of McDiarmi ({prf:ref}`def-status-update-operator`)d.
:::
## 15. The Status Update Operator ({prf:ref}`def-status-update-operator`)
After any change in position (either from cloning or perturbation), a walker ({prf:ref}`def-walker`)'s status must be re-evaluated. This operator pe ({prf:ref}`def-status-update-operator`)rforms that check deterministically.
### 14.1 Definition: Status Update Operator ({prf:ref}`def-status-update-operator`)
:::{prf:definition} Status Update Operator ({prf:ref}`def-status-update-operator`)
:label: def-status-update-operator
The **Status Update Operator ({prf:ref}`def-status-update-operator`)**, denoted $\Psi_{\text{status}}: \Sigma_N \to \Sigma_N$, is a deterministic function that maps an input swarm ({prf:ref}`def-swarm-and-state-space`) to an output swarm where only the aliveness statuses have been updated to reflect their current positions.
For each walker ({prf:ref}`def-walker`) $i$, its output state $w_{\text{out},i} = (x_{\text{out},i}, s_{\text{out},i})$ is determined as follows:
1.  Its position remains unchanged: $x_{\text{out},i} = x_{\text{in},i}$.
2.  Its output status is determined by the validity of its projected position:

$$

    s_{\text{out},i} = \mathbb{1}_{\text{valid}}(\varphi(x_{\text{in},i}))

$$
This operator is applied element-wise to all N walker ({prf:ref}`def-walker`)s.
:::
:::{admonition} Independence in probabilistic status variants
:class: note
The definition above is deterministic. If an implementation uses a probabilistic status rule (e.g., $s_{\text{out},i}\sim\mathrm{Bernoulli}(p_i)$ based on position‑dependent validity), then conditional on $\mathcal S_t$ the draws are taken independently across walker ({prf:ref}`def-walker`)s, using per‑walker uniforms. This aligns the status stage with Assumption A.
:::
### 14.2 Continuity of the Composed Post-Perturbation St ({prf:ref}`def-status-update-operator`)atus Update
The **Status Update Operator ({prf:ref}`def-status-update-operator`) ($Ψ_status$)** is a deterministic and inherently discontinuous function. Its effect on the system's stability can only be analyzed probabilistically by considering its composition with a stochastic position-update operator, such as the **Perturbation Operator ({prf:ref}`def-perturbation-operator`) ($Ψ_pert$)**. This section establishes the probabilistic continuity of this composed operator, $Ψ_status ∘ Ψ_pert$. It proves that the expected number of status changes between two output swarms is a Hölder-continuous function of the displacement between the input swarms. This result is a cornerstone of the full system's continuity proof and relies on the generalized N-particle axiom for boundary regularity ({prf:ref}`axiom-boundary-regularity`).
:::{note}
Compactness convention: When restricting to compact position sets in $\mathcal{X}$, their images under the continuous projection $\varphi$ lie in compact subsets of $\mathcal{Y}$. All continuity and Hölder estimates below are stated with respect to $d_{\text{Disp},\mathcal{Y}}$ on $\Sigma_N$.
:::
:::{prf:theorem} Probabilistic Continuity of the Post-Perturbation SBoundary ({prf:ref}`axiom-boundary-smoothness`)bel: thm-post-perturbation-status-update-continuity
Let $\mathcal{S}_1$smooth ({prf:ref}`axiom-boundary-smoothness`)al{S}_2$ be two input swarms. Let the output swarms be generated by the independent aBoundary ({prf:ref}`axiom-boundary-smoothness`)e composed operator: $\mathcal{S}'_1 \sim (\Psi_{\text{status}} \circ \Psi_{\text{pert}})(\mathcal{S}_1, \cdot)$ and $\mathcal{S}'_2 \sim (\Psi_{\text{status}} \circ \Psi_{\text{pert}})(\mathcal{S}_2, \cdot)$.
Assume the **Axiom of Boundary Regularity ({prf:ref}`axiom-boundary-regularity`) ([](#axiom-boundary-regularity))** holds.
The expected total number of status changes between the two output swarms, $\mathbb{E}[n_c(\mathcal{S}'_1, \mathcal{S}'_2)]$, is bounded by a function of the initial N-Particle Displacement ({prf:ref}`def-n-particle-displacement-metric`) Metric between the input swarms:

$$

\mathbb{E}[n_c(\mathcal{S}'_1, \mathcal{S}'_2)] \le \frac{N}{2} + N L_{\text{death}}^2 \left( d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1, \mathcal{S}_2) \right)^{2\alpha_B}

$$
where the term involving the **Boundary Instability Factor ($L_{\text{death}}$)** and **Boundary Smoothing Exponent ($\alpha_B$)** is a direct consequence of the axiom.
:::
:::{prf:proof}
:label: proof-thm-post-perturbation-status-update-continuity
**Proof.**
The proof proceeds by analyzing the expected squared difference between the final status variables for each walker ({prf:ref}`def-walker`) and then summing the results.
1.  **Decomposition of Expected Status Change:**
    The expected total status change is the sum of the expected squared differences for each walker ({prf:ref}`def-walker`):

$$

    \mathbb{E}[n_c(\mathcal{S}'_1, \mathcal{S}'_2)] = \mathbb{E}\left[\sum_{i=1}^N (s'_{1,i} - s'_{2,i})^2\right] = \sum_{i=1}^N \mathbb{E}[(s'_{1,i} - s'_{2,i})^2]

$$
For any two random variables $X, Y$, the expected squared difference can be expressed in terms of their variances and expected values: $\mathbb{E}[(X-Y)^2] = \operatorname{Var}(X) + \operatorname{Var}(Y) + (\mathbb{E}[X] - \mathbb{E}[Y])^2$. The final status variables $s'_{k,i}$ are Bernoulli random variables. Applying this identity for each walker ({prf:ref}`def-walker`) **i** gives:

$$

    \mathbb{E}[(s'_{1,i} - s'_{2,i})^2] = \operatorname{Var}[s'_{1,i}] + \operatorname{Var}[s'_{2,i}] + (\mathbb{E}[s'_{1,i}] - \mathbb{E}[s'_{2,i}])^2

$$
2.  **Analyze the Difference of Means:**
    The expected final status of walker ({prf:ref}`def-walker`) **i** starting from state $\mathcal{S}_k$ is its marginal probability of survival, $\mathbb{E}[s'_{k,i}] = P(s_{\text{out},i}=1 | \mathcal{S}_k)$. The squared difference of the means is:

$$

    (\mathbb{E}[s'_{1,i}] - \mathbb{E}[s'_{2,i}])^2 = (P(s_{\text{out},i}=1 | \mathcal{S}_1) - P(s_{\text{out},i}=1 | \mathcal{S}_2))^2

$$
The probability of survival is one minus the probability of death, so $|P(s_{\text{out},i}=1 | \mathcal{S}_1) - P(s_{\text{out},i}=1 | \mathcal{S}_2)| = |(1 - P(s_{\text{out},i}=0 | \mathcal{S}_1)) - (1 - P(s_{\text{out},i}=0 | \mathcal{S}_2))| = |P(s_{\text{out},i}=0 | \mathcal{S}_2) - P(s_{\text{out},i}=0 | \mathcal{S}_1)|$. We can now apply the **Axiom of Boundary Regularity ({prf:ref}`axiom-boundary-regularity`) ([](#axiom-boundary-regularity))** to this difference:

$$

    |P(s_{\text{out},i}=0 | \mathcal{S}_1) - P(s_{\text{out},i}=0 | \mathcal{S}_2)| \le L_{\text{death}} \cdot d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1, \mathcal{S}_2)^{\alpha_B}

$$
Squaring this inequality gives a bound for the squared difference of the means:

$$

    (\mathbb{E}[s'_{1,i}] - \mathbb{E}[s'_{2,i}])^2 \le L_{\text{death}}^2 \cdot \left( d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1, \mathcal{S}_2) \right)^{2\alpha_B}

$$
3.  **Sum Over All Walkers:**
    We sum the full expression from Step 1 over all walkers **i** and substitute the bound from Step 2:

$$

    \mathbb{E}[n_c(\mathcal{S}'_1, \mathcal{S}'_2)] = \sum_{i=1}^N \left( \operatorname{Var}[s'_{1,i}] + \operatorname{Var}[s'_{2,i}] \right) + \sum_{i=1}^N (\mathbb{E}[s'_{1,i}] - \mathbb{E}[s'_{2,i}])^2

$$
$$
    \le \sum_{i=1}^N \left( \operatorname{Var}[s'_{1,i}] + \operatorname{Var}[s'_{2,i}] \right) + \sum_{i=1}^N L_{\text{death}}^2 \left( d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1, \mathcal{S}_2) \right)^{2\alpha_B}

    $$
4.  **Finalize the Bound:**
    *   **Variance Term:** The variance of a Bernoulli random variable is $\operatorname{Var}(X) = p(1-p)$, which is always bounded above by $1/4$. Therefore, the sum of all variance terms is bounded by a constant: $\sum_{i=1}^N (\operatorname{Var}[s'_{1,i}] + \operatorname{Var}[s'_{2,i}]) \le \sum_{i=1}^N (\frac{1}{4} + \frac{1}{4}) = \frac{N}{2}$.
    *   **Hölder Term:** The bound on the squared difference of means is identical for all $N$ walkers. Summing this bound $N$ times gives $N L_{\text{death}}^2 \left( d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1, \mathcal{S}_2) \right)^{2\alpha_B}$.
    Combining these two bounds yields the final inequality as stated in the theorem.
**Q.E.D.**
:::
## 16. The Cloning Transition Measure
The final step in the measurement pipeline is to convert the N-dimensional fitness potential vector into a probabilistic cloning decision for each walker. This process is governed by a score function and the resulting $Cloning Bernoulli Measure$.
### 15.1 Definition: Cloning Transition Measure
#### 15.1.1 The Cloning Score Function
The core arithmetic of the cloning decision is encapsulated in a deterministic function that compares a walker's potential to that of its companion.
:::{prf:definition} Cloning Score Function
:label: def-cloning-score-function
The **Cloning Score Function**, $S: \mathbb{R}_{\ge 0} \times \mathbb{R}_{\ge 0} \to \mathbb{R}$, takes the fitness potential of a companion walker ({prf:ref}`def-walker`) ($v_c$) and a primary walker ($v_i$) and computes a raw score.

$$

S(v_c, v_i) := \frac{v_c - v_i}{v_i + \varepsilon}

$$
where $\varepsilon > 0$ is the cloning denominator regularizer.
::::
#### 15.1.2 Stochastic Threshold Cloning
This procedure defines the cloning action for each walker ({prf:ref}`def-walker`). It replaces a probabilistic model with a deterministic comparison between the walker's score and a randomly sampled threshold.
:::{prf:definition} Stochastic Threshold Cloning
:label: def-stochastic-threshold-cloning
This definition specifies the cloning mechanism used in {prf:ref}`def-cloning-measure` and {prf:ref}`def-fragile-gas-algorithm`.

For each walker ({prf:ref}`def-walker`) $i \in \{1, \dots, N\}$, the cloning action $a_i \in \{\text{Clone}, \text{Persist}\}$ is determined by the following procedure, which depends on the full fitness potential vector of the swarm ({prf:ref}`def-swarm-and-state-space`) and an independent random choice of a companion.
**Inputs:**
*   The full N-dimensional fitness potential vector, $\mathbf{V}_{\text{fit}}$.
*   The walker ({prf:ref}`def-walker`)'s index, $i$.
*   The **Companion Selection Measure ({prf:ref}`def-companion-selection-measure`)** for that walker ({prf:ref}`def-walker`), $\mathbb{C}_i$.
*   The **Clone Threshold Scale** parameter, $p_{\max}$.
**Operation:**
The action is determined as follows:
1.  **Sample Cloning Companion:** An independent cloning companion index, $c_{\text{clone}}(i)$, is sampled from the companion measure: $c_{\text{clone}}(i) \sim \mathbb{C}_i(\cdot)$.
2.  **Compute Score:** The walker ({prf:ref}`def-walker`)'s potential, $v_i = V_{\text{fit},i}$, and its companion's potential, $v_c = V_{\text{fit},c_{\text{clone}}(i)}$, are used to compute the cloning score using the [](#def-cloning-score-function):

$$

    S_i := S(v_c, v_i)

$$
3.  **Sample Cloning Threshold:** A random threshold, $T_{\text{clone}}$, is drawn from a uniform distribution over the interval defined by the Clone Threshold Scale:

$$

    T_{\text{clone}} \sim \text{Uniform}(0, p_{\max})

$$
4.  **Determine Action:** The action is determined by comparing the score to the threshold. A walker ({prf:ref}`def-walker`) is cloned only if its score exceeds the randomly drawn threshold.

$$

    a_i :=
    \begin{cases}
    \text{Clone} & \text{if } S_i > T_{\text{clone}} \\
    \text{Persist} & \text{if } S_i \le T_{\text{clone}}
    \end{cases}

$$
This unified definition handles both alive and dead walker ({prf:ref}`def-walker`)s. For a **dead walker** $i \in \mathcal{D}_t$, its potential $V_{\text{fit},i} = 0$. Its cloning companion $c_{\text{clone}}(i)$ is drawn from the alive set ({prf:ref}`def-alive-dead-sets`) $\mathcal{A}_t$, making its potential $V_{\text{fit},c_{\text{clone}}(i)} > 0$. The score simplifies to $S_i = V_{\text{fit},c_{\text{clone}}(i)} / \varepsilon$. Since the **Global Constraint** $\varepsilon \cdot p_{\max} < \eta^{(\alpha+\beta)}$ is satisfied, the minimum possible score for a dead walker is guaranteed to be greater than the maximum possible threshold: $S_{i, \min} > (\eta^{\alpha+\beta})/\varepsilon > p_{\max}$. Because $T_{\text{clone}}$ is always sampled from $[0, p_{\max}]$, the condition $S_i > T_{\text{clone}}$ is always met. This results in a "Clone" action with probability 1, ensuring the revival mechanism remains an emergent and guaranteed property of the framework.
:::
:::{admonition} Randomness discipline for cloning
:class: note
- For each walker ({prf:ref}`def-walker`) $i$, independently sample the cloning companion index $c_{\text{clone}}(i)\sim \mathbb C_i(\mathcal S)$ (with replacement) using an independent uniform $U_i^{\mathrm{comp}}$.
- Independently for each walker ({prf:ref}`def-walker`), draw the threshold $T_{\text{clone},i}\sim \mathrm{Unif}(0,p_{\max})$ using an independent uniform $U_i^{\mathrm{clone}}$.
These choices ensure per‑walker ({prf:ref}`def-walker`) independence in the cloning stage, consistent with Assumption A and the sampling policy in §7.1.1. Systematic resampling schemes that reuse a shared uniform are excluded from McDiarmid‑based analysis.
:::
### 15.2. Continuity of the Cloning Transition Measure
The Cloning Transition is a composite stochastic operator that determines the intermediate swarm state based on the calculated fitness potentials. Its continuity analysis is critical, as it governs how sensitively the cloning and revival mechanisms react to small changes in the swarm state. A discontinuous transition would imply chaotic behavior where small measurement fluctuations could lead to drastically different swarm configurations.
This section proves that the operator is probabilistically continuous. The analysis is centered on the key insight that the continuity of the overall transition depends on the continuity of the **total probability** of cloning. We first define this total probability, which averages over all stochasticity in the measurement pipeline, and then prove that it is a continuous function of the input swarm state. This result is the foundation for proving the continuity of the full operator.
:::{prf:remark} Cloning Scope and Companion Convention
:label: rem-cloning-scope-companion-convention

All bounds in §15.2.4–§15.2.8 are stated for the regime $k_1=|\mathcal A(\mathcal S_1)|\ge 2$ (at least two alive walkers), with the "no self‑companion" convention (an alive walker ({prf:ref}`def-walker`) samples companions from $\mathcal A\setminus\{i\}$). The edge case $k=1$ is handled separately in §15 (single‑survivor revival), after which analysis resumes with $k\ge 2$. Where intermediate formulas feature denominators $k_1-1$, they are interpreted under this precondition; if a generic statement is needed, replace $k_1-1$ by $\max(1, k_1-1)$ and invoke the $k=1$ section.
:::
#### 15.2.1. The Total Expected Cloning Action
The ultimate probability of a "Clone" action for a walker depends on the outcome of the stochastic distance measurement and the random companion choice. To analyze the operator's continuity as a function of the input swarm ({prf:ref}`def-swarm-and-state-space`) state, we must average over all sources of randomness.
:::{prf:definition} Total Expected Cloning Action
:label: def-total-expected-cloning-action
The **Total Expected Cloning Action** for a walker ({prf:ref}`def-walker`) $i$, denoted $\overline{P}_{\text{clone}}(\mathcal{S})_i$, is the probability that walker $i$ will be assigned the "Clone" action, given the swarm ({prf:ref}`def-swarm-and-state-space`) state $\mathcal{S}$. It is the expectation of the **Conditional Expected Cloning Action** (Def. 15.2.3) taken over the probability distribution of the raw distance vector $\mathbf{d} \sim \mathbf{d}(\mathcal{S})$.
Let $\mathbf{r}(\mathcal{S})$ be the deterministic raw reward vector for state $\mathcal{S}$, and let $\mathbf{V}(\mathbf{r}, \mathbf{d})$ be the fitness potential vector generated from a specific realization of the raw measurement vectors. The total expected action is:

$$

\overline{P}_{\text{clone}}(\mathcal{S})_i := \mathbb{E}_{\mathbf{d} \sim \mathbf{d}(\mathcal{S})} \left[ P_{\text{clone}}(\mathcal{S}, \mathbf{V}(\mathbf{r}(\mathcal{S}), \mathbf{d}))_i \right]

$$
This quantity is a deterministic function of the input swarm state ({prf:ref}`def-swarm-and-state-space`) $\mathcal{S}$ and is the central object for the continuity analysis of the cloning stage.
:::
#### 15.2.2. The Conditional Cloning Probability Function and its Continuity
The probability of a "Clone" action for a walker ({prf:ref}`def-walker`) **i** with a specific companion **c** can be expressed as a deterministic function of their respective fitness potentials. This represents the probability *conditional* on a specific realization of the potential vector.
:::{prf:definition} The Conditional Cloning Probability Function
:label: def-cloning-probability-function
The **Conditional Cloning Probability Function**, denoted $\pi: \mathbb{R}_{\ge 0} \times \mathbb{R}_{\ge 0} \to [0, 1]$, maps the fitness potential of a companion ($v_c$) and a primary walker ({prf:ref}`def-walker`) ($v_i$) to the probability that the "Clone" action is selected.
Given that the score is $S(v_c, v_i)$ and the threshold is $T_{\text{clone}} \sim \text{Uniform}(0, p_{\max})$, the probability is:

$$

\pi(v_c, v_i) := P(S(v_c, v_i) > T_{\text{clone}}) = \min\left(1, \max\left(0, \frac{S(v_c, v_i)}{p_{\max}}\right)\right)

$$
This function effectively clips the normalized score to the valid probability range $[0, 1]$.
:::
:::{prf:lemma} Lipschitz Continuity of the Conditional Cloning Probability ({prf:ref}`def-cloning-probability-function`)it)
:label: lem-cloning-probability-lipschitz

$$

|\pi(v_{c1}, v_{i1}) - \pi(v_{c2}, v_{i2})| \Lipschitz ({prf:ref}`axiom-reward-regularity`)} - v_{c2}| + L_{\pi,i}|v_{i1} - v_{i2}|

$$
where the **Cloning Probability Lipschitz Constants** can be chosen uniformly over both alive and dead walkers by the worst‑case (dead‑walker ({prf:ref}`def-walker`)) bounds:
*   **Companion Potential Lipschitz Constant ($L_{\pi,c}$):**

$$

L_{\pi,c} := \frac{1}{p_{\max} \cdot \varepsilon_{\text{clone}}}

$$
*   **Walker ({prf:ref}`def-walker`) Potential Lipschitz Constant ($L_{\pi,i}$):**

$$

L_{\pi,i} := \frac{V_{\text{pot,max}} + \varepsilon_{\text{clone}}}{p_{\max} \cdot \varepsilon_{\text{clone}}^{\,2}}

$$
:::
:::{prf:proof}
:label: proof-lem-cloning-probability-lipschitz

**Proof.**
The proof proceeds by finding the Lipschitz constant of the composition of the **clip** function and the normalized score function, $S(v_c, v_i)/p_{\max}$. The **clip** function (min(1, max(0, x))) has a Lipschitz constant of 1. Therefore, the Lipschitz constant of $\pi$ is bounded by the Lipschitz constant of the normalized score. We find this by bounding the partial derivatives of the score function $S(v_c, v_i)$.
1.  **Partial Derivative with respect to $v_c$:** $\partial S/\partial v_c = 1/(v_i + \varepsilon_{\text{clone}})$. For alive walker ({prf:ref}`def-walker`)s, [](#lem-potential-boundedness) gives $v_i\ge V_{\text{pot,min}}$, hence the bound $1/(V_{\text{pot,min}} + \varepsilon_{\text{clone}})$. For a dead walker ($v_i=0$), the bound is $1/\varepsilon_{\text{clone}}$. The worst case is the dead‑walker value $1/\varepsilon_{\text{clone}}$.
2.  **Partial Derivative with respect to $v_i$:** $\partial S/\partial v_i = (-\varepsilon_{\text{clone}} - v_c)/(v_i + \varepsilon_{\text{clone}})^2$. In magnitude, this is $\le (V_{\text{pot,max}} + \varepsilon_{\text{clone}})/(v_i + \varepsilon_{\text{clone}})^2$. For alive walker ({prf:ref}`def-walker`)s, use $v_i\ge V_{\text{pot,min}}$; for a dead walker, $v_i=0$ yields the worst‑case bound $(V_{\text{pot,max}} + \varepsilon_{\text{clone}})/\varepsilon_{\text{clone}}^2$.
3.  **Combine:** Divide the worst‑case partial‑derivative bounds by $p_{\max}$ to obtain the stated uniform Lipschitz constants $L_{\pi,c}$ and $L_{\pi,i}$ that cover both alive and dead cases.
**Q.E.D.**
:::
#### 15.2.3. Continuity of the Conditional Expected Cloning Action
The overall cloning decision for a walker ({prf:ref}`def-walker`) is stochastic due to the random choice of a companion. We can analyze the continuity of this process *conditional on a fixed fitness potential vector* by examining its expectation over the companion selection measure.
:::{prf:definition} Conditional Expected Cloning Action
:label: def-expected-cloning-action
The **Conditional Expected Cloning Action** for a walker ({prf:ref}`def-walker`) $i$, denoted $P_{\text{clone}}(\mathcal{S}, \mathbf{V})_i$, is the probability that walker $i$ will be assigned the "Clone" action, given the swarm ({prf:ref}`def-swarm-and-state-space`) state $\mathcal{S}$ and a specific, fixed fitness potential vector $\mathbf{V}$. It is the expectation of the **Conditional Cloning Probability Function** under the **Companion Selection ({prf:ref}`def-companion-selection-measure`) Measure**:

$$

P_{\text{clone}}(\mathcal{S}, \mathbf{V})_i := \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S})} [\pi(V_c, V_i)]

$$
:::
:::{prf:theorem} Continuity of the Conditional Expected Cloning ({prf:ref}`def-expected-cloning-action`)thm-expected-cloning-action-continuity
The **Conditional Expected Cloning Action** is continuous with respect to changes in both the swarm ({prf:ref}`def-swarm-and-state-space`) structure and the fitness potential vector. For any two states $(\mathcal{S}_1, \mathbf{V}_1)$ and $(\mathcal{S}_2, \mathbf{V}_2)$, with $k_1 = |\mathcal{A}(\mathcal{S}_1)| > 0$, the change in the conditional expected action for any walker ({prf:ref}`def-walker`) $i$ is bounded:

$$

|P_{\text{clone}}(\mathcal{S}_1, \mathbf{V}_1)_i - P_{\text{clone}}(\mathcal{S}_2, \mathbf{V}_2)_i| \le C_{\text{struct}}^{(\pi)}(k_1) \cdot n_c(\mathcal{S}_1, \mathcal{S}_2) + C_{\text{val}}^{(\pi)} \cdot \left( \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)}[|V_{1,c} - V_{2,c}|] + |V_{1,i} - V_{2,i}| \right)

$$
where the coefficients are:
*   $C_{\text{struct}}^{(\pi)}(k_1) := \frac{2}{\max(1, k_1-1)}$ (from structural change)
*   $C_{\text{val}}^{(\pi)} := \max(L_{\pi,c}, L_{\pi,i})$ (from potential vector change)
:::
:::{prf:proof}
:label: proof-thm-expected-cloning-action-continuity

**Proof.**
The proof decomposes the total error into a structural component and a value component using the triangle inequality:
$|E_1[π_1] - E_2[π_2]| \leq |E_1[π_1] - E_1[π_2]| + |E_1[π_2] - E_2[π_2]|$.
1.  **Bound the Value Error Component:** The first term is the error from the change in potentials for a fixed companion measure. Using Jensen's inequality and the Lipschitz continuity of $\pi$ ([](#lem-cloning-probability-lipschitz)), this is bounded by $C_{val}^{(π)}$ times the sum of the expected companion potential change and the walker ({prf:ref}`def-walker`)'s own potential change.
2.  **Bound the Structural Error Component:** The second term is the error from the change in the companion measure for a fixed potential vector. We apply the **Total Error Bound in Terms of Status Changes ([](#thm-total-error-status-bound))**. The function being evaluated is bounded by $M_f=1$. The size of the initial companion set is at least $max(1, k_1-1)$. This gives the bound $C_{\text{struct}}^{(π)}(k_1) \cdot n_c$.
Summing the two bounds gives the final result.
**Q.E.D.**
:::
#### 15.2.4. Continuity of the Total Expected Cloning Action
This theorem provides the main result for the continuity of the total probability of cloning. It proves that this probability, averaged over all sources of randomness, is a continuous function of the input swarm ({prf:ref}`def-swarm-and-state-space`) state.
:::{prf:theorem}Expected Cloning ({prf:ref}`def-expected-cloning-action`)d Cloning Action
:label: thm-total-expected-cloning-action-continuity
Let $\mathcal{S}_1$ and $\mathcal{S}_2$ be two swarm ({prf:ref}`def-swarm-and-state-space`) states, with $k_1Expected Cloning ({prf:ref}`def-expected-cloning-action`)> 0$. The **Total Expected Cloning Action** is continuous with respect to the N-Particle Displacement ({prf:ref}`def-n-particle-displacement-metric`) Metric. For any walker ({prf:ref}`def-walker`) $icloning probability ({prf:ref}`def-cloning-probability-function`)bability is bounded:

$$

|\overline{P}_{\text{clone}}(\mathcal{S}_1)_i - \overline{P}_{\text{clone}}(\mathcal{S}_2)_i| \le E_{\text{struct}}^{(\overline{P})}(\mathcal{S}_1, \mathcal{S}_2) + E_{\text{val}}^{(\overline{P})}(\mathcal{S}_1, \mathcal{S}_2)

$$
where the two error components are bounded in the subsequent lemmas.
:::
:::{prf:proof}
:label: proof-thm-total-expected-cloning-action-continuity

**Proof.**
Let $\overline{P}_{k,i} = \overline{P}_{\text{clone}}(\mathcal{S}_k)_i$ ({prf:ref}`def-swarm-and-state-space`). We introduce an intermediate term and apply the triangle inequality to decompose the total error. Let $P_{k,i}(\mathbf{V}) := P_{\text{clone}}(\mathcal{S}_k, \mathbf{V})_i$ be the conditional expected action. The total error is $|\mathbb{E}_{\mathbf{V}_1}[P_{1,i}(\mathbf{V}_1)] - \mathbb{E}_{\mathbf{V}_2}[P_{2,i}(\mathbf{V}_2)]|$.
We add and subtract the term $\mathbb{E}_{\mathbf{V}_1}[P_{2,i}(\mathbf{V}_1)]$:

$$

\le |\mathbb{E}_{\mathbf{V}_1}[P_{1,i}(\mathbf{V}_1) - P_{2,i}(\mathbf{V}_1)]| + |\mathbb{E}_{\mathbf{V}_1}[P_{2,i}(\mathbf{V}_1)] - \mathbb{E}_{\mathbf{V}_2}[P_{2,i}(\mathbf{V}_2)]|

$$
The first term is the **Structural Error Component**, $E_{\text{struct}}^{(\overline{P})}$. The second term is the **Value Error Component**, $E_{\text{val}}^{(\overline{P})}$.
**Q.E.D.**
:::
#### 15.2.5. Bounding the Structural Component of Cloning Probability Error
:label: lem-total-clone-prob-structural-error
Let $E_{\text{struct}}^{(\overline{P})}$ be the structural error component from **[](#thm-total-expected-cloning-action-continuity)**. It is deterministically bounded by the number of status changes between the swarms.

$$

E_{\text{struct}}^{(\overline{P})}(\mathcal{S}_1, \mathcal{S}_2) \le C_{\text{struct}}^{(\pi)}(k_1) \cdot n_c(\mathcal{S}_1, \mathcal{S}_2)

$$
:::{prf:proof}
:label: proof-lem-total-clone-prob-structural-error

**Proof.**
The structural error is $|E_V_1[P_1,i(V_1) - P_2,i(V_1)]|$. By Jensen's inequality, this is $\leq E_V_1[|P_1,i(V_1) - P_2,i(V_1)|]$. From [](#thm-expected-cloning-action-continuity), the term inside the expectation is bounded by $C_{\text{struct}}^{(π)}(k_1) \cdot n_c$. Since this bound is a deterministic constant, its expectation is the bound itself.
**Q.E.D.**
:::
#### 15.2.6. Theorem: The Fitness Potential Operator is Mean-Square Continuous

:::{prf:theorem} The Fitness Potential Operator is Mean-Square Continuous
:label: thm-potential-operator-is-mean-square-continuous

This theorem establishes mean-square continuity of {prf:ref}`def-alive-set-potential-operator`, building on the standardization continuity results.

The **Fitness Potential Operator** is **mean-square continuous**. There exists a deterministic function $F_{\text{pot}}(\mathcal{S}_1, \mathcal{S}_2)$, the **Expected Squared Potential Error Bound**, such that:

$$

\mathbb{E}[\|\mathbf{V}_1 - \mathbf{V}_2\|_2^2] \le F_{\text{pot}}(\mathcal{S}_1, \mathcal{S}_2)

$$
:::

:::{prf:proof}
:label: proof-thm-potential-operator-is-mean-square-continuous

**Proof.**
This property is established by the detailed analysis in Section 12.2, culminating in **[](#thm-fitness-potential-mean-square-continuity)**. The explicit form of $F_{\text{pot}}$ is constructed from the composition of the mean-square continuity bounds of all preceding operators.
**Q.E.D.**
:::
#### 15.2.7. Bounding the Value Component of Cloning Probability Error
:label: lem-total-clone-prob-value-error
Let $E_{\text{val}}^{(\overline{P})}$ be the value error component from **[](#thm-total-expected-cloning-action-continuity)**. It is deterministically bounded as follows:

$$

E_{\text{val}}^{(\overline{P})}(\mathcal{S}_1, \mathcal{S}_2) \le C_{\text{val}}^{(\pi)} \sqrt{2N \cdot F_{\text{pot}}(\mathcal{S}_1, \mathcal{S}_2)}

$$
where $F_{\text{pot}}$ is the **Expected Squared Potential Error Bound**.
:::{prf:proof}
:label: proof-lem-total-clone-prob-value-error

**Proof.**
The error is $|E_V_1[f(V_1)] - E_V_2[f(V_2)]|$ where $f(V) = P_{\text{clone}}(S_2, V)_i$.
1.  **Lipschitz Continuity of **f**:** From [](#thm-expected-cloning-action-continuity), $|f(V_1) - f(V_2)| \leq C_{val}^{(π)} (E_c[|V_1,c-V_2,c|] + |V_1,i-V_2,i|)$. Using properties of L1/L2 norms, this is $\leq C_{val}^{(π)}√2\|V_1-V_2\|_1 \leq C_{val}^{(π)}√2√N\|V_1-V_2\|_2$. So, **f** is Lipschitz with constant $L_f = C_{val}^{(π)}√(2N)$.
2.  **Bound the Difference in Expectations:** The difference is $|E[f(V_1)-f(V_2)]| \leq E[|f(V_1)-f(V_2)|] \leq E[L_f \|V_1-V_2\|_2] = L_f E[\|V_1-V_2\|_2]$.
3.  **Apply Jensen's Inequality and Mean-Square Bound:** $E[X] \leq √E[X^2]$. So, the error is $\leq L_f √E[\|V_1-V_2\|_2^2]$. Substituting $L_f$ and the $F_{\text{pot}}$ bound from [](#thm-potential-operator-is-mean-square-continuous) yields the final result.
**Q.E.D.**
:::
#### 15.2.8. Theorem: Mean-Square Continuity of the Cloning Transition Operator

:::{prf:theorem} Mean-Square Continuity of the Cloning Transition Operator
:label: thm-cloning-transition-operator-continuity-recorrected

Let $\mathcal{S}_1$ and $\mathcal{S}_2$ be two input states from the alive state space, with $k_1 = |\mathcal{A}(\mathcal{S}_1)| Lipschitz ({prf:ref}`axiom-reward-regularity`)cal{S}'_1 \sim \Psi_{\text{clone}}(\mathcal{S}_1, \cdot)$ and $\mathcal{S}'_2 \sim \Psi_{\text{clone}}(\mathcal{S}_2, \cdoLipschitz ({prf:ref}`axiom-reward-regularity`)diate swarm ({prf:ref}`def-swarm-and-state-space`)s sampled independently from the cloning transition ({prf:ref}`def-cloning-measure`) measure.
The Cloning Transition Operator is mean-square continuous. The expected squared **N-Particle Displacement ({prf:ref}`def-n-particle-displacement-metric`) Metric** between the two output swarms is bounded by a sum of a Lipschitz term and a Hölder term of the input squared displacement:

$$

\mathbb{E}[d_{\text{Disp},\mathcal{Y}}(\mathcal{S}'_1, \mathcal{S}'_2)^2] \le C_{\text{clone},L}(\mathcal{S}_1, \mathcal{S}_2) \cdot d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1, \mathcal{S}_2)^2 + C_{\text{clone},H}(\mathcal{S}_1, \mathcal{S}_2) \cdot d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1, \mathcal{S}_2) + K_{\text{clone}}(\mathcal{S}_1, \mathcal{S}_2)

$$

where $C_{\text{clone},L}$, $C_{\text{clone},H}$, and $K_{\text{clone}}$ are the **Cloning Operator Continuity Coefficients**, which are deterministic, state-dependent functions defined in the subsequent sections.
:::

##### 15.2.8.1. Definition: Cloning Operator Continuity Coefficients
:label: def-cloning-operator-continuity-coeffs-recorrected
The state-dependent functions in the continuity bound for the Cloning Transition Operator are defined as follows:
1.  **The Cloning Lipschitz Amplification Factor ($C_{\text{clone},L}(\mathcal{S}_1, \mathcal{S}_2)$):** The coefficient of the term that is linear in the input squared displacement, $d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1, \mathcal{S}_2)^2$. This term primarily captures the propagation of the positional component of the input displacement.
2.  **The Cloning Hölder Amplification Factor ($C_{\text{clone},H}(\mathcal{S}_1, \mathcal{S}_2)$):** The coefficient of the Hölder term, $d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1, \mathcal{S}_2)$. This term arises from the complex, non-linear error propagation originating from the **Distance-to-Companion Measurement**, specifically from the component of its error bound that is quadratic in the number of status changes.
3.  **The Cloning Stochastic Offset ($K_{\text{clone}}(\mathcal{S}_1, \mathcal{S}_2)$):** A state-dependent term that is independent of the input displacement. It represents the baseline displacement introduced by the operator's intrinsic stochasticity, which exists even if the two input swarms are identical.
##### 15.2.8.2. Proof of Mean-Square Continuity for the Cloning Operator
:label: proof-cloning-transition-operator-continuity-recorrected
:::{prf:proof}
:label: proof-proof-cloning-transition-operator-continuity-recorrected
**Proof.**
The proof establishes the bound by relating the expected output displacement to the expected intermediate positional displacement, and then bounding the latter by composing the continuity bounds of the underlying measurement and potential-calculation pipeline.
Let $V_{\text{in}} := d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1, \mathcal{S}_2)^2$ be the initial squared displacement. Let $\mathcal{S}'_1$ and $\mathcal{S}'_2$ be the intermediate swarms after the cloning transition.
1.  **Bound the Expected Intermediate Positional Displacement.**
    Since all intermediate walker ({prf:ref}`def-walker`)s are assigned an "alive" status, the status component of their displacement is zero, and the expected output displacement is given by $\mathbb{E}[d_{\text{Disp},\mathcal{Y}}(\mathcal{S}'_1, \mathcal{S}'_2)^2] = \frac{1}{N} \mathbb{E}[\Delta_{\text{pos}}^2(\mathcal{S}'_1, \mathcal{S}'_2)]$.
    For any single walker ({prf:ref}`def-walker`) **i**, using the triangle inequality and the property $(a+b+c)^2 \le 3(a^2+b^2+c^2)$, we have:

$$

    \mathbb{E}[d_{\text{alg}}(x'_{1,i}, x'_{2,i})^2] \le 3d_{\text{alg}}(x_{1,i}, x_{2,i})^2 + 3\mathbb{E}[d_{\text{alg}}(x'_{1,i}, x_{1,i})^2] + 3\mathbb{E}[d_{\text{alg}}(x'_{2,i}, x_{2,i})^2]

$$
The expected squared displacement of a walker ({prf:ref}`def-walker`) from its own initial position, $\mathbb{E}[d_{\text{alg}}(x'_{k,i}, x_{k,i})^2]$, is bounded by the total probability of it being cloned, $\overline{P}_{\text{clone}}(\mathcal{S}_k)_i$, multiplied by the maximum possible squared displacement, which is bounded by $D_{\mathcal{Y}}^2$. Summing over all **N** walkers gives a bound on the total expected intermediate positional displacement:

$$

    \mathbb{E}[\Delta_{\text{pos}}^2(\mathcal{S}'_1, \mathcal{S}'_2)] \le 3\Delta_{\text{pos}}^2(\mathcal{S}_1, \mathcal{S}_2) + 3D_{\mathcal{Y}}^2 \sum_{i=1}^N \left( \overline{P}_{\text{clone}}(\mathcal{S}_1)_i + \overline{P}_{\text{clone}}(\mathcal{S}_2)_i \right)

$$
2.  **Bound the Sum of Cloning Probabilities.**
    The core of the proof is to bound the sum of the total cloning probabilities in terms of the input displacement $V_{\text{in}}$. As rigorously proven in the subsequent **Sub-Lemma 15.2.8.3**, this sum can be bounded by a function that contains both a linear and a square-root term of the input displacement:

$$

    \sum_{i=1}^N \left( \overline{P}_{\text{clone}}(\mathcal{S}_1)_i + \overline{P}_{\text{clone}}(\mathcal{S}_2)_i \right) \le C_P(\mathcal{S}_1, \mathcal{S}_2) \cdot V_{\text{in}} + H_P(\mathcal{S}_1, \mathcal{S}_2) \cdot \sqrt{V_{\text{in}}} + K_P(\mathcal{S}_1, \mathcal{S}_2)

$$
where $C_P$, $H_P$, and $K_P$ are state-dependent coefficients derived in the sub-lemma.
3.  **Assemble the Final Bound.**
    We substitute the bound from Step 2 into the inequality from Step 1. We also use the fact that the initial positional displacement is a component of the total displacement, so $\Delta_{\text{pos}}^2(\mathcal{S}_1, \mathcal{S}_2) \le N \cdot V_{\text{in}}$.

$$

    \mathbb{E}[\Delta_{\text{pos}}^2(\mathcal{S}'_1, \mathcal{S}'_2)] \le 3(N \cdot V_{\text{in}}) + 3D_{\mathcal{Y}}^2 \left( C_P V_{\text{in}} + H_P \sqrt{V_{\text{in}}} + K_P \right)

$$
Finally, we divide by **N** to get the bound on the expected output squared metric, $\mathbb{E}[d_{\text{Disp},\mathcal{Y}}(\mathcal{S}'_1, \mathcal{S}'_2)^2]$:

$$

    \mathbb{E}[d_{\text{out}}^2] \le \left(3 + \frac{3D_{\mathcal{Y}}^2 C_P}{N}\right)V_{\text{in}} + \left(\frac{3D_{\mathcal{Y}}^2 H_P}{N}\right)\sqrt{V_{\text{in}}} + \frac{3D_{\mathcal{Y}}^2 K_P}{N}

$$
This expression is of the required form $C_L V + C_H sqrt(V) + K$. By inspection, we can identify the coefficients $C_{\text{clone},L}$, $C_{\text{clone},H}$, and $K_{\text{clone}}$ from this final expression. This completes the proof.
**Q.E.D.**
:::
##### 15.2.8.3. Sub-Lemma: Bounding the Sum of Total Cloning Probabilities
:::{prf:lemma} Bounding the Sum of Total Cloning Probabilities
:label: lem-sub-bound-sum-total-cloning-probs

Let $\mathcal{S}_1$ and $\mathcal{S}_2$ be two swarm ({prf:ref}`def-swarm-and-state-space`) states. Let $V_{\text{in}} := d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1, \matExpected Cloning ({prf:ref}`def-expected-cloning-action`)ared displacement.

The sum of the **Total Expected Cloning Probabilities**, $\sum_{i=1}^N (\overline{P}_{\text{clone}}(\mathcal{S}_1)_i + \overline{P}_{\text{clone}}(\mathcal{S}_2)_i)$, is bounded by a sum of a linear term, a Hölder term, and a constant offset of the initial displacement:

$$

\sum_{i=1}^N \left( \overline{P}_{\text{clone}}(\mathcal{S}_1)_i + \overline{P}_{\text{clone}}(\mathcal{S}_2)_i \right) \le C_P(\mathcal{S}_1, \mathcal{S}_2) \cdot V_{\text{in}} + H_P(\mathcal{S}_1, \mathcal{S}_2) \cdot \sqrt{V_{\text{in}}} + K_P(\mathcal{S}_1, \mathcal{S}_2)

$$

where $C_P$, $H_P$, and $K_P$ are finite, state-dependent, non-negative coefficients.
:::
:::{prf:proof}
:label: proof-lem-sub-bound-sum-total-cloning-probs

**Proof.**
The proof proceeds by bounding the change in the cloning probability and then relating that change to the input displacement.
1.  **Decompose the Sum.**
    Using the triangle inequality, we can bound the sum:

$$

    \sum_{i=1}^N \left( \overline{P}_{\text{clone}}(\mathcal{S}_1)_i + \overline{P}_{\text{clone}}(\mathcal{S}_2)_i \right) \le \sum_{i=1}^N |\overline{P}_{\text{clone}}(\mathcal{S}_1)_i - \overline{P}_{\text{clone}}(\mathcal{S}_2)_i| + 2\sum_{i=1}^N \overline{P}_{\text{clone}}(\mathcal{S}_2)_i

$$
The second term, $2\sum \overline{P}_{\text{clone}}(\mathcal{S}_2)_i$, is bounded by the state-dependent constant $2N$. This will be absorbed into the final offset, $K_P$. The core of the proof is to bound the first term, which is the L1-norm of the difference between the total cloning probability vectors, $\|\Delta \overline{\mathbf{P}}\|_1$.
2.  **Bound the L1-Norm of the Probability Difference.**
    From the continuity of the total expected cloning action ([](#thm-total-expected-cloning-action-continuity)), we have a bound for each component, which we sum over all **N** walker ({prf:ref}`def-walker`)s:

$$

    \|\Delta \overline{\mathbf{P}}\|_1 = \sum_{i=1}^N |\overline{P}_{\text{clone}}(\mathcal{S}_1)_i - \overline{P}_{\text{clone}}(\mathcal{S}_2)_i| \le \sum_{i=1}^N (E_{\text{struct}}^{(\overline{P})} + E_{\text{val}}^{(\overline{P})})

$$
*   The structural error term from [](#lem-total-clone-prob-structural-error) is bounded by $N \cdot C_{\text{struct}}^{(\pi)}(k_1) \cdot n_c$.
    *   The value error term from [](#lem-total-clone-prob-value-error) is bounded by $N \cdot C_{\text{val}}^{(\pi)} \sqrt{2N \cdot F_{\text{pot}}}$.
3.  **Substitute the Bound for the Fitness Potential Error ($F_{\text{pot}}$).**
    The crucial step is to substitute the bound for the **Expected Squared Potential Error Bound** ($F_{\text{pot}}$) from [](#thm-fitness-potential-mean-square-continuity). $F_{\text{pot}}$ is itself a function of the input displacement components: $F_{\text{pot}}(S_1, S_2) = F_unstable + F_{\text{stable}}$, where $F_{\text{stable}}$ is bounded by the mean-square errors of the standardization pipelines for reward and distance. The distance standardization error ($E_[\|\Deltaz_d\|^2]$) from [](#thm-distance-operator-mean-square-continuity) contains a term proportional to $n_c^2$.
    Therefore, the full bound for $F_{\text{pot}}$ takes the form:

$$

    F_{\text{pot}} \le A_1 \cdot \Delta_{\text{pos}}^2 + A_2 \cdot n_c + A_3 \cdot n_c^2 + A_4

$$
where $A_k$ are state-dependent coefficients.
4.  **Relate Displacement Components to $V_{\text{in}}$.**
    We relate the input displacement components to the total input squared displacement $V_{\text{in}}$ using the definitions from [](#def-displacement-components):
    *   $\Delta_{\text{pos}}^2 \le N \cdot V_{\text{in}}$
    *   $n_c \le \frac{N}{\lambda_{\text{status}}} \cdot V_{\text{in}}$
    *   $n_c^2 \le \left(\frac{N}{\lambda_{\text{status}}}\right)^2 \cdot V_{\text{in}}^2$
    Substituting these into the bound for $F_{\text{pot}}$ shows that $F_{\text{pot}}$ is bounded by a quadratic function of $V_{\text{in}}$: $F_{\text{pot}} <= B_2 V_{\text{in}}^2 + B_1 V_{\text{in}} + B_0$.
5.  **Finalize the Bound on the L1-Norm.**
    The term $\sqrt{F_{\text{pot}}}$ is therefore bounded by $\sqrt{B_2 V_{\text{in}}^2 + B_1 V_{\text{in}} + B_0}$, which is asymptotically linear in $V_{\text{in}}$ for large $V_{\text{in}}$. Applying [](#lem-subadditivity-power) with $\alpha=1/2$ yields $\sqrt{a+b} \le \sqrt{a} + \sqrt{b}$, so we can bound $\sqrt{F_{\text{pot}}}$ by a sum of linear and square-root terms of $V_{\text{in}}$.
    Combining all terms, the total L1-norm $\|\DeltaP\|_1$ is bounded by an expression of the form $C'_P V_{\text{in}} + H'_P sqrt(V_{\text{in}}) + K'_P$. Absorbing the term **2N** into the constant offset gives the final result as stated in the sub-lemma.
**Q.E.D.**
:::
## 17. The Revival State: Dynamics at $k=1$
:::{prf:remark} The Near-Extinction Recovery Mechanism (Phoenix Effect)
:label: rem-phoenix-effect

This is perhaps the most dramatic moment in the swarm ({prf:ref}`def-swarm-and-state-space`)'s life cycle: when disaster strikes and only one walker ({prf:ref}`def-walker`) survives. Will the swarm go extinct, or can it rebuild itself from a single survivor?

**The Beautiful Result**: Under the right conditions, one survivor is enough to resurrect the entire swarm! This section proves that the "last walker standing" scenario triggers a guaranteed revival ({prf:ref}`axiom-guaranteed-revival`) mechanism that brings all dead walkers back to life in a single step.

This is like having a "phoenix effect" built into the algorithm - the swarm can always rise from the ashes as long as one walker remains.
:::
The general continuity analysis presented in the preceding sections is valid for the regime where the number of alive walkers is at least two. The state where exactly one walker survives represents a critical boundary condition where the system's dynamics change fundamentally. This section provides a formal theorem to characterize the behavior in this "revival state," demonstrating how the foundational axioms ensure the swarm can recover from near-extinction events.
### 16.1 Theorem of Guaranteed Revival from a Single Survivor
:::{prf:theorem} Theorem of Guaranteed Revival from a Single Survivor
:label: thm-k1-revival-state
:::{admonition} The Phoenix Theorem Intuition
:class: note
:open:
**The Setup**: Only one walker ({prf:ref}`def-walker`) remains alive - the "last one standing" scenario.
**The Magic**: This theorem proves that the one survivor automatically becomes a "life generator." Here's what happens:
1. **The Survivor Stays Put**: The lone walker ({prf:ref}`def-walker`) gets score 0 (comparing itself to itself), so it doesn't clone - it just persists.
2. **All Dead Walker ({prf:ref}`def-walker`)s Revive**: Every dead walker gets an infinite cloning score (comparing to the survivor vs. their own zero fitness), guaranteeing revival.
3. **Full Resurrection**: In one step, the swarm ({prf:ref}`def-swarm-and-state-space`) goes from 1 alive walker ({prf:ref}`def-walker`) to N alive walkers!
The key insight: when there's only one alive walker ({prf:ref}`def-walker`), the cloning math becomes deterministic rather than probabilistic. The survivor can't help but revive everyone else!
:::
Let $\mathcal{S}_t$ be a swarm state ({prf:ref}`def-swarm-and-state-space`) with exactly one alive walker ({prf:ref}`def-alive-dead-sets`), such that $|\mathcal{A}(\mathcal{S}_t)| = 1$. Let the index of this single survivor ({prf:ref}`def-walker`) be $j$, so $\mathcal{A}(\mathcal{S}_t) = \{j\}$.
Assume the Axiom of Guaranteed Revival ({prf:ref}`axiom-guaranteed-revival`) holds, such that the revival score ratio $\kappa_{\text{revival}} > 1$.
Then, the one-step transition $\mathcal{S}_t \to \mathcal{S}_{t+1}$ is characterized by the following three properties with probability 1:
1.  **Survivor Persistence:** The single alive walker ({prf:ref}`def-walker`) $j$ will be assigned the "Persist" action. Its intermediate position will be its current position, $x_j^{(t+0.5)} = x_j^{(t)}$. Its subsequent evolution is that of a single, persistent random walker for the remainder of the timestep.
2.  **Dead Walker ({prf:ref}`def-walker`) Revival:** Every dead walker $i \in \mathcal{D}(\mathcal{S}_t)$ ({prf:ref}`def-alive-dead-sets`) (for $i \neq j$) will be assigned the "Clone" action. Its intermediate position $x_i^{(t+0.5)}$ will be sampled from the Cloning Measure ({prf:ref}`def-cloning-measure`) centered on the survivor's position, $\mathcal{Q}_\delta(x_j^{(t)}, \cdot)$.
3.  **Swarm Revival and Failure Condition:** The swarm is guaranteed to enter the intermediate state $\mathcal{S}_{t+0.5}$ with all $N$ walker ({prf:ref}`def-walker`)s alive ($|\mathcal{A}(\mathcal{S}_{t+0.5})| = N$). The risk of swarm extinction ($|\mathcal{A}(\mathcal{S}_{t+1})|=0$) is therefore isolated to the single, simultaneous event where all $N$ walkers in the revived intermediate swarm independently move to an invalid state during the final perturbation and status update phase.
:::{attention}
**The Only Remaining Risk**: After revival, all N walker ({prf:ref}`def-walker`)s are alive again, but they still need to survive the perturbation step. The swarm can still go extinct if ALL walkers simultaneously wander into forbidden territory during this final step. However, this is now a single, well-defined probabilistic event rather than gradual attrition - much easier to analyze and control!
:::
:::
:::{prf:proof}
:label: proof-thm-k1-revival-state

**Proof.**
The proof proceeds by analyzing the cloning decision for the single survivor and for an arbitrary dead walker ({prf:ref}`def-walker`), demonstrating that their actions are deterministic under the given conditions.
1.  **Proof of Survivor Persistence (Walker ({prf:ref}`def-walker`) **j**):**
    *   **Companion Selection:** As per the **Companion Selection Measure ({prf:ref}`def-companion-selection-measure`) ([](#def-companion-selection-measure))**, when $|\mathcal{A}|=1$, the single alive walker ({prf:ref}`def-walker`) is its own companion. Therefore, the cloning companion is deterministically $c_{\text{clone}}(j) = j$.
    *   **Cloning Score:** The fitness potentials are $V_j$ for the walker ({prf:ref}`def-walker`) and $V_{c(j)}=V_j$ for the companion. The cloning score from ([](#def-cloning-score-function)) is:

$$

        S_j = S(V_j, V_j) = \frac{V_j - V_j}{V_j + \varepsilon_{\text{clone}}} = 0

$$
*   **Cloning Decision:** The random threshold is sampled $T_{\text{clone}} \sim \text{Uniform}(0, p_{\max})$. Since $p_{\max} > 0$, the probability of sampling $T_{\text{clone}}=0$ is zero. The condition for cloning, $S_j > T_{\text{clone}}$, becomes $0 > T_{\text{clone}}$, which is false with probability 1.
    *   **Conclusion:** Walker ({prf:ref}`def-walker`) $j$ is assigned the "Persist" action. Its intermediate position is unchanged, $x_j^{(t+0.5)} = x_j^{(t)}$. This proves the first property.
2.  **Proof of Dead Walker ({prf:ref}`def-walker`) Revival (Walker **i** where **i ≠ j**):**
    *   **Companion Selection:** For a dead walker ({prf:ref}`def-walker`) $i$, the companion set is the entire alive set ({prf:ref}`def-alive-dead-sets`), $\mathcal{A}(\mathcal{S}_t)$. Since this set only contains walker $j$, the companion is deterministically $c_{\text{clone}}(i) = j$.
    *   **Fitness Potential:** As a dead walker ({prf:ref}`def-walker`), $V_i=0$. As an alive walker, the companion's potential $V_j$ is strictly positive and bounded below by $V_{\text{pot,min}} = \eta^{\alpha+\beta}$ ([](#lem-potential-boundedness)).
    *   **Cloning Score:** The cloning score for walker ({prf:ref}`def-walker`) $i$ is:

$$

        S_i = S(V_j, 0) = \frac{V_j - 0}{0 + \varepsilon_{\text{clone}}} = \frac{V_j}{\varepsilon_{\text{clone}}}

$$
Using the lower bound for $V_j$, we have a lower bound for the score: $S_i \ge \frac{\eta^{\alpha+\beta}}{\varepsilon_{\text{clone}}}$.
    *   **Cloning Decision:** The cloning action occurs if $S_i > T_{\text{clone}}$. We compare the lower bound of the score to the upper bound of the threshold ($p_{\max}$). The **Axiom of Guaranteed Revival ({prf:ref}`axiom-guaranteed-revival`)** requires $\kappa_{\text{revival}} = \frac{\eta^{\alpha+\beta}}{\varepsilon_{\text{clone}} \cdot p_{\max}} > 1$. Rearranging this axiom gives:

$$

        \frac{\eta^{\alpha+\beta}}{\varepsilon_{\text{clone}}} > p_{\max}

$$
Therefore, we have the following guaranteed inequality: $S_i \ge \frac{\eta^{\alpha+\beta}}{\varepsilon_{\text{clone}}} > p_{\max} \ge T_{\text{clone}}$.
    *   **Conclusion:** The score $S_i$ is guaranteed to be greater than any possible sampled threshold $T_{\text{clone}}$. Walker $i$ is assigned the "Clone" action with probability 1. Its intermediate position is sampled from $\mathcal{Q}_\delta(x_j^{(t)}, \cdot)$. This proves the second property for all $N-1$ dead walkers.
3.  **Proof of Swarm Revival and Failure Condition:**
    *   From (1) and (2), all $N$ walkers persist or are cloned. As per the **Swarm Update Procedure**, all walkers in the intermediate swarm ({prf:ref}`def-swarm-and-state-space`) $\mathcal{S}_{t+0.5}$ are assigned a status of alive. Thus, $|\mathcal{A}(\mathcal{S}_{t+0.5})| = N$ is guaranteed.
    *   The final state $\mathcal{S}_{t+1}$ is d ({prf:ref}`def-status-update-operator`)etermined by applying the **Perturbation Operator ({prf:ref}`def-perturbation-operator`)** and **Status Update Operator** to $\mathcal{S}_{t+0.5}$. The only way for the swarm to become extinct is if every walker $i \in \{1, \dots, N\}$ has its new position $x_i^{(t+1)}$ fall within the invalid domain.
    *   Since the perturbations are independent for each walker, the probability of total swarm failure is the product of the individual probabilities of failure. This isolates the extinction risk to a single, quantifiable event, contingent entirely on the outcomes of the **N** post-revival random walks. This proves the third property.
**Q.E.D.**
:::
## 18. Swarm Update Operator: A Composition of Measures
### 17.1 Definition: Swarm Update Operator
:::{prf:definition} Swarm Update Procedure
:label: def-swarm-update-procedure
The **swarm update operator** $\Psi: \Sigma_N \to \mathcal{P}(\Sigma_N)$ defines the one-step transition measure of the Markov process, evolving a swarm state ({prf:ref}`def-swarm-and-state-space`) $\mathcal{S}_t$ to a probability distribution over the subsequent state $\mathcal{S}_{t+1}$. A single realization $\mathcal{S}_{t+1} \sim \Psi(\mathcal{S}_t, \cdot)$ is generated by the sequential application of the following operators.
1.  **Stage 1: Cemetery State Absorption**
    *   If the input swarm is in the absorbing cemetery state ({prf:ref}`def-distance-to-cemetery-state`) ({prf:ref}`def-cemetery-state-measure`), $|\mathcal{A}(\mathcal{S}_t)|=0$ ({prf:ref}`def-alive-dead-sets`), the process terminates. The operator returns a Dirac measure on the input state: $\Psi(\mathcal{S}_t, \cdot) = \delta_{\mathcal{S}_t}(\cdot)$, such that $\mathcal{S}_{t+1} = \mathcal{S}_t$. Otherwise, the transition is defined by the composition of the following stages.
2.  **Stage 2: Stochastic Measurement and Potential Calculation**
    This stage maps the input swarm state ({prf:ref}`def-swarm-and-state-space`) $\mathcal{S}_t$ to a single, fixed N-dimensional fitness potential vector $\mathbf{V}_{\text{fit}}$, which is then used as a deterministic parameter for the remainder of the timestep.
    *   **a. Raw Measurement (Stochastic):**
        *   The raw reward vector for the alive set ({prf:ref}`def-alive-dead-sets`), $\mathbf{r}_{\mathcal{A}}$, is generated deterministically: $\mathbf{r}_{\mathcal{A}} := (R(x_i))_{i \in \mathcal{A}_t}$.
        *   The raw distance vector, $\mathbf{d}_{\mathcal{A}}$, is generated stochastically by first sampling a *potential companion* $c_{\text{pot}}(i) \sim \mathbb{C}_i(\mathcal{S}_t)$ ({prf:ref}`def-companion-selection-measure`) for each alive walker ({prf:ref}`def-walker`) $i \in \mathcal{A}_t$, then computing the algorithmic distance ({prf:ref}`def-alg-distance`): $\mathbf{d}_{\mathcal{A}} := (d_{\text{alg}}(x_i, x_{c_{\text{pot}}(i)}))_{i \in \mathcal{A}_t}$.
    *   **b. Potential Vector Calculation (Deterministic):**
        *   Using the single realization of the raw vectors $(\mathbf{r}_{\mathcal{A}}, \mathbf{d}_{\mathcal{A}})$ from the previous step, the potential vector for the alive set ({prf:ref}`def-alive-dead-sets`) is computed by applying the deterministic **Rescaled Potential Operator for the Alive Set** ([](#def-alive-set-potential-operator)):

$$

            \mathbf{V}_{\mathcal{A}} \leftarrow V_{\text{op},\mathcal{A}}(\mathcal{S}_t, \mathbf{r}_{\mathcal{A}}, \mathbf{d}_{\mathcal{A}})

$$
*   The full N-dimensional fitness potential vector is then assembled using the deterministic **Swarm Potential Assembly Operator** ([](#def-swarm-potential-assembly-operator)):

$$

            \mathbf{V}_{\text{fit}} \leftarrow A_{\text{pot}}(\mathcal{S}_t, \mathbf{V}_{\mathcal{A}})

$$
3.  **Stage 3: Cloning Transition**
    This stage maps the input swarm $\mathcal{S}_t$ and the fixed potential vector $\mathbf{V}_{\text{fit}}$ to a distribution over an intermediate swarm state ({prf:ref}`def-swarm-and-state-space`) $\mathcal{S}_{t+0.5}$. The process is defined as a product measure over the N walkers ({prf:ref}`def-walker`). For each walker $i \in \{1, \dots, N\}$:
    *   **a. Sample Cloning Companion:** An independent *cloning companion* index, $c_{\text{clone}}(i)$, is sampled from the Companion Selection Measure ({prf:ref}`def-companion-selection-measure`) $\mathbb{C}_i(\mathcal{S}_t)$.
    *   **b. Determine Action:** An action $a_i \in \{\text{Clone}, \text{Persist}\}$ is determined via the **Stochastic Threshold Cloning** procedure (Def. 15.2), which compares the walker ({prf:ref}`def-walker`)'s score against a random threshold sampled from $[0, p_{\max}]$.
    *   **c. Sample Intermediate Position:** A **Conditional Intermediate Position Measure** $\mathbb{M}_i$ on $\mathcal{X}$ is defined based on the determined action:

$$

        \mathbb{M}_i(\cdot | a_i) :=
        \begin{cases}
        \mathcal{Q}_\delta(x_{c_{\text{clone}}(i)}^{(t)}, \cdot) & \text{if } a_i = \text{Clone} \\
        \delta_{x_i^{(t)}}(\cdot) & \text{if } a_i = \text{Persist}
        \end{cases}

$$
where $\mathcal{Q}_\delta$ is the Cloning Measure ({prf:ref}`def-cloning-measure`) and $\delta_{x}$ is the Dirac delta measure. The intermediate position is then sampled: $x_i^{(t+0.5)} \sim \mathbb{M}_i(\cdot | a_i)$.
    *   **d. Form Intermediate Walker ({prf:ref}`def-walker`):** The intermediate status is set deterministically to alive, $s_i^{(t+0.5)} \leftarrow 1$, yielding the intermediate walker $w_i^{(t+0.5)} = (x_i^{(t+0.5)}, s_i^{(t+0.5)})$. The intermediate swarm is $\mathcal{S}_{t+0.5} = (w_i^{(t+0.5)})_{i=1}^N$.
4.  **Stage 4: Perturbation and Final Status Update**
    The final swarm state ({prf:ref}`def-swarm-and-state-space`) $\mathcal{S}_{t+1}$ is generated by the composition of the final two operators.
    *   **a. Perturbation:** The positions of the intermediate swarm are updated by sampling from the measure defined by the **Perturbation Operator ({prf:ref}`def-perturbation-operator`)** ([](#def-perturbation-operator)), resulting in a new swarm $\mathcal{S}_{\text{pert}}$:

$$

        \mathcal{S}_{\text{pert}} \sim \Psi_{\text{pert}}(\mathcal{S}_{t+0.5}, \cdot)

$$
*   **b. Status Update:** The final aliveness statuses are determined by applying the deterministic **Status Update Operator ({prf:ref}`def-status-update-operator`)** (Def. 14) to the perturbed swarm, yielding the final state:

$$

        \mathcal{S}_{t+1} \leftarrow \Psi_{\text{status}}(\mathcal{S}_{\text{pert}})

$$
:::
### 17.2. Continuity of the Swarm Update Operator
The **Swarm Update Operator (**Ψ**)** represents the complete one-step evolution of the Markov process. Its continuity is the cornerstone of the entire stability analysis, as it determines whether the system's dynamics are well-behaved. A continuous update operator ensures that small differences between two initial swarms will, on average, lead to small differences between the resulting swarms in the next timestep.
This section provides the capstone result of the continuity analysis by proving that the full operator is probabilistically continuous. The proof is achieved by composing the continuity bounds established for each of the sequential sub-operators in the preceding chapters.
#### 17.2.1. Formal Decomposition of the Final Operator
The final output swarm $\mathcal{S}'_k$ is the result of applying a composite operator, which we will call the **Post-Cloning Operator** $\Psi_{\text{final}} = \Psi_{\text{status}} \circ \Psi_{\text{pert}}$, to the intermediate swarm $\mathcal{S}_{k,\text{clone}}$ generated by the cloning stage. To analyze its continuity, we must bound the expected displacement between the output swarms, $\mathbb{E}[d_{\text{out}}^2] = \mathbb{E}[d_{\text{Disp},\mathcal{Y}}(\mathcal{S}'_1, \mathcal{S}'_2)^2]$, in terms of the displacement between the intermediate swarms.
The output displacement metric can be decomposed into its two constituent parts:

$$

\mathbb{E}[d_{\text{out}}^2] = \frac{1}{N}\mathbb{E}[\Delta_{\text{pos,final}}^2] + \frac{\lambda_{\mathrm{status}}}{N} \mathbb{E}[n_{c,\text{final}}]

$$
where $\Delta_{\text{pos,final}}^2$ is the final positional displacement and $n_{c,\text{final}}$ is the final number of status changes. The subsequent lemmas provide bounds for each of these two terms.
#### 17.2.2. Sub-Lemma: Bounding the Final Positional Displacement (unconditional)
:label: lem-final-positional-displacement-bound
Let $\mathcal{S}_{1,\text{clone}}$ and $\mathcal{S}_{2,\text{clone}}$ be two intermediate swarms, and let $\mathcal{S}'_1, \mathcal{S}'_2$ be the swarms that result from applying the composed Post-Cloning Operator. For any $\delta\in(0,1)$, the expected final squared positional displacement admits the unconditional bound

$$

\mathbb{E}[\Delta_{\text{pos,final}}^2] \;\le\; 3 \,\mathbb{E}[\Delta_{\text{pos,clone}}^2] \, + \, 6\,B_M(N) \, + \, 6\, D_{\mathcal{Y}}^2 \,\sqrt{\tfrac{N}{2}\,\ln\!\big(\tfrac{2}{\delta}\big)} \, + \, \delta\, N\, D_{\mathcal{Y}}^2.

$$
where $B_M(N)$ is the deterministic Mean Displacement Bound from the Perturbation Operator ({prf:ref}`def-perturbation-operator`) analysis ([](#def-perturbation-fluctuation-bounds-reproof)).
:::{prf:proof}
:label: proof-lem-final-positional-displacement-bound
**Proof.**
This follows from the probabilistic continuity of the Perturbation Operator ({prf:ref}`def-perturbation-operator`) via a standard $\delta$–split argument. From [](#thm-perturbation-operator-continuity-reproof), with probability at least $1-\delta$,

$$

\Delta_{\text{pos,final}}^2 \;\le\; 3\,\Delta_{\text{pos,clone}}^2 \, + \, 6\,\Big( B_M(N) + D_{\mathcal{Y}}^2 \,\sqrt{\tfrac{N}{2}\,\ln\!\big(\tfrac{2}{\delta}\big)}\Big).

$$
Taking expectations on this event and using the trivial bound $\Delta_{\text{pos,final}}^2 \le N D_{\mathcal{Y}}^2$ on its complement of probability $\delta$ yields

$$

\mathbb{E}[\Delta_{\text{pos,final}}^2] \;\le\; 3 \,\mathbb{E}[\Delta_{\text{pos,clone}}^2] \, + \, 6\,B_M(N) \, + \, 6\, D_{\mathcal{Y}}^2 \,\sqrt{\tfrac{N}{2}\,\ln\!\big(\tfrac{2}{\delta}\big)} \, + \, \delta\, N\, D_{\mathcal{Y}}^2.

$$
**Q.E.D.**
:::
#### 17.2.3. Bounding the Expected Final Status Change
To bound the final displacement, we must first establish a bound on the expected number of status changes that occur after the perturbation stage. This bound will depend on the expected positional displacement of the intermediate swarms generated by the cloning operator.
##### 17.2.3.1. Definition: Final Status Change Bound Coefficients

:::{prf:definition} Final Status Change Bound Coefficients
:label: def-final-status-change-coeffs

The bound on the expected final status change is determined by two coefficients derived from the foundational axioms and global parameters:
1.  **The Status Change Hölder Coefficient ($C_{\text{status},H}$):** This coefficient captures the Hölder‑continuous scaling between positional displacement and expected status changes, aggregated over $N$ walker ({prf:ref}`def-walker`)s:

$$

C_{\text{status},H} := L_{\text{death}}^2 \, N^{\,1-\alpha_B}.

$$

This choice matches the explicit inequality of Theorem 14.2, where the per‑walker ({prf:ref}`def-walker`) Hölder bound contributes $L_{\text{death}}^2\, d^{2\alpha_B}$ and summing over $N$ walkers yields the factor $N$.
2.  **The Status Change Variance Bound ($K_{\text{status},\text{var}}$):** This coefficient provides a state-independent upper bound on the total variance of the final status variables, which represents irreducible stochasticity.

$$

K_{\text{status},\text{var}} := \frac{N}{2}

$$
:::

##### 17.2.3.2. Lemma: Bounding the Expected Final Status Change

:::{prf:lemma} Bounding the Expected Final Status Change
:label: lem-final-status-change-bound

This lemma bounds the status changes introduced by {prf:ref}`def-status-update-operator`.

The expected final number of status changes, $\mathbb{E}[n_{c,\text{final}}]$, is bounded by a Hölder-continuous function of the *expected* intermediate positional displacement.

$$

\mathbb{E}[n_{c,\text{final}}] \le K_{\text{status},\text{var}} + C_{\text{status},H} \left( \mathbb{E}[\Delta_{\text{pos,clone}}^2] \right)^{\alpha_B}

$$

where $\mathbb{E}[\Delta_{\text{pos,clone}}^2]$ is the expected squared positional displacement between the two intermediate swarms.
:::

:::{prf:proof}
:label: proof-lem-final-status-change-bound
**Proof.**
The proof establishes the bound by applying the law of total expectation to the result from the **Probabilistic Continuity of the Post-Perturbation Status Update ([](#thm-post-perturbation-status-update-continuity))**.
1.  **Apply Law of Total Expectation:**
    Let the full expectation over all stochastic processes be $\mathbb{E}[\cdot]$. Let $\mathbb{E}_{\text{pert}}[\cdot | \mathcal{S}_{\text{clone}}]$ be the expectation over the perturbation process, conditioned on a specific realization of the intermediate swarms, $\mathcal{S}_{\text{clone}} = (\mathcal{S}_{1,\text{clone}}, \mathcal{S}_{2,\text{clone}})$.

$$

\mathbb{E}[n_{c,\text{final}}] = \mathbb{E}_{\text{clone}} \left[ \mathbb{E}_{\text{pert}}[n_{c,\text{final}} | \mathcal{S}_{\text{clone}}] \right]

$$
2.  **Bound the Inner Expectation:**
    The inner expectation is bounded by [](#thm-post-perturbation-status-update-continuity). Noting that $d_{\text{Disp},\mathcal{Y}}^2 = (1/N)\Delta_{\text{pos}}^2$ for the intermediate swarms (since $n_c=0$), we have:

$$

\mathbb{E}_{\text{pert}}[n_{c,\text{final}} | \mathcal{S}_{\text{clone}}] \le \frac{N}{2} + N L_{\text{death}}^2 \left( \frac{1}{N} \Delta_{\text{pos,clone}}^2 \right)^{\alpha_B} = K_{\text{status},\text{var}} + C_{\text{status},H} (\Delta_{\text{pos,clone}}^2)^{\alpha_B}

$$
3.  **Take the Outer Expectation:**
    We take the expectation of this inequality over the distribution of intermediate swarms. By linearity of expectation:

$$

\mathbb{E}[n_{c,\text{final}}] \le K_{\text{status},\text{var}} + C_{\text{status},H} \cdot \mathbb{E}_{\text{clone}}\left[\left( \Delta_{\text{pos,clone}}^2 \right)^{\alpha_B}\right]

$$
4.  **Apply Jensen's Inequality:**
    Let $X = \Delta_{\text{pos,clone}}^2$. The function $f(x) = x^{\alpha_B}$ is concave for $\alpha_B \in (0, 1]$. By Jensen's inequality for concave functions (see [](#lem-inequality-toolbox)), $\mathbb{E}[f(X)] \le f(\mathbb{E}[X])$. This gives:

$$

\mathbb{E}_{\text{clone}}\left[\left( \Delta_{\text{pos,clone}}^2 \right)^{\alpha_B}\right] \le \left( \mathbb{E}_{\text{clone}}[\Delta_{\text{pos,clone}}^2] \right)^{\alpha_B}

$$
5.  **Finalize the Bound:**
    Substituting the bound from Step 4 into the inequality from Step 3 yields the final result.
**Q.E.D.**
:::
### 17.2.4. Theorem: Continuity of the Swarm Update Operator
##### 17.2.4.0. Lemma: Inequality Toolbox
::{prf:lemma} Inequality Toolbox
:label: lem-inequality-toolbox
For non-negative reals $a,b$ and any random variable $X$ with finite second moment, the following inequalities hold:
1.  (Concavity/Jensen) For every $\alpha \in (0,1]$ and non-negative weights $(p_i)$ with $\sum_i p_i = 1$,

$$
    \left(\sum_i p_i x_i\right)^{\alpha} \ge \sum_i p_i x_i^{\alpha}.
$$
2.  (Cauchy-Schwarz) The second moment controls the squared mean:

$$
    (\mathbb{E}[X])^2 \le \mathbb{E}[X^2].
$$
3.  (Square-root subadditivity)

$$
    \sqrt{a + b} \le \sqrt{a} + \sqrt{b}.
$$
:::
::{prf:proof}
All three inequalities are classical. The first is Jensen's inequality applied to the concave function $x \mapsto x^{\alpha}$. The second is the Cauchy-Schwarz inequality with the constant function $1$. The third follows by squaring both sides and simplifying: $(\sqrt{a} + \sqrt{b})^2 = a + b + 2\sqrt{ab} \ge a + b$. \hfill$\square$
:::
:label: thm-swarm-update-operator-continuity-recorrected
Let $\mathcal{S}_1$ and $\mathcal{S}_2$ be two input swarms. Let the output swarms be generated by independent applications of the full Swarm Update Operator: $\mathcal{S}'_1 \sim \Psi(\mathcal{S}_1, \cdot)$ and $\mathcal{S}'_2 \sim \Psi(\mathcal{S}_2, \cdot)$.
:::{admonition} Scope
:class: note
Statements below are for the $k\ge 2$ regime (at least two alive walkers). The $k=1$ case is handled by the revival mechanism (Axiom of Guaranteed Revival ({prf:ref}`axiom-guaranteed-revival`)) and the dedicated single‑survivor lemmas in §15; continuity then resumes after the deterministic cloning step.
:::
The Swarm Update Operator is probabilistically continuous. For transitions occurring within the $k\ge 2$ regime (i.e., $k_1=|\mathcal A(\mathcal S_1)|\ge 2$), the expected squared **N-Particle Displacement Metric ({prf:ref}`def-n-particle-displacement-metric`)** between the two output swarms is bounded by a sum of a Lipschitz term and a Hölder term of the input displacement:

$$

\mathbb{E}[d_{\text{Disp},\mathcal{Y}}(\mathcal{S}'_1, \mathcal{S}'_2)^2] \le C_{\Psi,L}(\mathcal{S}_1, \mathcal{S}_2) \cdot d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1, \mathcal{S}_2)^2 + C_{\Psi,H}(\mathcal{S}_1, \mathcal{S}_2) \cdot \left( d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1, \mathcal{S}_2)^2 \right)^{\alpha_H^{\mathrm{global}}} + K_{\Psi}(\mathcal{S}_1, \mathcal{S}_2)

$$
where the coefficients $C_{\Psi,L}$, $C_{\Psi,H}$, and $K_{\Psi}$ are non-negative, state-dependent functions defined in the subsequent sections, and the **Composite Hölder Exponent** $\alpha_H^{\mathrm{global}}$ aggregates the strictly sub-linear exponents by taking the largest among them:

$$

\alpha_H^{\mathrm{global}} := \max\left(\alpha_B, \frac{1}{2}\right)

$$
with $\alpha_B$ being the **Boundary Smoothing Exponent**.
##### 17.2.4.1. Definition: Composite Continuity Coefficients
:label: def-composite-continuity-coeffs-recorrected
The state-dependent functions in the final continuity bound for the full Swarm Update Operator are constructed from the sequential composition of the continuity bounds of the cloning and post-cloning operators.
1.  **The Composite Lipschitz Amplification Factor ($C_{\Psi,L}(\mathcal{S}_1, \mathcal{S}_2)$):** The coefficient of the term that is linear in the input squared displacement. An explicit admissible choice is

$$
    C_{\Psi,L}(\mathcal{S}_1, \mathcal{S}_2) := 3\, C_{\text{clone},L}(\mathcal{S}_1, \mathcal{S}_2).
$$
2.  **The Composite Hölder Amplification Factor ($C_{\Psi,H}(\mathcal{S}_1, \mathcal{S}_2)$):** The coefficient of the non-linear, Hölder-continuous term. Before unifying powers, two sub‑linear contributions appear:

$$
    3\, C_{\text{clone},H}(\mathcal{S}_1, \mathcal{S}_2)\, V_{\text{in}}^{1/2}\quad\text{and}\quad \lambda_{\mathrm{status}}\, C_{\text{status},H}\, (C_{\text{clone},L}(\mathcal{S}_1, \mathcal{S}_2))^{\alpha_B}\, V_{\text{in}}^{\alpha_B}.
$$
After unification (Sub‑Lemma 17.2.4.3), these are aggregated under $\alpha_H^{\mathrm{global}} = \max(\alpha_B, \tfrac12)$. A convenient explicit choice is

$$
    C_{\Psi,H}(\mathcal{S}_1, \mathcal{S}_2) := 3\, C_{\text{clone},H}(\mathcal{S}_1, \mathcal{S}_2) + \lambda_{\mathrm{status}}\, C_{\text{status},H}\, (C_{\text{clone},L}(\mathcal{S}_1, \mathcal{S}_2))^{\alpha_B}.
$$
3.  **The Composite Offset ($K_{\Psi}(\mathcal{S}_1, \mathcal{S}_2)$):** Collect the constants from the positional and status parts and from the cloning bound. With $K_{\text{pert}}(\delta)$ from [](#lem-final-positional-displacement-bound), an explicit admissible choice is

$$

    K_{\Psi}(\mathcal{S}_1, \mathcal{S}_2) := \frac{K_{\text{pert}}(\delta)}{N} + \frac{\lambda_{\mathrm{status}}}{N} K_{\text{status},\text{var}} + 3\, K_{\text{clone}}(\mathcal{S}_1, \mathcal{S}_2) + \lambda_{\mathrm{status}}\, C_{\text{status},H}\, (K_{\text{clone}}(\mathcal{S}_1, \mathcal{S}_2))^{\alpha_B}.

$$
##### 17.2.4.2. Proof of the Composite Continuity Bound
:label: proof-composite-continuity-bound-recorrected
:::{prf:proof}
:label: proof-proof-composite-continuity-bound-recorrected
**Proof.**
The proof establishes the final continuity bound by sequentially composing the bounds for the underlying operators. The strategy is to first state the bounds on the final expected displacement in terms of the intermediate (cloning) displacement, and then substitute the bound for the intermediate displacement in terms of the initial displacement.
Let $V_{\text{in}} := d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1, \mathcal{S}_2)^2$ be the initial squared displacement. Let $\mathcal{S}_{1,\text{clone}}$ and $\mathcal{S}_{2,\text{clone}}$ be the intermediate swarms after the cloning stage, and let $\mathcal{S}'_1$ and $\mathcal{S}'_2$ be the final output swarms.
1.  **Bound the Final Displacement in Terms of the Intermediate State.**
    The expected final displacement, $\mathbb{E}[d_{\text{out}}^2] = \mathbb{E}[d_{\text{Disp},\mathcal{Y}}(\mathcal{S}'_1, \mathcal{S}'_2)^2]$, is decomposed into its positional and status components:

$$

    \mathbb{E}[d_{\text{out}}^2] = \frac{1}{N}\mathbb{E}[\Delta_{\text{pos,final}}^2] + \frac{\lambda_{\mathrm{status}}}{N} \mathbb{E}[n_{c,\text{final}}]

$$
We substitute the bounds for these two terms from the preceding lemmas:
    *   From [](#lem-final-positional-displacement-bound), the positional component is bounded unconditionally: $\mathbb{E}[\Delta_{\text{pos,final}}^2] \le 3 \cdot \mathbb{E}[\Delta_{\text{pos,clone}}^2] + K_{\text{pert}}(\delta)$, where $K_{\text{pert}}(\delta) = 6B_M(N) + 6 D_{\mathcal{Y}}^2 \sqrt{\tfrac{N}{2}\ln(\tfrac{2}{\delta})} + \delta N D_{\mathcal{Y}}^2$.
    *   From [](#lem-final-status-change-bound), the status component is bounded: $\mathbb{E}[n_{c,\text{final}}] \le K_{\text{status},\text{var}} + C_{\text{status},H} \left( \mathbb{E}[\Delta_{\text{pos,clone}}^2] \right)^{\alpha_B}$.
    Combining these gives:

$$

    \mathbb{E}[d_{\text{out}}^2] \le \frac{1}{N} \left( 3 \mathbb{E}[\Delta_{\text{pos,clone}}^2] + K_{\text{pert}} \right) + \frac{\lambda_{\mathrm{status}}}{N} \left( K_{\text{status},\text{var}} + C_{\text{status},H} \left( \mathbb{E}[\Delta_{\text{pos,clone}}^2] \right)^{\alpha_B} \right)

$$
The intermediate swarms have all walkers ({prf:ref}`def-walker`) set to "alive", so their displacement metric is purely positional: $V_{\text{clone}} = d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_{1,\text{clone}}, \mathcal{S}_{2,\text{clone}})^2 = \frac{1}{N}\Delta_{\text{pos,clone}}^2$. Thus, $\mathbb{E}[\Delta_{\text{pos,clone}}^2] = N \cdot \mathbb{E}[V_{\text{clone}}]$. Substituting this relation yields a bound in terms of the expected intermediate displacement metric, $\mathbb{E}[V_{\text{clone}}]$:

$$

    \mathbb{E}[d_{\text{out}}^2] \le 3 \mathbb{E}[V_{\text{clone}}] + \frac{K_{\text{pert}}}{N} + \frac{\lambda_{\mathrm{status}}}{N}K_{\text{status},\text{var}} + \lambda_{\mathrm{status}} C_{\text{status},H} \left(\mathbb{E}[V_{\text{clone}}]\right)^{\alpha_B}

$$
2.  **Bound the Intermediate Displacement in Terms of the Initial State.**
    From the **Mean-Square Continuity of the Cloning Transition Operator** ([](#thm-cloning-transition-operator-continuity-recorrected)), the expected intermediate displacement is bounded by a function of the initial displacement, $V_{\text{in}}$:

$$

    \mathbb{E}[V_{\text{clone}}] \le C_{\text{clone},L}(\mathcal{S}_1, \mathcal{S}_2) \cdot V_{\text{in}} + C_{\text{clone},H}(\mathcal{S}_1, \mathcal{S}_2) \cdot \sqrt{V_{\text{in}}} + K_{\text{clone}}(\mathcal{S}_1, \mathcal{S}_2)

$$
3.  **Final Composition and Simplification.**
    We substitute the bound from Step 2 into the inequality from Step 1. This results in a complex expression containing terms with exponents $1, 1/2, \alpha_B, \alpha_B/2$ of $V_{\text{in}}$. Let's analyze the structure:

$$

    \mathbb{E}[d_{\text{out}}^2] \le 3 \left( C_{\text{clone},L}V_{\text{in}} + \dots \right) + \lambda_{\mathrm{status}} C_{\text{status},H} \left( C_{\text{clone},L}V_{\text{in}} + \dots \right)^{\alpha_B} + (\text{constant terms})

$$
The expression contains a sum of multiple Hölder terms. For example, the term $(C_{\text{clone},L}V_{\text{in}} + C_{\text{clone},H}\sqrt{V_{\text{in}}} + K_{\text{clone}})^{\alpha_B}$ can be bounded. By [](#lem-subadditivity-power) (a direct consequence of [](#lem-inequality-toolbox)), for any $\alpha\in(0,1]$ and nonnegative $a,b,c$, we have $(a+b+c)^{\alpha} \le a^{\alpha} + b^{\alpha} + c^{\alpha}$. Applying this with $\alpha=\alpha_B$ gives:

$$

    (\dots)^{\alpha_B} \le (C_{\text{clone},L}V_{\text{in}})^{\alpha_B} + (C_{\text{clone},H}\sqrt{V_{\text{in}}})^{\alpha_B} + (K_{\text{clone}})^{\alpha_B}

$$
The full expression for $\mathbb{E}[d_{\text{out}}^2]$ is therefore bounded by a sum of terms of the form $A_1 V_{\text{in}} + A_2 \sqrt{V_{\text{in}}} + A_3 (V_{\text{in}})^{\alpha_B} + A_4 (V_{\text{in}})^{\alpha_B/2} + K$, where the coefficients $A_k$ and $K$ are non-negative, state-dependent functions.
4.  **Unify the Hölder Terms (case split in $V_{\text{in}}$).**
    We now have a bound that is a sum of multiple terms with different exponents: $1, 1/2, \alpha_B,$ and $\alpha_B/2$. We apply the corrected global unification from **Sub-Lemma 17.2.4.3**, which distinguishes between the regimes $V_{\text{in}}\in[0,1]$ and $V_{\text{in}}\ge 1$.
    - For $V_{\text{in}}\in[0,1]$, every sub-linear power is $\le 1$ and can be absorbed into a constant.
    - For $V_{\text{in}}\ge 1$, we bound all sub-linear powers by the largest among them. In our case, among $\{1/2,\ \alpha_B,\ \alpha_B/2\}$ the largest is $\alpha_H^{\mathrm{global}} := \max(1/2,\ \alpha_B)$.
    Keeping the linear term in $V_{\text{in}}$ separate, the sub-linear terms are aggregated into a single composite term proportional to $(V_{\text{in}})^{\alpha_H^{\mathrm{global}}}$, plus an additive constant.
    *   The term linear in $V_{\text{in}}$ defines the composite Lipschitz coefficient $C_{\Psi,L}$.
    *   The aggregated sub-linear contribution, unified by the largest sub-linear exponent $\alpha_H^{\mathrm{global}}$, defines the composite Hölder coefficient $C_{\Psi,H}$.
    *   All constant terms are collected into the composite offset $K_{\Psi}$.
    ::{admonition} Note on normalization
    The $1/N$ normalization in $d_{\text{Disp},\mathcal{Y}}^2$ is carried through by expressing Hölder terms in the normalized positional displacement $V_{\text{in}}=(1/N)\,\Delta_{\text{pos}}^2$. This avoids spurious factors of $N^{\alpha_B-1}$.
    ::
    This yields the final form of the inequality as stated in the theorem, with the case distinction implicitly handled by the sub-lemma.
**Q.E.D.**
:::
##### 17.2.4.2a. Lemma: Subadditivity of Fractional Powers
:label: lem-subadditivity-power
For any $\alpha\in(0,1]$ and any nonnegative reals $a_1,\dots,a_m$, the map $x\mapsto x^{\alpha}$ is concave and subadditive on $\mathbb{R}_{\ge 0}$. In particular,

$$

\Big( \sum_{i=1}^m a_i \Big)^{\!\alpha} \le \sum_{i=1}^m a_i^{\alpha}.

$$
:::
:::{prf:proof}
:label: proof-lem-subadditivity-power
Concavity of $x^{\alpha}$ on $[0,\infty)$ for $\alpha\in(0,1]$ is classical. For $m=2$, subadditivity follows from concavity and $f(0)=0$ via $f(a+b) \le f(a)+f(b)$. The $m$-term inequality follows by induction.
**Q.E.D.**
:::
##### 17.2.4.3. Sub-Lemma: Unifying Multiple Hölder Terms (global, with case split)
:label: lem-sub-unify-holder-terms
Let $V\ge 0$ be a non-negative real number. Let $\{p_k\}_{k=1}^M\subset(0,1]$ be a finite set of exponents and let $\{A_k\}_{k=1}^M\subset[0,\infty)$ be coefficients. Define the maximal exponent $p_{\max}:=\max_k p_k$ and the sum of coefficients $A_\Sigma:=\sum_{k=1}^M A_k$.
Then, uniformly for all $V\ge 0$, the sum of Hölder terms satisfies the global bound

$$

\sum_{k=1}^M A_k\,V^{p_k} \;\le\; A_\Sigma\,\big(\,\mathbf{1}_{[0,1]}(V) + V^{p_{\max}}\,\mathbf{1}_{[1,\infty)}(V)\,\big) \;\le\; A_\Sigma\,\big(1+V^{p_{\max}}\big).

$$
:::
:::{prf:proof}
:label: proof-lem-sub-unify-holder-terms
**Proof (case split).**
1. If $V\in[0,1]$, then $V^{p_k}\le 1$ for every $k$, hence

$$

\sum_k A_k V^{p_k} \le \sum_k A_k = A_\Sigma.

$$
2. If $V\ge 1$, then $V^{p_k}\le V^{p_{\max}}$ for every $k$, hence

$$

\sum_k A_k V^{p_k} \le \left(\sum_k A_k\right) V^{p_{\max}} = A_\Sigma\,V^{p_{\max}}.

$$
Combining the two cases gives the stated global bound. This is sharp in the sense that no uniform inequality of the form $\sum_k A_k V^{p_k} \le C\,V^{q}+K$ can hold with $q<p_{\max}$ for all $V\ge 0$.
**Q.E.D.**
:::
:::{prf:remark}
:label: rem-remark-context-4997
Following the proof of the Hölder term unification lemma ({prf:ref}`proof-lem-sub-unify-holder-terms`), this remark justifies the global exponent choice in unifying sub-linear terms. Local vs global: for $V\in[0,1]$ all sub-linear powers are $\le 1$ and can be absorbed in a constant; for $V\ge 1$ every sub-linear term is bounded above by the term with exponent $p_{\max}$. This is the only global (uniform in $V\ge 0$) way to replace a sum of distinct powers by a single power, and it justifies using $\alpha_H^{\mathrm{global}}=\max(\tfrac12,\alpha_B)$ when aggregating sub-linear exponents.
:::
##### 17.2.4.4. W2 Coupling: Removing Constant Offsets
:label: subsec-w2-coupling-offset-removal
We now recast the continuity bound in the 2-Wasserstein metric on the output space, which removes additive offsets that arise from independent randomness.
:::{prf:definition} Wasserstein-2 on the output space (quotient)
:label: def-w2-output-metric

This metric measures distances between swarm configurations ({prf:ref}`def-swarm-and-state-space`) in the output space.

et $(\overline{\Sigma}_N, \overline d_{\text{Disp},\mathcal{Y}})$ denote the $N$-particle quotient state space with the displacement metric. For two probability measures $\mu,\nu$ on $\overline{\Sigma}_N$, define

$$

W_2^2(\mu,\nu) := \inf_{\pi\in\Pi(\mu,\nu)} \int \overline d_{\text{Disp},\mathcal{Y}}(s',\tilde s')^2\,\mathrm{d}\pi(s',\tilde s'),

$$
where $\Pi(\mu,\nu)$ is the set of couplings with marginals $\mu$ and $\nu$.
:::
:::{prf:proposition} W2 continuity bound without offset (for $k\ge 2$)
:label: prop-w2-bound-no-offset
Using the Wasserstein-2 metric ({prf:ref}`def-w2-output-metric`), this proposition establishes W_2 continuity of the {prf:ref}`def-swarm ({prf:ref}`def-swarm-and-state-space`)-update-procedure` without additive offset terms.

Let $\mathcal{S}_1,\mathcal{S}_2\in\overline{\Sigma}_N$ with $k_1=|\mathcal A(\mathcal S_1)|\ge 2$ and let $\Psi$ be the Swarm Update Operator. Then

$$

W_2^2\big(\Psi(\mathcal{S}_1,\cdot),\,\Psi(\mathcal{S}_2,\cdot)\big)
\;\le\; C_{\Psi,L}(\mathcal{S}_1,\mathcal{S}_2)\, \overline d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1,\mathcal{S}_2)^2
\;+\; C_{\Psi,H}(\mathcal{S}_1,\mathcal{S}_2)\, \big(\overline d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1,\mathcal{S}_2)^2\big)^{\alpha_H^{\mathrm{global}}}.

$$
In particular, $W_2\big(\Psi(\mathcal{S},\cdot),\Psi(\mathcal{S},\cdot)\big)=0$ and the bound is compatible with continuity at zero displacement without an additive constant.
:::
:::{prf:proof}
:label: proof-prop-w2-bound-no-offset
Fix a probability space $(\Omega,\mathcal{F},\mathbb{P})$ supporting all algorithmic randomness and a measurable update map $F: \overline{\Sigma}_N\times\Omega\to\overline{\Sigma}_N$ such that $\Psi(\mathcal{S},\cdot)$ is the law of $F(\mathcal{S},\Xi)$ for $\Xi\sim\mathbb{P}$. Consider the synchronous coupling

$$

\pi_{\text{sync}} := \mathcal{L}\big( F(\mathcal{S}_1,\Xi),\, F(\mathcal{S}_2,\Xi) \big),\quad \Xi\sim\mathbb{P}.

$$
By definition of $W_2$, for any coupling $\pi$ we have $W_2^2\le \mathbb{E}_{\pi}[\overline d_{\text{Disp},\mathcal{Y}}^2]$, hence

$$

W_2^2\big(\Psi(\mathcal{S}_1,\cdot),\Psi(\mathcal{S}_2,\cdot)\big)\;\le\;\mathbb{E}\big[ \overline d_{\text{Disp},\mathcal{Y}}\big(F(\mathcal{S}_1,\Xi),F(\mathcal{S}_2,\Xi)\big)^2\big].

$$
Repeating the stagewise bounds from Section 17.2.4 under this synchronous coupling yields the same linear and sub-linear dependencies on $V_{\text{in}}=d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1,\mathcal{S}_2)^2$, while eliminating additive offsets that originate solely from independent randomness. In particular, when $\mathcal{S}_1=\mathcal{S}_2$, we have $F(\mathcal{S}_1,\Xi)=F(\mathcal{S}_1,\Xi)$ almost surely and the expectation on the right-hand side is zero. This gives the stated bound with no constant term.
The coefficients $C_{\Psi,L}, C_{\Psi,H}$ are exactly those defined in 17.2.4.1, and $\alpha_H^{\mathrm{global}}$ is as in 17.2.4. The inequality follows by combining the intermediate estimates with the unification lemma (17.2.4.3).
**Q.E.D.**
:::
:::{prf:remark}
:label: rem-remark-context-5042
Continuing from {prf:ref}`prop-w2-bound-no-offset`, the offset $K_{\Psi}$ appearing in the expectation-based bound corresponds to allowing arbitrary (e.g., independent) couplings of the output randomness. When the comparison is made in $W_2$—or, operationally, under synchronous coupling—the artificial offset vanishes at zero input distance, yielding a cleaner continuity statement. The composite constants $C_{\Psi,L}$ and $C_{\Psi,H}$ are exactly those defined in [](#def-composite-continuity-coeffs-recorrected) and inherit boundedness/continuity from [](#subsec-coefficient-regularity).
:::
##### 17.2.4.5. Measurability and Markov Kernel Structure
:label: subsec-measurability-markov-kernel
:::{prf:proposition} The Swarm Update defines a Markov kernel
:label: prop-psi-markov-kernel
This proposition establishes that {prf:ref}`def-swarm-update-procedure` defines a valid Markov kernel on the swarm space ({prf:ref}`def-swarm-and-state-space`).

Let $(\Sigma_N,\mathcal{B}(\Sigma_N))$ be the measurable state space. Assume each stage of the update—cloning, perturbation, and status update ({prf:ref}`def-status-update-operator`)—is defined by a measurable map with respect to its inputs and driven by a measurable noise kernel on a Polish probability space. Then the full Swarm Update Operator $\Psi$ is a Markov kernel on $\Sigma_N$; i.e., for each $\mathcal{S}\in\Sigma_N$, $\Psi(\mathcal{S},\cdot)$ is a probability measure on $\Sigma_N$, and for each measurable $A\in\mathcal{B}(\Sigma_N)$, the map $\mathcal{S}\mapsto \Psi(\mathcal{S},A)$ is measurable.
:::
:::{prf:proof}
:label: proof-prop-psi-markov-kernel
Let $F_{\text{clone}},F_{\text{pert}},F_{\text{status}}$ denote the measurable stage maps and let $\mathcal{K}_{\text{clone}},\mathcal{K}_{\text{pert}},\mathcal{K}_{\text{status}}$ be their noise kernels. For each fixed input, pushforward of a measurable kernel under a measurable map yields a measurable kernel. By composition, the concatenation of these stagewise kernels is a measurable kernel (standard closure of Markov kernels under composition on measurable spaces). Thus $\Psi$ is a Markov kernel on $(\Sigma_N,\mathcal{B}(\Sigma_N))$.
Moreover, the synchronous coupling used in [](#subsec-w2-coupling-offset-removal) is realized by taking the product probability space for the stagewise noises and identifying the same noise coordinate for the paired inputs. Hence the $W_2$ bound in [](#prop-w2-bound-no-offset) is a continuity statement for the Markov kernel $\Psi$ viewed as a map $\mathcal{S}\mapsto \Psi(\mathcal{S},\cdot)$.
**Q.E.D.**
:::
:::{prf:remark}
:label: rem-context-5056
The Markov kernel structure ({prf:ref}`prop-psi-markov-kernel`) implies Feller-type (continuity-preserving) properties for $\Psi$ follow from the stagewise measurability and continuity assumptions stated in Section 2 for the operators and aggregators; on compact (or sublevel) sets these imply boundedness and continuity of the induced kernel maps.
:::
##### 17.2.4.6. Regularity of State-Dependent Coefficients on Sublevel Sets
:label: subsec-coefficient-regularity
:::{prf:proposition} Boundedness and continuity of composite coefficients
:label: prop-coefficient-regularity
This proposition establishes boundedness and continuity of all state-dependent coefficients used in the continuity bounds, relying on {prf:ref}`lem-sigma-reg-derivative-bounds` and the standardization framework.

Let $\mathcal{K}_R\subset \Sigma_N\times\Sigma_N$ be any set where (i) the number of alive walkers ({prf:ref}`def-walker`) is bounded between 1 and $N$ for both inputs, (ii) positions lie in a common compact subset of $\mathcal{X}$ under $\varphi$, and (iii) the aggregator Lipschitz/Hölder functions and the regularized standard deviation parameters remain bounded. Then the state-dependent coefficients

$$

(\mathcal{S}_1,\mathcal{S}_2)\mapsto C_{\text{clone},L/H},\ K_{\text{clone}},\ C_{\Psi,L/H},\ K_{\Psi},\ C_{S,\text{direct}},\ C_{S,\text{indirect}},\ C_{V,\text{total}}

$$
are bounded on $\mathcal{K}_R$ and jointly continuous in $(\mathcal{S}_1,\mathcal{S}_2)$.
:::
:::{prf:proof}
:label: proof-prop-coefficient-regularity
By definitions in Sections 11.3 and 17.2.4.1, each coefficient is obtained from the stagewise Lipschitz/Hölder functions and bounded parameters via finite sums, products, maxima, and composition with continuous operations (including the map $x\mapsto x^{\alpha}$ for $\alpha\in(0,1]$). Under assumptions (i)–(iii), the inputs to these algebraic operations are bounded and continuous on $\mathcal{K}_R$ by the aggregator axioms and the properties of the patched $\sigma'$ (see [](#lem-sigma-reg-derivative-bounds)). Continuity is preserved under sums, products, and composition; boundedness follows from continuity on compact (or closed, bounded) sets. Hence all listed coefficients are bounded on $\mathcal{K}_R$ and jointly continuous in $(\mathcal{S}_1,\mathcal{S}_2)$.
**Q.E.D.**
:::
:::{prf:remark}
:label: rem-context-5075
In particular, on such sublevel sets the $W_2$ continuity bound and the deterministic standardization bounds promote to genuine continuity statements for the composite operators since the constants do not blow up along admissible sequences.
##### 17.2.5. Feller property (stagewise and composition)
::{prf:lemma} Deterministic continuous maps induce Feller kernels (Meyn–Tweedie)
If $T: \Sigma_N\to\Sigma_N$ is continuous, then the kernel $\mathcal{K}_T(\mathcal{S},\cdot):=\delta_{T(\mathcal{S})}$ is Feller; for every $f\in C_b(\Sigma_N)$ the map $\mathcal{S}\mapsto \int f\,\mathrm{d}\mathcal{K}_T(\mathcal{S},\cdot)=f\big(T(\mathcal{S})\big)$ is continuous.
:::{prf:proof}
:label: proof-prop-coefficient-regularity-2
Continuity of $T$ and $f$ implies continuity of $f\circ T$, so evaluating the kernel against $f$ preserves continuity.
**Q.E.D.**
:::
::{prf:lemma} Perturbation is Feller (Meyn–Tweedie)
Assume [](#def-axiom-bounded-second-moment-perturbation) and that $x\mapsto \mathcal{P}_\sigma(x,\cdot)$ has a continuous density on the algorithmic space ({prf:ref}`def-algorithmic-space-generic`). Then the perturbation kernel $\mathcal{K}_{\text{pert}}$ is Feller.
:::{prf:proof}
:label: proof-prop-coefficient-regularity-3
Let $f\in C_b(\Sigma_N)$. The perturbed state is obtained by sampling positions independently according to the product density $\prod_i p_\sigma(x_i,\cdot)$ while statuses remain fixed. The integrand $f$ is bounded and continuous, and the density is jointly continuous in $(x_i)_{i=1}^N$. Dominated convergence (dominator $\|f\|_\infty$) therefore gives continuity of $\mathcal{S}\mapsto \int f\,\mathrm d\mathcal{K}_{\text{pert}}(\mathcal{S},\cdot)$.
**Q.E.D.**
:::
::{prf:lemma} Status after perturbation and cloning are Feller
The deterministic status map $T_{\text{status}}$ is generally discontinuous, so the kernel $\mathcal{K}_{\text{status}}(\mathcal{S},\cdot)=\delta_{T_{\text{status}}(\mathcal{S})}$ need not be Feller. However, if the perturbation kernel has a continuous density and the boundary-regularity axiom holds, then $\mathcal{K}_{\text{status}}\circ\mathcal{K}_{\text{pert}}$ is Feller. Moreover, the cloning kernel is Feller under the stated Lipschitz/Hölder continuity of the selection and replication maps together with the measurability convention below.
:::{prf:proof}
:label: proof-line-5092
Let $f\in C_b(\Sigma_N)$. The composition kernel integrates $f\circ T_{\text{status}}$ against the perturbation density. By [](#axiom-boundary-regularity), the boundary separating alive and dead configurations has zero measure under the perturbation density; away from that null set $f\circ T_{\text{status}}$ is continuous. Dominated convergence thus yields continuity of $\mathcal{S}\mapsto \int f\circ T_{\text{status}}(\mathcal{S}')\,\mathcal{K}_{\text{pert}}(\mathcal{S},\mathrm d\mathcal{S}')$. For cloning, the selection probabilities and displacement kernels are continuous in the input state by the axioms in Section 2; applying the deterministic lemma above shows that evaluating $f$ against the cloning kernel is continuous. (Assumption A ensures the within-step independence required by the concentration bounds earlier, and those same independent draws define the cloning kernel here.)
**Q.E.D.**
:::
::{prf:proposition} Composition preserves Feller (Meyn–Tweedie)
The composition of Feller kernels is Feller. Hence, under the axioms in Section 2, the full update kernel $\Psi$ is Feller on $(\Sigma_N,d_{\text{Disp},\mathcal{Y}})$.
:::
:::{admonition} Measurability note
:class: note
All selection, cloning and aggregation maps are Borel on $\Sigma_N$, being built from basic Borel operations (finite products, CDF inversion, order statistics, and continuous compositions).
:::
:::{admonition} Analytical coupling vs. in‑run independence
:class: note
We compare two runs via synchronous coupling (same noise seeds) to control $W_2$ distances. This is a proof device only. Within a single run and timestep, per‑walker random inputs remain independent (Assumption A). The $W_2$‑optimized bound and the expectation‑based bound are distinct results proved with different couplings; the former eliminates additive offsets at zero input distance.
:::
:::
## 19. Fragile Gas: The Algorithm's Execution
The preceding sections have defined all the necessary mathematical objects, operators, and axiomatic constraints that constitute the Fragile framework. This final section formally defines the algorithm's execution, describing how the one-step **Swarm Update Operator** is used to evolve a swarm over time.
### 18.1 The Fragile Swarm Instantiation
To distinguish the general analytical framework from a specific, runnable instance of the algorithm, we define a **Fragile Swarm**. This object is a complete instantiation of the system, where all user-selectable parameters, functions, and measures have been fixed.
:::{prf:definition} Fragile Swarm Instantiation
:label: def-fragile-swarm-instantiation
This definition packages all the components required to execute the {prf:ref}`def-fragile-gas-algorithm`, which applies the swarm update procedure ({prf:ref}`def-swarm-update-procedure`) iteratively.

A **Fragile Swarm**, denoted $\mathcal{F}$, is a tuple that encapsulates a complete and fixed configuration of the algorithm. It contains:
1.  **The Foundational & Environmental Parameters:** The full set of environmental structures, including the State Space $(\mathcal{X}, d_{\mathcal{X}})$, Valid Domain $\mathcal{X}_{\mathrm{valid}}$, Reward Function $R$, Algorithmic Space ({prf:ref}`def-algorithmic-space-generic`) $(\mathcal{Y}, d_{\mathcal{Y}})$, and Projection Map $\varphi$.
2.  **The Core Algorithmic Parameters:** A specific, fixed set of all tunable values, including the number of walker ({prf:ref}`def-walker`)s $N$, dynamics weights $(\alpha, \beta)$, noise scales $(\sigma, \delta)$, and all regulation and threshold parameters $(p_{\max}, \eta, \varepsilon_{\text{std}}, z_{\max}, \varepsilon_{\text{clone}})$.
3.  **The Concrete Operator Choices:** The specific, user-chosen functions for the **Reward Aggregation Operator** ($R_{agg}$) and the **Distance Aggregation Operator** ($M_D$).
4.  **The Concrete Noise Measure Choices:** The specific, user-chosen probability measures for the **Perturbation Measure ({prf:ref}`def-perturbation-measure`)** ($\mathcal{P}_\sigma$) and the **Cloning Measure ({prf:ref}`def-cloning-measure`)** ($\mathcal{Q}_\delta$).
A Fragile Swarm instantiation must satisfy all axioms defined in Section 2 for the analytical framework to apply. It represents a single, well-defined point in the algorithm's vast parameter space.

The concrete instantiation for the Euclidean Gas is provided in {prf:ref}`02_euclidean_gas`, where all parameters and operators are specified with explicit values.
:::
### 18.2 The Fragile Gas Algorithm
The **Fragile Gas** is the algorithm that describes the temporal evolution of a swarm of N walker ({prf:ref}`def-walker`)s. It generates a trajectory of swarm states by repeatedly applying the Swarm Update Operator, which is fully parameterized by a specific Fragile Swarm instantiation.
:::{prf:definition} The Fragile Gas Algorithm
:label: def-fragile-gas-algorithm
The **Fragile Gas Algorithm** generates a sequence of swarm ({prf:ref}`def-swarm-and-state-space`) states (a trajectory) by evolving an initial swarm over a discrete number of timesteps.
**Inputs:**
*   A **Fragile Swarm Instantiation**, $\mathcal{F}$, which fixes all parameters and operators.
*   An **initial swarm state ({prf:ref}`def-swarm-and-state-space`)**, $\mathcal{S}_0 \in \Sigma_N$.
*   A total number of **timesteps**, $T \in \mathbb{N}$.
**Process:**
The algorithm generates a trajectory of swarm states, $(\mathcal{S}_t)_{t=0}^T$, as a realization of a time-homogeneous Markov chain on the state space $\Sigma_N$.
Let $\Psi_{\mathcal{F}}$ be the **Swarm Update Operator** ([](#def-swarm-update-procedure)) fully parameterized by the choices fixed in the Fragile Swarm $\mathcal{F}$.
For each timestep $t$ from $0$ to $T-1$, the subsequent swarm state ({prf:ref}`def-swarm-and-state-space`) $\mathcal{S}_{t+1}$ is generated by sampling from the probability measure produced by applying the update operator to the current state $\mathcal{S}_t$:

$$

\mathcal{S}_{t+1} \sim \Psi_{\mathcal{F}}(\mathcal{S}_t, \cdot)

$$
**Output:**
The algorithm outputs the full trajectory of swarm states: $(\mathcal{S}_0, \mathcal{S}_1, \dots, \mathcal{S}_T)$.

This general algorithm definition is instantiated as the Euclidean Gas in {prf:ref}`02_euclidean_gas` with axiom-by-axiom validation.
:::
## 20. Canonical Instantiation and Axiom Validation
The preceding framework is built upon a comprehensive set of axioms that a user's specific instantiation of the algorithm must satisfy for the stability and convergence guarantees to hold. A critical question for the rigor of the entire framework is whether these axioms are satisfiable in practice for any non-trivial system, or if they are mutually exclusive or so restrictive as to be vacuous.
This section serves as an existence proof. We define a **Canonical Fragile Swarm** by making a set of standard, well-behaved choices for the environment, noise measures, and aggregation operators. We then formally prove, axiom by axiom, that this canonical instantiation satisfies the full set of foundational requirements. This demonstrates that the axiomatic framework is not empty and provides a concrete, verifiable baseline for users implementing their own systems.
### 19.1. Definition: The Canonical Fragile Swarm
The Canonical Fragile Swarm is defined by the following set of concrete choices.
*   **A. Foundational & Environmental Parameters:**
    *   **State Space ($\mathcal{X}$):** A bounded, convex subset of Euclidean space, $\mathcal{X} \subset \mathbb{R}^d$.
    *   **State Space Metric ($d_{\mathcal{X}}$):** The standard Euclidean distance.
    *   **Valid Domain ($\mathcal{X}_{\mathrm{valid}}$):** A compact, convex subset of $\mathcal{X}$ with a smooth ($C^1$) boundary.
    *   **Reward Function ($R$):** A function $R: \mathcal{X} \to [0, R_{\max}]$ that is globally Lipschitz on the working domain: either $\mathcal X$ is compact with $R\in C^1$ or $\sup_{x\in\mathcal X}\|\nabla R(x)\|\le L_R<\infty$. In either case, $|R(x)-R(y)|\le L_R\, d_{\mathcal X}(x,y)$.
    *   **Algorithmic Space ({prf:ref}`def-algorithmic-space-generic`) ($\mathcal{Y}, d_{\mathcal{Y}}$):** Identical to the state space, $(\mathcal{X}, d_{\mathcal{X}})$.
    *   **Projection Map ($\varphi$):** The identity map, $\varphi(x) = x$. Its Lipschitz constant is $L_\varphi = 1$.
*   **B. Core Algorithmic Parameters & Operators:**
    *   **Number of Walker ({prf:ref}`def-walker`)s ($N$):** Any integer $N \ge 2$.
    *   **Aggregation Operators ($R_{agg}, M_D$):** Both are chosen to be the **Empirical Measure Aggregator** as defined and analyzed in [](#lem-empirical-aggregator-properties).
    *   **Perturbation Measure ({prf:ref}`def-perturbation-measure`) ($\mathcal{P}_\sigma$):** A Gaussian (Heat Kernel) measure with covariance $\sigma^2 I$, where $I$ is the identity matrix. The probability density is $p_\sigma(x'|x) \propto \exp(-\|x'-x\|^2 / (2\sigma^2))$.
    *   **Cloning Measure ({prf:ref}`def-cloning-measure`) ($\mathcal{Q}_\delta$):** A Gaussian measure with covariance $\delta^2 I$.
    *   All other scalar parameters ($\alpha, \beta, \sigma, \delta, p_{\max}, \eta, \varepsilon_{\text{std}}, z_{\max}, \varepsilon_{\text{clone}}$) are assumed to be chosen as positive constants that collectively satisfy the global constraints, such as the **Axiom of Guaranteed Revival ({prf:ref}`axiom-guaranteed-revival`)**.
This complete, concrete instantiation will now be used to validate the axiomatic framework.
## 21. Continuity Constants: Canonical Values and Glossary
This section consolidates the constants used across absolute-value and mean‑square continuity results, with short definitions and their explicit expressions under the Canonical Fragile Swarm (Section 19). It is organized to show how algorithmic parameters and design choices impact stability bounds.
### 20.1 Quick Table (symbols, meaning, canonical value)
| Symbol | Type | Meaning | Canonical value (closed form) |
|---|---|---|---|
| $R_{\max}$ | env | Reward upper bound | given by environment |
| $L_R$ | env | Reward Lipschitz const. on $\mathcal{X}$ | given by environment |
| $D_{\mathcal{Y}}$ | geom | Diameter of algorithmic space ({prf:ref}`def-algorithmic-space-generic`) | $\text{diam}_{d_{\mathcal{Y}}}(\mathcal{Y})$ |
| $L_\varphi$ | geom | Projection Lipschitz const. | $1$ (identity) |
| $\sigma$ | noise | Perturbation noise scale | user‑set $> 0$ |
| $\delta$ | noise | Cloning noise scale | user‑set $> 0$ |
| $\alpha, \beta$ | dyn | Dynamics weights (reward/diversity) | user‑set $\geq 0$ |
| $\varepsilon_{\text{std}}$ | reg | Std‑dev regularizer | user‑set $> 0$ |
| $\kappa_{\text{var,min}}$ | reg | Variance floor threshold | user‑set $> 0$ |
| $\eta$ | reg | Rescale lower bound (shift) | user‑set $> 0$ |
| $z_{\max}$ | reg | Rescale saturation knee | user‑set $> 0$ |
| $p_{\max}$ | clone | Max clone threshold | user‑set $> 0$ |
| $\varepsilon_{\text{clone}}$ | clone | Clone denom. regularizer | user‑set $> 0$ |
| $k$ | struct | Alive count $|\mathcal{A}(\mathcal{S})|$ | state‑dependent $\geq 1$ |
| $n_c$ | struct | Status changes between swarms | state‑dependent $\in \{0,\ldots,N\}$ |
| $V_{\max}$ | meas | Bound on raw values $|v_i|$ | given by measurement |
| $\sigma'_{\min,\text{bound}}$ | std | Lower bound on smoothed std | $\sqrt{\kappa_{\text{var,min}} + \varepsilon_{\text{std}}^2}$ |
| $L_{\mu,M}$ | agg | Mean Lipschitz (value) | empirical: $1/\sqrt{k}$ |
| $L_{m_2,M}$ | agg | Second moment Lipschitz (value) | empirical: $2 V_{\max}/\sqrt{k}$ |
| $L_{\text{var}}$ | agg | Variance Lipschitz const. | empirical: $L_{m_2,M}+2 V_{\max} L_{\mu,M} = 4 V_{\max}/\sqrt{k}$ |
| $L_{\sigma\'_{\text{reg}}}$ | std | Lipschitz of $\sigma\'_{\text{reg}}$ | explicit from Lemma {prf:ref}`lem-sigma-reg-derivative-bounds` |
| $L_{\sigma',M}$ | std | Lipschitz of regularized $\sigma'$ | $L_{\sigma\'_{\text{reg}}} \cdot (L_{m_2,M}+2V_{\max} L_{\mu,M}) = \frac{1}{2\sigma'_{\min}} \cdot \frac{4 V_{\max}}{\sqrt{k}}$ |
| $L_P$ | rescale | Poly patch deriv. bound | $1 + \frac{(3 \log 2 - 2)^2}{3(2 \log 2 - 1)} \approx 1.0054$ |
| $L_{g_A}$ | rescale | Rescale Lipschitz | $L_P$ |
| $L_{g_A \circ z}$ | std+rescale | Std→rescale Lipschitz | $\leq \max\{L_P,1\} \cdot \tfrac{2 L_{\sigma\'_{\text{reg}}} V_{\max}}{\sqrt{k}}$ |
| $V_{\text{pot,min}}$ | pot | Min fitness potential | $\eta^{\alpha+\beta}$ |
| $V_{\text{pot,max}}$ | pot | Max fitness potential | piecewise rescale: $(g_{A,\max}+\eta)^{\alpha+\beta}$, with $g_{A,\max}=\log(1+z_{\max})+1$ |
| $L_{\pi,c}$ | clone prob | Lipschitz w.r.t. companion potential | $1/(p_{\max}\,\varepsilon_{\text{clone}})$ |
| $L_{\pi,i}$ | clone prob | Lipschitz w.r.t. walker potential | $(V_{\text{pot,max}}+\varepsilon_{\text{clone}})/(p_{\max}\,\varepsilon_{\text{clone}}^2)$ |
| $C_{\text{struct}}^{(\pi)}(k)$ | clone struct | Companion‑measure change factor | $2/\max(1,k-1)$ |
| $C_{\text{val}}^{(\pi)}$ | clone value | $\max(L_{\pi,c}, L_{\pi,i})$ | as left |
| $M_{\text{pert}}^2$ | pert | Max expected sq. displacement | $L_\varphi^2 \cdot \mathbb{E}[\|\xi\|^2] = d \sigma^2$ (Gaussian in $\mathbb{R}^d$) |
| $B_M(N)$ | pert | Mean displacement bound | $N \cdot M_{\text{pert}}^2 = N d \sigma^2$ |
| $B_S(N,\delta)$ | pert | Stochastic fluctuation bound | $D_{\mathcal{Y}}^2 \sqrt{N/2 \cdot \ln(2/\delta)}$ |
| $L_{\text{death}}$ | boundary | Hölder const. for death prob. | heat kernel: $\leq C_d (\text{Per}(\varphi(E))/\sigma) \cdot L_\varphi$ |
| $\alpha_B$ | boundary | Boundary Hölder exponent | heat kernel on smooth boundary: $1$ |
| $C_{\text{pos},d}$ | dist | Positional part constant | $12$ |
| $C_{\text{status},d}^{(1)}$ | dist | Linear status part const. | $D_{\mathcal{Y}}^2$ |
| $C_{\text{status},d}^{(2)}(k_1)$ | dist | Quadratic status part const. | $\frac{8 k_1 D_{\mathcal{Y}}^2}{(k_1 - 1)^2}$ (for $k_1\geq 2$) |
Notes:
- Empirical aggregator constants (row group “agg”) come from the closed‑form gradient bounds of empirical moments.
- For logistic rescale (Section 8.3), set $g_{A,\max}=2$ and keep $L_{g_A}=1/2$ instead of $L_P$.
### 20.2 Fully Expanded mean‑square constant for standardization
For a fixed state with $k=|\mathcal{A}|\geq 1$, the Expected Squared Value Error obeys

$$

E^2_{V,\text{ms}}(\mathcal{S}_1,\mathcal{S}_2) \leq C_{V,\text{total}}(\mathcal{S}_1) \cdot F_{V,\text{ms}}(\ldots, \ldots)

$$
with the Total Value Error Coefficient

$$

C_{V,\text{total}}(\mathcal{S}) = 3 \cdot (C_{V,\text{direct}} + C_{V,\mu}(\mathcal{S}) + C_{V,\sigma}(\mathcal{S}))

$$
and, under the canonical empirical aggregator with regularized $\sigma'$:
- $C_{V,\text{direct}} = 1/\sigma'^2_{\min,\text{bound}} = 1/(\kappa_{\text{var,min}} + \varepsilon_{\text{std}}^2)$
- $C_{V,\mu}(\mathcal{S}) = k (L_{\mu,M})^2 / \sigma'^2_{\min,\text{bound}} = 1/(\kappa_{\text{var,min}} + \varepsilon_{\text{std}}^2)$
- $C_{V,\sigma}(\mathcal{S}) = \dfrac{64 V_{\max}^4 L_{\sigma\'_{\text{reg}}}^2}{\sigma'^4_{\min,\text{bound}}}$ with $L_{\sigma\'_{\text{reg}}}$ given explicitly in [](#lem-sigma-reg-derivative-bounds).
Putting it together:

$$

C_{V,\text{total}}(\mathcal{S}) = 3 \cdot \left( \frac{2}{\sigma'^2_{\min,\text{bound}}} + \frac{64 V_{\max}^4 L_{\sigma\'_{\text{reg}}}^2}{\sigma'^4_{\min,\text{bound}}} \right)

$$
When the variance floor satisfies $\kappa_{\text{var,min}} \ll \varepsilon_{\text{std}}^2$, the leading term behaves as $C_{V,\text{total}}(\mathcal{S}) \in O(\varepsilon_{\text{std}}^{-6})$, reproducing the familiar sensitivity to the standardization regularizer.
### 20.3 Distance measurement mean‑square bound (empirical companion)
For distance‑to‑companion $d$ (Sections 10.2–10.3, $k_1=|\mathcal{A}(\mathcal{S}_1)|$):

$$

F_{d,\text{ms}}(\mathcal{S}_1, \mathcal{S}_2)

$$
given by [](#thm-distance-operator-mean-square-continuity) with canonical constants from the table:
- $C_{\text{pos},d} = 12$
- $C_{\text{status},d}^{(1)} = D_{\mathcal{Y}}^2$
- $C_{\text{status},d}^{(2)}(k_1) = \frac{8 k_1 D_{\mathcal{Y}}^2}{(k_1 - 1)^2}$ for $k_1\geq 2$.
Here $\Delta_{\text{pos}}^2 = \sum_i \|\varphi(x_{1,i})-\varphi(x_{2,i})\|_{\mathcal{Y}}^2$ and $n_c=\sum_i (s_{1,i}-s_{2,i})^2$.
### 20.4 Cloning probability and potential pipeline
- Clone‑probability Lipschitz (per‑walker):
  - $L_{\pi,c} = 1/(p_{\max}\,\varepsilon_{\text{clone}})$
  - $L_{\pi,i} = (V_{\text{pot,max}}+\varepsilon_{\text{clone}})/(p_{\max}\,\varepsilon_{\text{clone}}^2)$
  - $C_{\text{val}}^{(\pi)} = \max(L_{\pi,c}, L_{\pi,i})$, $C_{\text{struct}}^{(\pi)}(k)=2/\max(1,k-1)$
- Potential bounds: $V_{\text{pot,min}}=\eta^{\alpha+\beta}$, $V_{\text{pot,max}}=(g_{A,\max}+\eta)^{\alpha+\beta}$ with piecewise rescale $g_{A,\max}=\log(1+z_{\max})+1$.
- Fitness potential is mean‑square continuous with bound assembled from the reward and distance standardization pipelines; its coefficients inherit the sensitivities above and $L_{g_A}$.
### 20.5 Perturbation and boundary constants (final‑stage composition)
- Perturbation (Gaussian, $\varphi=\text{Id}$):
  - $M_{\text{pert}}^2 = d \sigma^2$, $B_M(N)=N d \sigma^2$, $B_S(N,\delta)=D_{\mathcal{Y}}^2 \sqrt{N/2 \cdot \ln(2/\delta)}$.
- Post‑perturbation death probability: heat‑kernel smoothing on smooth boundary yields Hölder exponent $\alpha_B=1$ and
  $L_{\text{death}} \leq C_d (\text{Per}(E)/\sigma) \cdot L_\varphi = C_d \text{Per}(E)/\sigma$.
### 20.6 One‑step mean‑square continuity (structure of the final bound)
The Swarm Update Operator satisfies

$$

\mathbb{E}[d_{\text{out}}^2] \leq C_{\Psi,L} \cdot V_{\text{in}} + C_{\Psi,H} \cdot (V_{\text{in}})^{\alpha_H^{\mathrm{global}}} + K_{\Psi}

$$

with $V_{\text{in}} = d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1,\mathcal{S}_2)^2$ and $\alpha_H^{\mathrm{global}} = \max(1/2, \alpha_B)$.

Identifying the composed coefficients from Sections 15 and 17 under the canonical choices:
- Intermediate (cloning) stage constants via the sum of clone probabilities:
  - $C_{\text{clone},L} = 3 + (3 D_{\mathcal{Y}}^2 \cdot C_P)/N$
  - $C_{\text{clone},H} = (3 D_{\mathcal{Y}}^2 \cdot H_P)/N$
  - $K_{\text{clone}} = (3 D_{\mathcal{Y}}^2 \cdot K_P)/N$
  where $(C_P, H_P, K_P)$ are finite, state‑dependent aggregates of the standardization/potential and cloning‑probability constants above.
- Final coefficients:
  - $C_{\Psi,L} = 3 \cdot C_{\text{clone},L}$
  - $C_{\Psi,H} \leq 3 \cdot C_{\text{clone},H} + \lambda_{\text{status}} \cdot C_{\text{status},H} \cdot (C_{\text{clone},L})^{\alpha_B}$, with $C_{\text{status},H} = L_{\text{death}}^2 \cdot N^{1-\alpha_B}$
  - $K_{\Psi} = 6 M_{\text{pert}}^2 + 6 D_{\mathcal{Y}}^2 \sqrt{\frac{1}{2N} \ln\left(\frac{2}{\delta}\right)} + \delta D_{\mathcal{Y}}^2 + \frac{\lambda_{\text{status}}}{2} + 3 K_{\text{clone}} + \lambda_{\text{status}} C_{\text{status},H} (K_{\text{clone}})^{\alpha_B}$

Interpretation:
- Increasing $\varepsilon_{\text{std}}$ or $\kappa_{\text{var,min}}$ decreases $C_{V,\text{total}}$, reducing sensitivity in the standardization stage.
- Larger $k$ (alive walkers) tightens empirical‑moment Lipschitz factors (via $1/\sqrt{k}$).
- Smaller $\sigma$ (tighter perturbation) and smoother boundaries (smaller $L_{\text{death}}$, $\alpha_B$ closer to 1) reduce the post‑perturbation constants.
- Conservative $p_{\max}$ and larger $\eta^{\alpha+\beta}$ (relative to $\varepsilon_{\text{clone}}$) shrink $L_{\pi,\cdot}$ and thus the cloning‑related amplification.

This consolidated view should make it straightforward to reason about parameter trade‑offs: which knobs control the Lipschitz vs. Hölder parts of the one‑step bound, and how canonical choices translate into concrete constants.

## Appendix A. Dependent bounded differences (optional)

:::{admonition} If in‑step independence is relaxed
:class: note
If an implementation intentionally uses within‑step dependence (e.g., systematic resampling via a shared uniform), McDiarmid’s inequality does not apply. In that case, a dependent bounded‑differences inequality (e.g., Warnke’s “typical bounded differences”) can be substituted, introducing a dependency parameter into the constants. The qualitative continuity structure remains but tail bounds weaken accordingly.
:::

## Appendix B. References (selected)

- Santambrogio, F. Optimal Transport for Applied Mathematicians. Springer. Foundational OT/Wasserstein facts on Polish spaces and existence of optimal plans.
- Boucheron, S.; Lugosi, G.; Massart, P. Concentration Inequalities. Standard statements of McDiarmid/bounded differences.
- Meyn, S. P.; Tweedie, R. L. Markov Chains and Stochastic Stability. Feller properties and composition of Markov kernels.
- Fritsch, F. N.; Carlson, R. E. Monotone Piecewise Cubic Interpolation (1980). Hyman, J. M. Monotonicity Preserving Cubic Interpolation (1983).
- Arendt, W. Heat kernels. Lecture notes. Weierstrass transform / smoothing by heat semigroup.
- Comi, G. E.; Torres, M. Sets of Finite Perimeter and Geometric Variational Problems. Boundary regularity and perimeter.
- Warnke, L. Typical Bounded Differences. Combinatorics, Probability and Computing. Dependent bounded differences.
- Salmon, J. K. et al. Random123: Parallel Random Numbers as Easy as 1, 2, 3. Counter‑based PRNG streams.
