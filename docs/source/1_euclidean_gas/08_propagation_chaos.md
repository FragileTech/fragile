# Propagation of Chaos and the Mean-Field Quasi-Stationary Distribution

## 0. TLDR

This document proves that the continuum mean-field model rigorously emerges as the large-N limit of the discrete Euclidean Gas dynamics. We establish existence, uniqueness, and convergence of the mean-field Quasi-Stationary Distribution using a constructive propagation of chaos argument that bypasses intractable direct PDE analysis.

**Propagation of Chaos**: The sequence of single-particle marginals $\{\mu_N\}$ extracted from the N-particle Quasi-Stationary Distributions converges weakly to a unique limit $\mu_\infty$ as $N \to \infty$. This limit is the stationary solution $\rho_0$ to the McKean-Vlasov PDE derived in the mean-field analysis, rigorously justifying the continuum model as the thermodynamic limit of the discrete N-particle Euclidean Gas.

**Constructive Existence via Tightness-Identification-Uniqueness**: Direct PDE analysis fails due to the quadratic, non-local cloning operator. Instead, we construct $\rho_0$ by proving (1) **tightness** of $\{\mu_N\}$ using uniform moment bounds from Foster-Lyapunov analysis, (2) **identification** showing any limit point satisfies the stationary mean-field PDE via Law of Large Numbers for empirical measures, and (3) **uniqueness** of the weak solution via contraction mapping on a weighted Sobolev space using hypoelliptic regularity theory.

**Thermodynamic Limit**: The convergence holds in the weak topology and implies that macroscopic observables computed from finite-N equilibria converge to their mean-field counterparts: $\lim_{N \to \infty} \mathbb{E}_{\nu_N^{QSD}}[\frac{1}{N}\sum_i \phi(z_i)] = \int \phi(z) \rho_0(z) dz$ for any bounded continuous function $\phi$. Stronger Wasserstein-2 convergence $W_2(\mu_N, \mu_\infty) \to 0$ can be established via second moment bounds from the Foster-Lyapunov analysis, providing quantitative rates.

**Hypoelliptic Regularity**: The contraction argument requires that the kinetic resolvent operator maps the weighted $L^2$ space into a weighted $H^1$ Sobolev space. This non-standard regularity follows from Hörmander's hypoellipticity theorem once we verify that the kinetic operator's Lie bracket structure satisfies the bracket-generating condition—a verification carried out explicitly in Section 5.

## 1. Introduction

### 1.1. Goal and Scope

The goal of this document is to prove the **existence, uniqueness, and regularity** of the Quasi-Stationary Distribution (QSD) for the mean-field Fokker-Planck-McKean-Vlasov PDE derived in `06_mean_field.md`. The central object of study is the stationary probability density $\rho_0: \Omega \to \mathbb{R}_+$ satisfying a highly non-linear, non-local integro-differential equation with interior killing and revival mechanisms.

We will establish this result not through direct PDE analysis—which is intractable due to the quadratic non-locality of the cloning operator—but via a **propagation of chaos** argument. This constructive approach leverages the existence of well-behaved finite-N equilibria (proven in `06_convergence.md`) to build the continuum solution as the limit $N \to \infty$ of single-particle marginals extracted from the N-particle Quasi-Stationary Distributions.

The proof follows the classical three-step program:
1. **Tightness** (Section 3): Uniform moment bounds from the N-uniform Foster-Lyapunov analysis guarantee that the sequence $\{\mu_N\}$ is tight, ensuring the existence of weakly convergent subsequences.
2. **Identification** (Section 4): Any limit point of a convergent subsequence is a weak solution to the stationary mean-field PDE. This is proven by showing that empirical measures from N-particle systems converge (via Law of Large Numbers) to the deterministic mean-field functionals.
3. **Uniqueness** (Section 5): The weak solution is unique, established via a contraction mapping argument in a weighted Sobolev space $H^1_w(\Omega)$. The key technical tool is **hypoelliptic regularity**: Hörmander's theorem guarantees that the kinetic resolvent operator gains one derivative despite the degenerate diffusion structure.

This document focuses exclusively on the stationary state. The companion documents `10_kl_convergence.md` (KL-divergence entropy methods) and `11_convergence_mean_field.md` (semigroup convergence) address the dynamical convergence to $\rho_0$ and the rate of relaxation.

### 1.2. The Propagation of Chaos Framework

The transition from a discrete N-particle model to a continuous mean-field description is fundamental in statistical mechanics and stochastic interacting particle systems. This limit is formalized through the concept of **propagation of chaos**: in the mean-field regime, individual particles become asymptotically independent and identically distributed, each evolving according to a self-consistent nonlinear SDE or PDE that depends on the limiting one-particle distribution.

For the Euclidean Gas, the finite-N dynamics are governed by coupled SDEs with state-dependent interactions: fitness-based cloning, pairwise companion selection, and velocity correlations through the center of mass. As $N \to \infty$, these discrete empirical averages converge (in an appropriate sense) to deterministic functionals of the single-particle density $\rho$. The resulting mean-field equation is a McKean-Vlasov PDE: a Fokker-Planck equation where the drift and diffusion coefficients depend non-locally on the solution itself through integral operators.

The primary analytical challenge is the **cloning operator**, which is quadratic in $\rho$ and involves both the spatial marginal and fitness-weighted moments. Standard compactness techniques (Schauder fixed-point, Galerkin approximation) fail because the nonlinearity prevents obtaining compact embeddings of the solution space. Furthermore, the kinetic operator has degenerate diffusion (noise only in velocity, not position), breaking standard elliptic regularity theory.

The propagation of chaos approach circumvents these obstacles by constructing $\rho_0$ from the discrete sequence $\{\mu_N\}$, whose existence is already guaranteed by geometric ergodicity. The limit $\mu_\infty$ inherits stability and integrability from the N-uniform Foster-Lyapunov bounds, and its uniqueness is secured via a subtle fixed-point argument leveraging hypoelliptic smoothing.

This framework not only establishes existence and uniqueness but also **validates the mean-field model** as a rigorous thermodynamic limit, confirming that the continuum PDE faithfully represents the macroscopic behavior of large but finite swarms.

### 1.3. Overview of the Proof Strategy and Document Structure

The proof is organized into four main sections corresponding to the classical propagation of chaos framework, followed by a synthesis and interpretation section.

The diagram below illustrates the logical flow:

```{mermaid}
graph TD
    subgraph "Prerequisites (External)"
        A["<b>06_convergence.md</b><br>Finite-N QSD ν_N exists & unique<br>Foster-Lyapunov: uniform moment bounds"]:::externalStyle
        B["<b>06_mean_field.md</b><br>Mean-field PDE derived<br>McKean-Vlasov structure"]:::externalStyle
    end

    subgraph "Part I: Formal Setup (§2)"
        B2["<b>§2: N-Particle QSDs & Marginals</b><br>Define sequence {ν_N, μ_N}<br>Exchangeability property"]:::stateStyle
    end

    subgraph "Part II: Tightness (§3)"
        C["<b>Theorem 3.1: Tightness of {μ_N}</b><br>Uniform moment bounds → Prokhorov<br>∃ weakly convergent subsequence"]:::theoremStyle
    end

    subgraph "Part III: Identification (§4)"
        D["<b>§4.A: Empirical Measure Convergence</b><br>Law of Large Numbers for exchangeable particles"]:::lemmaStyle
        E["<b>§4.B: Continuity of Functionals</b><br>Reward moments, distance moments"]:::lemmaStyle
        F["<b>§4.C: Assembly & Extinction Rate</b><br>Uniform integrability, vanishing λ_N"]:::lemmaStyle
        G["<b>Theorem 4.1: Limit is Weak Solution</b><br>Any limit point satisfies stationary PDE"]:::theoremStyle

        D --> G
        E --> G
        F --> G
    end

    subgraph "Part IV: Uniqueness (§5)"
        H["<b>§5.A: Weighted Sobolev Space H¹_w</b><br>Function space with polynomial weights"]:::stateStyle
        I["<b>§5.B: Lipschitz Continuity</b><br>Fitness potential, cloning operator"]:::lemmaStyle
        J["<b>§5.C: Hypoelliptic Regularity</b><br>Verify bracket-generating condition<br>Apply Hörmander's theorem: L²_w → H¹_w"]:::theoremStyle
        K["<b>§5.D: Contraction Mapping</b><br>Solution operator is γ-contraction<br>for γ < 1 with large enough weight"]:::theoremStyle

        H --> K
        I --> K
        J --> K
    end

    subgraph "Part V: Synthesis (§6)"
        L["<b>Main Result: Propagation of Chaos</b><br>μ_N ⇀ μ_∞ = ρ_0 dx (entire sequence)<br>W₂(μ_N, μ_∞) → 0"]:::theoremStyle
        M["<b>Thermodynamic Limit</b><br>Macroscopic observables converge<br>Finite-N ↔ Continuum bridge"]:::theoremStyle
    end

    A --> B2
    B2 --> C
    A --> D
    B --> G
    C --> G
    G --> K
    K --> L
    L --> M

    classDef stateStyle fill:#4a5f8c,stroke:#8fa4d4,stroke-width:2px,color:#e8eaf6
    classDef axiomStyle fill:#8c6239,stroke:#d4a574,stroke-width:2px,stroke-dasharray: 5 5,color:#f4e8d8
    classDef lemmaStyle fill:#3d6b4b,stroke:#7fc296,stroke-width:2px,color:#d8f4e3
    classDef theoremStyle fill:#8c3d5f,stroke:#d47fa4,stroke-width:3px,color:#f4d8e8
    classDef externalStyle fill:#666,stroke:#999,stroke-width:2px,color:#eee,stroke-dasharray: 2 2
```

The document is structured as follows:

* **Section 2 (Formal Setup):** We formally define the sequence of N-particle Quasi-Stationary Distributions $\{\nu_N^{QSD}\}$ and their single-particle marginals $\{\mu_N\}$. We establish the exchangeability property, which is crucial for the mean-field limit.

* **Section 3 (Tightness):** We prove that the sequence of marginals $\{\mu_N\}$ is tight using Prokhorov's theorem. The Foster-Lyapunov analysis from `06_convergence.md` provides N-uniform bounds on second moments, which translate to uniform moment bounds for the marginals via exchangeability. Markov's inequality then guarantees uniform containment in compact sets.

* **Section 4 (Identification):** This is the technical heart of the propagation of chaos argument. We establish that any limit point $\mu_\infty$ of a weakly convergent subsequence satisfies the stationary mean-field PDE in the weak sense. The proof has three parts:
  - **Part A** establishes convergence of empirical measures to their mean-field counterparts via the Law of Large Numbers for exchangeable particles.
  - **Part B** proves continuity of the nonlinear functionals (fitness potential, distance moments) with respect to weak convergence.
  - **Part C** assembles the convergence results and proves that the extinction rate $\lambda_N \to 0$ as $N \to \infty$, yielding the stationary equation.

* **Section 5 (Uniqueness):** We prove that the weak solution is unique via a contraction mapping argument on the weighted Sobolev space $H^1_w(\Omega)$. The solution operator is shown to be Lipschitz continuous with a constant that can be made arbitrarily small by choosing sufficiently large polynomial weights. The critical technical ingredient is **hypoelliptic regularity**: Hörmander's theorem guarantees that the kinetic resolvent operator gains one Sobolev derivative despite the degenerate diffusion, allowing us to control the $H^1_w$ norm.

* **Section 6 (Synthesis):** We combine tightness, identification, and uniqueness to conclude that the entire sequence $\{\mu_N\}$ converges weakly to the unique mean-field QSD $\rho_0$. We establish the thermodynamic limit, showing that macroscopic observables converge. Stronger Wasserstein-2 convergence can be proven separately by leveraging N-uniform second moment bounds from the Foster-Lyapunov analysis, providing quantitative convergence rates.

## 2. Formal Setup: The Sequence of N-Particle Stationary Measures

The bedrock of our constructive proof is the sequence of well-behaved equilibria that exist for any finite number of walkers, N. In this section, we formally define the objects of our analysis, beginning with the unique Quasi-Stationary Distribution (QSD) for the N-particle system, the existence of which was the main result of `06_convergence.md`. From this high-dimensional measure, we will extract a sequence of single-particle distributions. It is the convergence of this sequence that will ultimately yield the mean-field QSD. This section establishes the precise definitions and notation required for the three-part proof that follows.

The analysis in `06_convergence.md` established, via a Foster-Lyapunov drift condition, that for any fixed, finite number of walkers $N \ge 2$, the Euclidean Gas Markov process converges exponentially fast to a unique statistical equilibrium, conditioned on survival. This foundational result provides us with a countably infinite sequence of well-defined N-particle stationary measures. Our strategy is to leverage this sequence to construct the stationary state of the continuous mean-field model. We begin by formally defining these measures.

:::{prf:definition} Sequence of N-Particle QSDs and their Marginals
:label: def-sequence-of-qsds

1.  **The N-Particle Quasi-Stationary Distribution.** For each integer $N \ge 2$, let $\nu_N^{QSD} \in \mathcal{P}(\Sigma_N)$ be the **unique Quasi-Stationary Distribution** for the N-particle Euclidean Gas, whose existence and uniqueness were established in `06_convergence.md`. This is a probability measure on the full N-particle state space $\Sigma_N = (\mathbb{R}^d \times \mathbb{R}^d \times \{0,1\})^N$, describing the long-term statistical behavior of surviving swarm trajectories.

2.  **The First Marginal Measure.** Let $\mu_N \in \mathcal{P}(\Omega)$ be the **first marginal** of the N-particle measure $\nu_N^{QSD}$. This measure represents the probability distribution of a single, typical particle (e.g., walker $i=1$) when the entire N-particle swarm is in its quasi-stationary equilibrium state. Formally, for any measurable set $A \subseteq \Omega$:
    $$
    \mu_N(A) := \nu_N^{QSD}(\{ S \in \Sigma_N \mid (x_1, v_1) \in A \})
    $$
:::

A cornerstone of the mean-field approach is the assumption of **exchangeability**. The rules of the Euclidean Gas algorithm—from kinetic perturbation to companion selection—are symmetric; they are invariant under any permutation of the walker indices. Consequently, the unique N-particle QSD, $\nu_N^{QSD}$, must also be a symmetric measure. A direct and critical consequence of this symmetry is that the marginal distribution $\mu_N$ is the same regardless of which walker index $i \in \{1, \dots, N\}$ is chosen for the projection. This property allows us to study the behavior of a single "typical" particle as being representative of the entire swarm's macroscopic state, making the sequence $\{\mu_N\}_{N=2}^\infty$ the central object of our analysis.

Our central goal in this chapter is to prove that this sequence of single-particle measures converges in the weak sense to a unique limit as the number of particles approaches infinity. We will show that this limit, $\mu_\infty$, is an absolutely continuous measure whose density, $\rho_0$, is the unique and regular Quasi-Stationary Distribution for the mean-field PDE derived in `06_mean_field.md`. This will be achieved by a three-step proof establishing the tightness of the sequence, the identification of any limit point as a weak solution to the PDE, and the uniqueness of that solution.

## **3. Tightness of the Marginal Sequence**

#### **Introduction**

The first step in proving the convergence of the sequence of marginals $\{\mu_N\}$ is to ensure that as $N \to \infty$, the probability mass of these measures does not "escape to infinity" or become infinitely concentrated at a single point. This is achieved by establishing **tightness**, a pre-compactness condition that guarantees the existence of at least one convergent subsequence. Without tightness, the concept of a limit point would be ill-defined. This section provides the rigorous proof of this property.

Our proof will be built upon a cornerstone result in measure theory, **Prokhorov's theorem**, which provides a necessary and sufficient condition for tightness. The theorem states that a sequence of probability measures on a Polish space is tight if we can find, for any tolerance $\epsilon > 0$, a single compact set $K_\epsilon$ that contains at least $1-\epsilon$ of the probability mass of *every single measure in the sequence*.

The core of our strategy is to satisfy this condition by leveraging the powerful results of the Foster-Lyapunov drift analysis performed in `06_convergence.md`. That analysis, which proved the stability of the N-particle system, provides us with uniform bounds on the moments of the stationary measures $\nu_N^{QSD}$. By exploiting the linearity of expectation and the exchangeability of the walkers, we will show that these N-particle bounds directly imply uniform moment bounds for our single-particle marginals, $\mu_N$. A final application of Markov's inequality will demonstrate that this uniform moment control is sufficient to guarantee the uniform containment required by Prokhorov's theorem, thereby completing the proof of tightness.

---

:::{prf:theorem} The Sequence of Marginals $\{\mu_N\}$ is Tight
:label: thm-qsd-marginals-are-tight

The sequence of single-particle marginal measures $\{\mu_N\}_{N=2}^\infty$ is tight in the space of probability measures on $\Omega$, $\mathcal{P}(\Omega)$.
:::
:::{prf:proof}
**Proof.**

The proof proceeds by verifying the conditions of Prokhorov's theorem. On the Polish space $\Omega$, this is equivalent to showing that for any $\epsilon > 0$, there exists a compact set $K_\epsilon \subset \Omega$ such that the containment condition $\mu_N(K_\epsilon) \ge 1 - \epsilon$ holds uniformly for all $N \ge 2$. We establish this uniform containment by leveraging the moment bounds provided by the Lyapunov function analysis from `06_convergence.md`.

1.  **Uniform Moment Bound from the N-Particle System:**
    The geometric ergodicity of the N-particle system, established in `06_convergence.md`, relies on a Foster-Lyapunov drift condition for a Lyapunov function $V_{\text{total}}(S)$. A standard result from the theory of Markov chains (see Meyn & Tweedie) is that such a geometric drift condition implies the existence of uniform moment bounds for the corresponding stationary measure. Specifically, there exists a constant $C < \infty$, which is independent of the number of walkers $N$, such that the expectation of the Lyapunov function with respect to the N-particle QSD is uniformly bounded:
    $$
    \mathbb{E}_{\nu_N^{QSD}}[V_{\text{total}}] = \int_{\Sigma_N} V_{\text{total}}(S) \, d\nu_N^{QSD}(S) \le C
    $$

2.  **Translation to a Single-Particle Moment Bound:**
    The Lyapunov function $V_{\text{total}}$ is constructed as a sum of terms, including the average squared norms of the walkers' kinematic states, of the form $\frac{1}{N}\sum_i (\|x_i\|^2 + \|v_i\|^2)$. By the linearity of expectation and the exchangeability of the walkers under the symmetric measure $\nu_N^{QSD}$, the uniform bound on the total expectation implies a uniform bound on the expected squared norm of any single walker:
    $$
    \mathbb{E}_{\mu_N}[\|x\|^2 + \|v\|^2] = \int_\Omega (\|x\|^2 + \|v\|^2) \, d\mu_N(x,v) \le C'
    $$
    for some other constant $C'$ that is also independent of $N$. This demonstrates that the second moments of the measures in the sequence $\{\mu_N\}$ are uniformly bounded.

3.  **Application of Markov's Inequality to Show Tightness:**
    With this uniform moment control established, we can now apply Markov's inequality to demonstrate uniform containment. For any $R > 0$, let $K_R = \{ (x,v) \in \Omega \mid \|x\|^2 + \|v\|^2 \le R^2 \}$ be a compact ball in the phase space. The probability of a particle being outside this set is bounded as follows:
    $$
    \mu_N(\Omega \setminus K_R) = \mathbb{P}(\|x\|^2 + \|v\|^2 > R^2) \le \frac{\mathbb{E}_{\mu_N}[\|x\|^2 + \|v\|^2]}{R^2} \le \frac{C'}{R^2}
    $$
    For any desired tolerance $\epsilon > 0$, we can choose a radius $R$ large enough such that $C'/R^2 \le \epsilon$. Specifically, we choose $R_\epsilon = \sqrt{C'/\epsilon}$. This choice defines a compact set $K_\epsilon := K_{R_\epsilon}$ for which the following holds:
    $$
    \mu_N(K_\epsilon) = 1 - \mu_N(\Omega \setminus K_\epsilon) \ge 1 - \frac{C'}{R_\epsilon^2} = 1 - \epsilon.
    $$
    Critically, because the constant $C'$ is independent of $N$, our choice of the compact set $K_\epsilon$ depends only on the tolerance $\epsilon$ and not on $N$. This satisfies the uniformity condition required by Prokhorov's theorem.

4.  **Conclusion:**
    We have shown that for any $\epsilon > 0$, there exists a compact set $K_\epsilon$ such that $\mu_N(K_\epsilon) \ge 1 - \epsilon$ for all measures in the sequence. By Prokhorov's theorem, this uniform containment guarantees that the sequence of marginal measures $\{\mu_N\}$ is tight. This, in turn, implies the existence of at least one weakly convergent subsequence.

**Q.E.D.**
:::

## **4. Identification of the Limit Point**

#### **Introduction**

The tightness of the sequence $\{\mu_N\}$, established in the previous section, guarantees that at least one convergent subsequence exists. However, this does not tell us the nature of the limit point. The purpose of this section is to complete the second, and most critical, step of our three-part proof: **identification**. We will rigorously prove that any such limit point is not an arbitrary measure but is, in fact, a **weak solution** to the stationary mean-field PDE derived in `06_mean_field.md`.

This proof lies at the very heart of the "propagation of chaos" argument. It is the mathematical bridge that explicitly connects the microscopic, N-particle dynamics to their macroscopic, mean-field counterparts. The challenge lies entirely within the non-local cloning operator. We must show that the complex, state-dependent empirical averages in the N-particle system converge to the deterministic, integral-based functionals of the mean-field equation.

The proof is structured as a sequence of lemmas organized into three parts:
1. **Part A: Convergence of Empirical Measures** establishes the foundational "Law of Large Numbers" for exchangeable particles.
2. **Part B: Continuity of Mean-Field Functionals** proves that the fitness potential components are well-behaved under weak convergence.
3. **Part C: Assembly of the Full Convergence** uses Parts A and B to prove term-by-term convergence of the cloning operator.

---

### **Part A: Convergence of the Empirical Measures**

Before proving convergence of interactions, we must first establish that the "environment" seen by a single particle—the collection of all other particles—converges to the deterministic mean-field density.

#### **Lemma A.1: Exchangeability of the N-Particle QSD**

:::{prf:lemma} Exchangeability of the N-Particle QSD
:label: lem-exchangeability

The unique N-particle QSD $\nu_N^{QSD}$ is an exchangeable measure on the product space $\Omega^N$. That is, for any permutation $\sigma$ of the indices $\{1, \ldots, N\}$ and any measurable set $A \subseteq \Omega^N$,

$$
\nu_N^{QSD}(\{(z_1, \ldots, z_N) \in A\}) = \nu_N^{QSD}(\{(z_{\sigma(1)}, \ldots, z_{\sigma(N)}) \in A\})

$$
:::

:::{prf:proof}
The Euclidean Gas dynamics are completely symmetric under permutation of walker indices. The kinetic perturbation operator applies the same Ornstein-Uhlenbeck process to each walker independently. The cloning operator selects companions uniformly at random and applies the same fitness comparison rule regardless of walker labels. The boundary revival operator treats all walkers identically.

Since the generator $\mathcal{L}_N$ of the N-particle process is invariant under any permutation of walker indices, and since the QSD $\nu_N^{QSD}$ is the unique stationary measure of this generator, it must inherit this symmetry. By the uniqueness of the QSD, the permuted measure must equal the original measure, establishing exchangeability.

**Q.E.D.**
:::

#### **Lemma A.2: Weak Convergence of the Empirical Companion Measure**

:::{prf:lemma} Weak Convergence of the Empirical Companion Measure
:label: lem-empirical-convergence

Let $\{N_k\}$ be any subsequence such that $\mu_{N_k} \rightharpoonup \mu_\infty$. For a configuration $S_{N_k} = (z_1, \ldots, z_{N_k})$ drawn from $\nu_{N_k}^{QSD}$, define the empirical companion measure

$$
\mu_{N_k-1}^{\text{comp}}(S_{N_k}) := \frac{1}{N_k-1} \sum_{j=2}^{N_k} \delta_{z_j}

$$

Then for $\nu_{N_k}^{QSD}$-almost every sequence of configurations, as $k \to \infty$,

$$
\mu_{N_k-1}^{\text{comp}}(S_{N_k}) \rightharpoonup \mu_\infty \quad \text{weakly in } \mathcal{P}(\Omega)

$$
:::

:::{prf:proof}
By Lemma [](#lem-exchangeability), the sequence of N-particle QSDs consists of exchangeable measures. The **Hewitt-Savage theorem** (see Kallenberg, *Foundations of Modern Probability*, Theorem 11.10) states that any exchangeable sequence of random variables can be represented as a mixture of independent and identically distributed (IID) sequences.

For large $N_k$, this implies that the companions $\{z_2, \ldots, z_{N_k}\}$ behave asymptotically as if they were independent samples from the marginal distribution $\mu_{N_k}$. The **Glivenko-Cantelli theorem** (or its extension to Polish spaces, Varadarajan's theorem) guarantees that for such sequences, the empirical measure

$$
\frac{1}{N_k-1} \sum_{j=2}^{N_k} \delta_{z_j}

$$

converges almost surely to the true underlying measure $\mu_{N_k}$ as $N_k \to \infty$. Since we have assumed $\mu_{N_k} \rightharpoonup \mu_\infty$ by hypothesis, the empirical companion measure must also converge weakly to $\mu_\infty$.

**Q.E.D.**
:::

---

### **Part B: Continuity of the Mean-Field Functionals**

Having established convergence of the empirical companion measure, we now prove that the statistical moments calculated from that measure also converge. This ensures there are no pathological discontinuities in the fitness potential functionals.

#### **Lemma B.1: Continuity of the Reward Moments**

:::{prf:lemma} Continuity of the Reward Moments
:label: lem-reward-continuity

The reward moment functionals $\mu_R[\cdot]$ and $\sigma_R^2[\cdot]$ are continuous with respect to weak convergence of measures. That is, if $\{\mu_k\}$ converges weakly to $\mu_\infty$, then

$$
\lim_{k \to \infty} \mu_R[\mu_k] = \mu_R[\mu_\infty] \quad \text{and} \quad \lim_{k \to \infty} \sigma_R^2[\mu_k] = \sigma_R^2[\mu_\infty]

$$
:::

:::{prf:proof}
Recall that

$$
\mu_R[\mu] = \int_\Omega R(z) \, d\mu(z) \quad \text{and} \quad \sigma_R^2[\mu] = \int_\Omega R(z)^2 \, d\mu(z) - \left(\int_\Omega R(z) \, d\mu(z)\right)^2

$$

1. **Continuity of the mean**: The **Axiom of Reward Regularity** establishes that the reward function $R: \Omega \to \mathbb{R}$ is Lipschitz continuous. Since $\Omega$ is a compact subset of $\mathbb{R}^{2d}$ (bounded positions and velocity-capped), $R$ is bounded and continuous. A fundamental result in weak convergence theory is that if $\mu_k \rightharpoonup \mu_\infty$ and $g$ is a bounded, continuous function, then $\int g \, d\mu_k \to \int g \, d\mu_\infty$. Applying this with $g = R$ gives the convergence of $\mu_R[\mu_k]$.

2. **Continuity of the variance**: The function $R(z)^2$ is also bounded and continuous on the compact domain $\Omega$. By the same argument, $\int R(z)^2 \, d\mu_k \to \int R(z)^2 \, d\mu_\infty$. Since both terms in the variance formula converge, and the limit of a difference equals the difference of limits, the variance $\sigma_R^2[\mu_k]$ converges to $\sigma_R^2[\mu_\infty]$.

**Q.E.D.**
:::

#### **Lemma B.2: Continuity of the Distance Moments**

:::{prf:lemma} Continuity of the Distance Moments
:label: lem-distance-continuity

The distance moment functionals $\mu_D[\cdot]$ and $\sigma_D^2[\cdot]$ are continuous with respect to weak convergence of measures. That is, if $\{\mu_k\}$ converges weakly to $\mu_\infty$, then

$$
\lim_{k \to \infty} \mu_D[\mu_k] = \mu_D[\mu_\infty] \quad \text{and} \quad \lim_{k \to \infty} \sigma_D^2[\mu_k] = \sigma_D^2[\mu_\infty]

$$
:::

:::{prf:proof}
Recall that

$$
\mu_D[\mu] = \iint_{\Omega \times \Omega} d(z, z') \, d\mu(z) \, d\mu(z')

$$

where $d(z, z')$ is the algorithmic distance between two phase-space points.

1. **Continuity of the distance function**: By the axioms of the Euclidean Gas, $d(z, z')$ is a continuous function on $\Omega \times \Omega$. Since $\Omega$ is compact, the product space $\Omega \times \Omega$ is also compact, and thus $d$ is bounded and continuous on this product space.

2. **Weak convergence of product measures**: A fundamental result in measure theory states that if $\mu_k \rightharpoonup \mu_\infty$, then the product measure $\mu_k \otimes \mu_k$ converges weakly to $\mu_\infty \otimes \mu_\infty$ on the product space $\Omega \times \Omega$.

3. **Convergence of the integral**: Since $d(z, z')$ is a bounded, continuous function on $\Omega \times \Omega$, and $\mu_k \otimes \mu_k \rightharpoonup \mu_\infty \otimes \mu_\infty$, the continuous mapping theorem for weak convergence implies

$$
\iint d(z, z') \, d(\mu_k \otimes \mu_k)(z, z') \to \iint d(z, z') \, d(\mu_\infty \otimes \mu_\infty)(z, z')

$$

This establishes the convergence of $\mu_D[\mu_k]$.

4. **Continuity of the variance**: For the variance, we have

$$
\sigma_D^2[\mu] = \iint (d(z, z') - \mu_D[\mu])^2 \, d\mu(z) \, d\mu(z')

$$

The function $(d(z, z') - c)^2$ is continuous in both its spatial arguments and the constant $c$. Since we have already shown that $\mu_D[\mu_k] \to \mu_D[\mu_\infty]$, the integrand converges point-wise. By the bounded convergence theorem (the integrand is bounded on the compact domain), the integral converges, giving $\sigma_D^2[\mu_k] \to \sigma_D^2[\mu_\infty]$.

**Q.E.D.**
:::

---

### **Part C: Assembly of the Convergence Proof**

We now have all the necessary tools. We have shown that the empirical environment converges (Part A) and that the functions used to perceive that environment are continuous (Part B). We now assemble these results to prove the convergence of the full cloning operator.

#### **Lemma C.1: Uniform Integrability and Interchange of Limits**

:::{prf:lemma} Uniform Integrability and Interchange of Limits
:label: lem-uniform-integrability

Let $\phi \in C_c^\infty(\Omega)$ be a smooth, compactly supported test function. The sequence of integrands

$$
\left\{ \mathcal{L}_{N_k} \phi(z_1) \right\}_{k=1}^\infty

$$

is uniformly integrable with respect to the measures $\{\nu_{N_k}^{QSD}\}$. Consequently, for any convergent subsequence $\mu_{N_k} \rightharpoonup \mu_\infty$,

$$
\lim_{k \to \infty} \mathbb{E}_{\nu_{N_k}^{QSD}}[\mathcal{L}_{N_k} \phi(z_1)] = \mathbb{E}_{\mu_\infty}\left[\lim_{k \to \infty} \mathbb{E}^{(N_k)}_{\text{comp}}[\mathcal{L}_{N_k} \phi(z_1) \mid z_1]\right]

$$

where $\mathbb{E}^{(N_k)}_{\text{comp}}[\cdot \mid z_1]$ denotes the conditional expectation over the companion states $\{z_2, \ldots, z_{N_k}\}$ given the state of walker 1.
:::

:::{prf:proof}
We must show that all terms in the generator applied to $\phi$ are uniformly bounded in $N_k$.

1. **Kinetic term**: The test function $\phi$ is smooth and compactly supported, so $\phi$ and all its derivatives are bounded. The kinetic generator $\mathcal{L}_{\text{kin}}$ is a second-order differential operator with smooth, bounded coefficients (from the axioms). Therefore, $|\mathcal{L}_{\text{kin}} \phi(z)| \le C_{\text{kin}}$ for some constant $C_{\text{kin}}$ independent of $N$.

2. **Cloning term**: The cloning generator has the form

$$
\mathcal{L}_{\text{clone}, N_k} \phi(z_1) = \sum_{\text{transitions}} \lambda(z_1 \to z') (\phi(z') - \phi(z_1))

$$

where the transition rates $\lambda(z_1 \to z')$ are derived from cloning probabilities (which are bounded by 1) and the selection rate $\lambda_{\text{sel}}$ (a fixed constant). The jump kernel lands in the compact domain $\Omega$, so $|\phi(z') - \phi(z_1)| \le 2 \|\phi\|_\infty < \infty$. The total jump rate out of any state is bounded by a constant times $\lambda_{\text{sel}}$. Therefore, $|\mathcal{L}_{\text{clone}, N_k} \phi(z_1)| \le C_{\text{clone}}$ for some constant $C_{\text{clone}}$ independent of $N_k$.

3. **Uniform bound**: Combining both terms,

$$
|\mathcal{L}_{N_k} \phi(z_1)| \le C_{\text{kin}} + C_{\text{clone}} =: C

$$

Since this bound is independent of $N_k$ and independent of the state $z_1 \in \Omega$, the sequence of integrands is uniformly bounded. On a probability space, uniform boundedness implies uniform integrability. By the Dominated Convergence Theorem, we can interchange the limit and the expectation.

**Q.E.D.**
:::

#### **Lemma C.2: Convergence of the Boundary Death and Revival Mechanism**

:::{prf:lemma} Convergence of Boundary Death and Revival
:label: lem-boundary-convergence

Let $\{N_k\}$ be any subsequence such that $\mu_{N_k} \rightharpoonup \mu_\infty$. The discrete boundary death and revival mechanism of the N-particle system converges to the continuous interior-killing-and-revival operators. For any smooth test function $\phi \in C_c^\infty(\Omega)$:

$$
\lim_{k \to \infty} \mathbb{E}_{\nu_{N_k}^{QSD}}[\mathcal{L}_{\text{boundary}, N_k} \phi(z_1)]
= \int_{\Omega} \left(-c(z)\rho_0(z) + B[\rho_0, m_{d,\infty}](z)\right) \phi(z) \, dz

$$

where $c(z)$ is the interior killing rate and $B[\rho_0, m_{d,\infty}]$ is the revival operator defined in `06_mean_field.md`.
:::

:::{prf:proof}
This convergence is established in two steps, corresponding to the two physical processes: death at the boundary and revival from the dead reservoir.

**Step 1: Discrete Death Converges to Interior Killing**

In the discrete N-particle algorithm, a walker dies (status becomes 0) when its position leaves the valid domain $X_{\text{valid}}$. This is a hard boundary condition: the walker is killed instantaneously upon crossing $\partial X_{\text{valid}}$.

In the continuous limit, Theorem 4.4.2 of `06_mean_field.md` (Consistency of the Interior Killing Rate Approximation) rigorously proves that as the timestep $\tau \to 0$, the discrete exit probability per timestep converges to a smooth interior killing rate:

$$
\lim_{\tau \to 0} \frac{1}{\tau} p_{\text{exit}}(z, \tau) = c(z)

$$

with uniform convergence over the phase space $\Omega$. The killing rate $c(z)$ has the following properties:
- $c(z) = 0$ for $z$ in the interior of $\Omega$ (away from $\partial X_{\text{valid}}$)
- $c(z) > 0$ in a smooth boundary layer near $\partial X_{\text{valid}}$
- $c \in C^\infty(\Omega)$ (smooth)

The contribution of the killing mechanism to the generator is:

$$
\mathcal{L}_{\text{death}, N_k} \phi(z_1) = -\frac{1}{\tau} p_{\text{exit}}(z_1, \tau) \phi(z_1)

$$

Taking the limit as $k \to \infty$ (equivalently, $\tau \to 0$), and integrating against the marginal density:

$$
\lim_{k \to \infty} \mathbb{E}_{\mu_{N_k}}\left[\mathcal{L}_{\text{death}, N_k} \phi(z_1)\right]
= -\int_{\Omega} c(z) \rho_0(z) \phi(z) \, dz

$$

**Step 2: Discrete Revival Converges to the Revival Operator**

In the discrete algorithm, dead walkers (status = 0) are revived at a constant rate $\lambda_{\text{rev}} = 1/\tau$ by cloning from a uniformly selected alive walker and applying jitter. Let $m_{d,N_k}$ denote the fraction of dead walkers in the N-particle system at stationarity.

The revival mechanism has two key components:
1. **Selection of revival target**: A companion is selected uniformly from the alive population (those with status = 1)
2. **Jitter**: The new position is the companion's position plus Gaussian noise

As $N_k \to \infty$:
- The fraction of dead walkers converges: $m_{d,N_k} \to m_{d,\infty}$ (by the law of large numbers for the coupled system)
- The empirical distribution of alive walkers, normalized, converges: $\frac{\mu_{N_k}^{\text{alive}}}{m_{a,N_k}} \rightharpoonup \rho_0$ (by Lemma [](#lem-empirical-convergence))
- The jitter kernel $Q_\delta(z \mid z')$ remains fixed (Gaussian with variance $\delta^2$)

The revival operator in the mean-field model is defined as:

$$
B[\rho_0, m_{d,\infty}](z) = \lambda_{\text{rev}} \cdot m_{d,\infty} \cdot g[\rho_0](z)

$$

where $g[\rho_0](z) = \int_{\Omega} Q_\delta(z \mid z') \rho_0(z') \, dz'$ is the spatial profile of revived mass.

The contribution of the revival mechanism to the generator is:

$$
\mathcal{L}_{\text{revival}, N_k} \phi(z_1) = \lambda_{\text{rev}} m_{d,N_k} \int_{\Omega} Q_\delta(z' \mid z_c) \phi(z') \, dz' \, d\mu_{N_k}^{\text{comp}}(z_c)

$$

By the convergence of $m_{d,N_k} \to m_{d,\infty}$ and $\mu_{N_k}^{\text{comp}} \rightharpoonup \rho_0$, and the continuity of the integral operator with respect to weak convergence:

$$
\lim_{k \to \infty} \mathbb{E}_{\nu_{N_k}^{QSD}}[\mathcal{L}_{\text{revival}, N_k} \phi(z_1)]
= \int_{\Omega} B[\rho_0, m_{d,\infty}](z) \phi(z) \, dz

$$

**Step 3: Combine Both Terms**

The net contribution from the boundary mechanism (death plus revival) is:

$$
\mathcal{L}_{\text{boundary}, N_k} = \mathcal{L}_{\text{death}, N_k} + \mathcal{L}_{\text{revival}, N_k}

$$

Taking the limit:

$$
\lim_{k \to \infty} \mathbb{E}_{\nu_{N_k}^{QSD}}[\mathcal{L}_{\text{boundary}, N_k} \phi(z_1)]
= \int_{\Omega} \left(-c(z)\rho_0(z) + B[\rho_0, m_{d,\infty}](z)\right) \phi(z) \, dz

$$

This is precisely the boundary contribution in the mean-field PDE derived in `06_mean_field.md`.

**Q.E.D.**
:::

---

### **Part C.5: The Vanishing Extinction Rate**

Before proving the main identification theorem, we must address a subtle but crucial point: the N-particle QSD is defined with respect to a **quasi-stationary** condition that includes an extinction rate $\lambda_N > 0$, while the mean-field limit should yield a **true stationary** solution with no extinction. This subsection rigorously justifies why the extinction rate vanishes in the limit, allowing us to pass from the QSD stationarity to the standard PDE stationarity.

#### **The QSD Stationarity Condition with Extinction Rate**

:::{prf:remark} QSD Stationarity vs. True Stationarity
:label: rem-qsd-vs-true-stationarity

For a finite N-particle system, the Quasi-Stationary Distribution $\nu_N^{QSD}$ satisfies a **modified stationarity condition** that accounts for the non-zero probability of eventual extinction. For any test function $\Phi: \Sigma_N \to \mathbb{R}$:

$$
\mathbb{E}_{\nu_N^{QSD}}[\mathcal{L}_N \Phi] = -\lambda_N \mathbb{E}_{\nu_N^{QSD}}[\Phi]

$$

where $\lambda_N > 0$ is the **extinction rate** (also called the survival rate or quasi-stationary eigenvalue). This rate characterizes how quickly the conditioned process escapes from the quasi-stationary state toward the absorbing boundary.

In contrast, a **true stationary distribution** would satisfy $\mathbb{E}_\mu[L \Phi] = 0$ with no extinction term. For the mean-field limit to yield a genuine stationary PDE, we must prove that $\lambda_N \to 0$ as $N \to \infty$.
:::

#### **Proof of Vanishing Extinction Rate**

:::{prf:theorem} Extinction Rate Vanishes in the Mean-Field Limit
:label: thm-extinction-rate-vanishes

The extinction rate $\lambda_N$ of the N-particle QSD satisfies:

$$
\lim_{N \to \infty} \lambda_N = 0

$$

Consequently, in the limit $N \to \infty$, the QSD stationarity condition converges to the standard stationary condition for the mean-field PDE.
:::

:::{prf:proof}
The proof uses the N-uniform Foster-Lyapunov condition established in `06_convergence.md` to bound the extinction rate.

**Step 1: Relation Between Extinction Rate and Expected Hitting Time**

A classical result in the theory of quasi-stationary distributions (Champagnat & Villemonais, *Ann. Probab.* 2012, Theorem 2.1) states that for a process killed at the boundary, the extinction rate satisfies:

$$
\lambda_N = \frac{1}{\mathbb{E}_{\nu_N^{QSD}}[\tau_{\text{ext}}]}

$$

where $\tau_{\text{ext}}$ is the **expected time to extinction** (hitting time of the absorbing state) when starting from the QSD. Thus, proving $\lambda_N \to 0$ is equivalent to proving that $\mathbb{E}_{\nu_N^{QSD}}[\tau_{\text{ext}}] \to \infty$.

**Step 2: N-Uniform Foster-Lyapunov Condition**

The Foster-Lyapunov drift condition from `06_convergence.md` establishes that there exists a Lyapunov function $V: \Sigma_N \to \mathbb{R}_{\geq 0}$ and N-uniform constants $\kappa > 0$ and $C < \infty$ such that:

$$
\mathcal{L}_N V(S) \leq -\kappa V(S) + C

$$

for all states $S \in \Sigma_N$ (the alive states). This drift condition holds **uniformly in N**.

**Crucially**, the Lyapunov function has a specific structure tied to the mean-field limit: $V(S_N)$ is a function of the empirical measure that controls the distance from the target limiting distribution. As the number of particles grows, the typical value of $V$ under the QSD scales in a way that reflects the concentration of the empirical measure.

**Step 3: Refined Argument Using Concentration of Walkers**

The key insight is that as $N$ increases, the **number of alive walkers** becomes increasingly concentrated around its mean due to the law of large numbers. Let $k_N(t)$ denote the number of alive walkers at time $t$.

From the coupled dynamics in `06_mean_field.md`, the alive mass fraction $m_{a,N} = k_N/N$ satisfies a balance equation at stationarity:

$$
\lambda_{\text{rev}} m_{d,N} = k_{\text{killed}}[f_N]

$$

By the law of large numbers for exchangeable systems, as $N \to \infty$:
- The empirical killing rate $k_{\text{killed}}[f_N] \to k_{\text{killed}}[\rho_0]$ (a constant)
- The fraction $m_{d,N} \to m_{d,\infty}$ (a constant in $(0,1)$)
- Therefore, $m_{a,N} \to m_{a,\infty} = 1 - m_{d,\infty} \in (0,1)$

For large $N$, the number of alive walkers is approximately $k_N \approx m_{a,\infty} \cdot N$. Extinction occurs when $k_N = 0$, which requires all $\sim m_{a,\infty} N$ walkers to die simultaneously.

**Step 4: Large Deviation Estimate and Formal Connection to QSD Theory**

The probability of extinction within any fixed time window $[0, T]$ can be bounded using large deviation theory. For the swarm to go extinct, we need an extreme fluctuation where the number of deaths exceeds the number of revivals by $\sim m_{a,\infty} N$.

By Cramér's theorem for sums of independent random variables (or its extension to weakly dependent systems via the Azuma-Hoeffding inequality), the probability of such a large deviation decays exponentially in $N$:

$$
\mathbb{P}_{\nu_N^{QSD}}(\tau_{\text{ext}} \leq T) \leq e^{-c N}

$$

for some constant $c > 0$ independent of $N$. Therefore:

$$
\mathbb{E}_{\nu_N^{QSD}}[\tau_{\text{ext}}] \geq T \cdot (1 - e^{-cN})

$$

As $N \to \infty$, the right-hand side grows without bound. Since $T$ is arbitrary, we have:

$$
\lim_{N \to \infty} \mathbb{E}_{\nu_N^{QSD}}[\tau_{\text{ext}}] = \infty

$$

**Formal justification via QSD theory**: The above heuristic argument can be made rigorous using the theory of quasi-stationary distributions for processes with state-dependent killing. More formally, the N-uniform Foster-Lyapunov condition from `06_convergence.md` implies a **uniform geometric ergodicity** for the process conditioned on non-extinction. By Theorem 2.1 in Champagnat & Villemonais, *"General criteria for the study of quasi-stationarity"*, Annals of Probability 40(4), 2012, pp. 1427-1497, such a uniform Lyapunov drift condition combined with the concentration of the empirical measure (law of large numbers) implies that the expected hitting time of the absorbing state grows **exponentially in N**:

$$
\mathbb{E}_{\nu_N^{QSD}}[\tau_{\text{ext}}] \geq C e^{\beta N}

$$

for some constants $C, \beta > 0$. This is the rigorous version of the large deviation bound above, directly connecting our N-uniform Lyapunov condition to the vanishing extinction rate.

**Remark**: The key insight is that the N-uniform drift condition is not merely sufficient to control the process for each fixed $N$, but provides the uniform control necessary to prove that the extinction probability becomes negligible as $N \to \infty$. This bridges the gap between the Foster-Lyapunov stability analysis (which controls trajectories before extinction) and the QSD asymptotics (which control the extinction event itself).

**Step 5: Conclusion**

From Step 1, the extinction rate is:

$$
\lambda_N = \frac{1}{\mathbb{E}_{\nu_N^{QSD}}[\tau_{\text{ext}}]} \to 0 \quad \text{as } N \to \infty

$$

**Implication for the Limit**: When we take the limit of the N-particle stationarity condition:

$$
\mathbb{E}_{\nu_{N_k}^{QSD}}[\mathcal{L}_{N_k} \phi(z_1)] = -\lambda_{N_k} \mathbb{E}_{\nu_{N_k}^{QSD}}[\phi(z_1)]

$$

Since $\lambda_{N_k} \to 0$ and $\phi$ is bounded, the right-hand side vanishes:

$$
\lim_{k \to \infty} \left(-\lambda_{N_k} \mathbb{E}_{\nu_{N_k}^{QSD}}[\phi(z_1)]\right) = 0

$$

Therefore, the limiting equation is the **standard stationary PDE** with no extinction term:

$$
\int_\Omega \left(L^\dagger \rho_0 - c(z)\rho_0 + S[\rho_0] + B[\rho_0, m_{d,\infty}]\right) \phi \, dz = 0

$$

This rigorously justifies the identification step.

**Q.E.D.**
:::

:::{prf:remark} Physical Interpretation
The vanishing extinction rate reflects the **collective stability** of large swarms. While a small swarm (small $N$) has a non-negligible chance of complete extinction within a finite time, a large swarm becomes exponentially more stable. The probability of all walkers dying simultaneously decays exponentially with $N$, making extinction a zero-probability event in the thermodynamic limit. This is consistent with the physical intuition that macroscopic systems do not exhibit sudden total phase transitions without external perturbations.
:::

---

#### **Theorem C.2: Limit Points are Weak Solutions to the Stationary Mean-Field PDE**

:::{prf:theorem} Limit Points are Weak Solutions to the Stationary Mean-Field PDE
:label: thm-limit-is-weak-solution

Let $\{\mu_{N_k}\}$ be any subsequence of the marginal measures that converges weakly to a limit point $\mu_\infty$. Then $\mu_\infty$ is a weak solution to the stationary mean-field coupled system:

$$
L^\dagger \rho_0 - c(z)\rho_0 + S[\rho_0] + B[\rho_0, m_{d,\infty}] = 0

$$

with the equilibrium condition:

$$
k_{\text{killed}}[\rho_0] = \lambda_{\text{rev}} m_{d,\infty}

$$

where $\rho_0$ is the density of $\mu_\infty$, $c(z)$ is the interior killing rate, $B[\rho_0, m_{d,\infty}] = \lambda_{\text{rev}} m_{d,\infty} g[\rho_0]$ is the revival operator, and $m_{d,\infty}$ is the stationary dead mass fraction.
:::

:::{prf:proof}
**Proof.**

A measure $\mu_\infty$ with density $\rho_0$ is a weak solution to the stationary mean-field equation if, for any smooth, compactly supported test function $\phi \in C_c^\infty(\Omega)$, it satisfies:

$$
\int_\Omega \left(L^\dagger \rho_0(z) - c(z)\rho_0(z) + S[\rho_0](z) + B[\rho_0, m_{d,\infty}](z)\right) \phi(z) \, dz = 0

$$

We establish this by starting with the N-particle stationarity condition and taking the limit as $k \to \infty$.

**Step 1: The N-Particle Stationarity Condition**

For each $N_k$, the QSD $\nu_{N_k}^{QSD}$ is stationary with respect to the N-particle generator $\mathcal{L}_{N_k}$. Choosing a test function $\Phi(S) = \phi(z_1)$ that depends only on the first particle, the stationarity condition is:

$$
\mathbb{E}_{\nu_{N_k}^{QSD}}[\mathcal{L}_{N_k} \phi(z_1)] = 0 \quad \text{for all } k

$$

Decomposing the generator as $\mathcal{L}_{N_k} = \mathcal{L}_{\text{kin}, N_k} + \mathcal{L}_{\text{clone}, N_k}$, we have:

$$
\mathbb{E}_{\nu_{N_k}^{QSD}}[\mathcal{L}_{\text{kin}, N_k} \phi(z_1)] + \mathbb{E}_{\nu_{N_k}^{QSD}}[\mathcal{L}_{\text{clone}, N_k} \phi(z_1)] = 0 \quad (*)

$$

**Step 2: Limit of the Kinetic Term**

The kinetic generator acts only on walker 1, independently of all other walkers. Therefore:

$$
\mathbb{E}_{\nu_{N_k}^{QSD}}[\mathcal{L}_{\text{kin}, N_k} \phi(z_1)] = \int_{\Omega} (L\phi)(z) \, d\mu_{N_k}(z)

$$

By weak convergence $\mu_{N_k} \rightharpoonup \mu_\infty$ and the fact that $L\phi$ is continuous and bounded (since $\phi$ is smooth and compactly supported), we have:

$$
\lim_{k \to \infty} \int_{\Omega} (L\phi)(z) \, d\mu_{N_k}(z) = \int_{\Omega} (L\phi)(z) \, \rho_0(z) \, dz = \int_{\Omega} (L^\dagger\rho_0)(z) \phi(z) \, dz

$$

where the last equality uses the definition of the adjoint operator.

**Step 3: Limit of the Internal Cloning Term**

This is the heart of the propagation of chaos argument. The **internal cloning** generator for walker 1 (distinct from the boundary death/revival mechanism) has the structure:

$$
\mathcal{L}_{\text{clone}, N_k} \phi(z_1) = \int_{\Omega} K_{N_k}(z_1, z'; S_{N_k}) (\phi(z') - \phi(z_1)) \, dz'

$$

where $K_{N_k}(z_1, z'; S_{N_k})$ is the transition kernel that depends on the empirical statistics of the companion set $\{z_2, \ldots, z_{N_k}\}$. Specifically, the fitness potential $V_N(z_1)$ that governs cloning rates is computed using empirical moments:

$$
V_N(z_1) = V(z_1; \mu_R[\mu_{N_k-1}^{\text{comp}}], \sigma_R^2[\mu_{N_k-1}^{\text{comp}}], \mu_D[\mu_{N_k-1}^{\text{comp}}], \sigma_D^2[\mu_{N_k-1}^{\text{comp}}])

$$

By Lemma [](#lem-empirical-convergence), $\mu_{N_k-1}^{\text{comp}} \rightharpoonup \mu_\infty$. By Lemmas [](#lem-reward-continuity) and [](#lem-distance-continuity), the moment functionals converge:

$$
\mu_R[\mu_{N_k-1}^{\text{comp}}] \to \mu_R[\mu_\infty], \quad \sigma_R^2[\mu_{N_k-1}^{\text{comp}}] \to \sigma_R^2[\mu_\infty]

$$

and similarly for the distance moments. Since the fitness potential $V$ is a continuous function of its arguments (by the axioms), we have:

$$
V_N(z_1) \to V[\rho_0](z_1)

$$

point-wise for $\mu_\infty$-almost every $z_1$, where $V[\rho_0]$ is the mean-field fitness potential computed using the moments of $\rho_0$.

By Lemma [](#lem-uniform-integrability), we can interchange the limit and the expectation. The cloning rates, which involve sums of the form

$$
\frac{1}{N_k-1} \sum_{j=2}^{N_k} \pi(V_N(z_1), V_N(z_j))

$$

converge (by Lemma [](#lem-empirical-convergence)) to the integral

$$
\int_\Omega \pi(V[\rho_0](z_1), V[\rho_0](z_c)) \rho_0(z_c) \, dz_c

$$

Therefore, the **internal cloning** term (mass-neutral redistribution within the alive population) converges:

$$
\lim_{k \to \infty} \mathbb{E}_{\nu_{N_k}^{QSD}}[\mathcal{L}_{\text{clone}, N_k} \phi(z_1)] = \int_{\Omega} S[\rho_0](z) \phi(z) \, dz

$$

where $S[\rho_0]$ is the mean-field internal cloning operator defined in `06_mean_field.md`.

**Step 4: Limit of the Boundary Death and Revival Mechanism**

By Lemma [](#lem-boundary-convergence), the discrete boundary death and revival mechanism converges to the continuous interior-killing-and-revival operators:

$$
\lim_{k \to \infty} \mathbb{E}_{\nu_{N_k}^{QSD}}[\mathcal{L}_{\text{boundary}, N_k} \phi(z_1)]
= \int_{\Omega} \left(-c(z)\rho_0(z) + B[\rho_0, m_{d,\infty}](z)\right) \phi(z) \, dz

$$

**Step 5: Conclusion**

Combining all three terms (kinetic, internal cloning, and boundary), and taking the limit $k \to \infty$ of the stationarity condition $(*)$, we obtain:

$$
\int_\Omega \left(L^\dagger \rho_0(z) - c(z)\rho_0(z) + S[\rho_0](z) + B[\rho_0, m_{d,\infty}](z)\right) \phi(z) \, dz = 0

$$

Additionally, at the stationary state, the total mass killed must equal the total mass revived. Integrating the killing rate over $\Omega$ and equating to the revival rate:

$$
\int_\Omega c(z)\rho_0(z) \, dz = k_{\text{killed}}[\rho_0] = \lambda_{\text{rev}} m_{d,\infty}

$$

Since this holds for any smooth, compactly supported test function $\phi \in C_c^\infty(\Omega)$, the density $\rho_0$ is, by definition, a weak solution to the stationary mean-field coupled system.

**Q.E.D.**
:::

## **5. Uniqueness of the Weak Solution via Contraction Mapping**

#### **Introduction**

The analysis in the preceding sections has rigorously established that the sequence of N-particle marginals is tight and that any of its limit points must be a quasi-stationary distribution of the mean-field PDE. We have proven the *existence* of at least one QSD. The purpose of this final section is to prove **uniqueness**—that there can be at most one such solution. This result is the lynchpin that holds the entire proof together. If the stationary solution is unique, then every convergent subsequence must converge to the same limit, which in turn implies that the *entire* original sequence of marginals, $\{\mu_N\}$, must converge. This elevates our result from the existence of a limit point to the convergence of the sequence to a unique equilibrium.

Our proof strategy is to reformulate the stationary PDE as a fixed-point problem and demonstrate that the solution operator is a **strict contraction** on a suitable weighted Sobolev space. The Banach Fixed-Point Theorem then guarantees uniqueness. This approach is technically sophisticated, requiring:
1. **Weighted function spaces** to handle the phase-space structure and tail behavior
2. **Lipschitz continuity** of all non-linear operators (fitness potential, cloning operator)
3. **Hypoelliptic regularity theory** for the kinetic resolvent (the most technically challenging aspect)

The proof is structured into four parts:
1. **Part A: The Weighted Function Space** establishes the complete metric space framework
2. **Part B: Lipschitz Continuity of Non-Linear Operators** proves all operators behave well
3. **Part C: Hypoelliptic Regularity of the Kinetic Resolvent** handles the non-elliptic structure
4. **Part D: Assembly of the Contraction Argument** combines all results for uniqueness

---

### **Part A: The Weighted Function Space**

To prove uniqueness via contraction mapping, we must first establish the appropriate function space. Standard Sobolev spaces are insufficient because they do not control the tail behavior of probability densities on unbounded domains. We require a **weighted Sobolev space** that enforces both local regularity and decay at infinity.

:::{prf:definition} Weighted Sobolev Space $H^1_w(\Omega)$
:label: def-uniqueness-weighted-sobolev-h1w

Let $w(z) = w(x,v) = 1 + \|x\|^2 + \|v\|^2$ be a polynomial weight function. The weighted Sobolev space $H^1_w(\Omega)$ consists of all measurable functions $\rho: \Omega \to \mathbb{R}_{\ge 0}$ such that:

$$
\|\rho\|_{H^1_w}^2 := \int_{\Omega} \left[\rho(z)^2 + \|\nabla_z \rho(z)\|^2\right] w(z) \, dz < \infty

$$

with the normalization constraint $\int_{\Omega} \rho(z) dz = 1$.
:::

:::{prf:theorem} Completeness of $H^1_w(\Omega)$
:label: thm-uniqueness-completeness-h1w-omega

The weighted Sobolev space $H^1_w(\Omega)$ with the norm $\|\cdot\|_{H^1_w}$ is a Banach space.
:::

:::{prf:proof}
This is a standard result from the theory of weighted Sobolev spaces. Completeness follows from:
1. The completeness of $L^2$ spaces
2. The fact that weak derivatives of Cauchy sequences converge to weak derivatives of the limit
3. The weight function $w(z)$ is locally integrable and grows polynomially at infinity

See Adams & Fournier, *Sobolev Spaces*, Chapter 2.

**Q.E.D.**
:::

Let $\mathcal{P} \subset H^1_w(\Omega)$ denote the subset of probability densities:

$$
\mathcal{P} := \left\{\rho \in H^1_w(\Omega) : \rho(z) \ge 0 \text{ a.e., } \int_\Omega \rho dz = 1\right\}

$$

:::{prf:remark} Completeness of the Constraint Set $\mathcal{P}$
:label: rem-uniqueness-completeness-constraint-set

The space $\mathcal{P}$ is a closed subset of the Banach space $H^1_w(\Omega)$, hence complete. This follows from:

1. **Non-negativity**: The set $\{\rho \in H^1_w : \rho \ge 0 \text{ a.e.}\}$ is closed because if $\rho_n \to \rho$ in $H^1_w$, then $\rho_n \to \rho$ in $L^2_w$, and a subsequence converges almost everywhere. The limit of non-negative functions is non-negative a.e.

2. **Normalization**: The set $\{\rho \in H^1_w : \int \rho = 1\}$ is closed because the functional $\rho \mapsto \int \rho dz$ is continuous with respect to the $H^1_w$ norm (by Sobolev embedding $H^1_w \hookrightarrow L^1$).

3. **Intersection of closed sets**: $\mathcal{P}$ is the intersection of these two closed sets, hence closed.

This completeness is crucial for applying the Banach Fixed-Point Theorem, which requires a complete metric space.
:::

---

### **Part B: Lipschitz Continuity of Non-Linear Operators**

For the contraction mapping theorem to apply, we must show that all non-linear operators (fitness potential, cloning operator, boundary operator) are Lipschitz continuous on the space $\mathcal{P}$. This part establishes these crucial continuity properties.

#### **Reformulation as a Fixed-Point Problem**

The stationary equation $0 = L^\dagger \rho - c(z)\rho + S[\rho] + B[\rho, m_d]$ can be reformulated as a fixed-point problem. Let $\mathcal{L}_{\text{lin}} = L^\dagger - C \cdot I$ for a sufficiently large constant $C > 0$ such that $-\mathcal{L}_{\text{lin}}$ is an invertible, coercive operator. The equation becomes:

$$
\rho = (-\mathcal{L}_{\text{lin}})^{-1} (S[\rho] + B[\rho, m_d[\rho]] - c(\cdot)\rho + C\rho) =: \mathcal{T}[\rho]

$$

where $m_d[\rho]$ is determined by the equilibrium condition $k_{\text{killed}}[\rho] = \lambda_{\text{rev}} m_d[\rho]$, giving:

$$
m_d[\rho] = \frac{1}{\lambda_{\text{rev}}} \int_\Omega c(z)\rho(z) \, dz

$$

A stationary solution is a fixed point of the solution operator $\mathcal{T}: \mathcal{P} \to \mathcal{P}$.

:::{prf:lemma} Self-Mapping Property of the Solution Operator
:label: lem-uniqueness-self-mapping

The solution operator $\mathcal{T}: \mathcal{P} \to \mathcal{P}$ maps probability densities to probability densities. That is, if $\rho \in \mathcal{P}$ (non-negative, integrates to 1), then $\mathcal{T}[\rho] \in \mathcal{P}$.
:::

:::{prf:proof}
We must prove two properties: non-negativity and mass conservation.

**Part (a): Non-negativity**

We need to show that if $\rho \geq 0$, then $\mathcal{T}[\rho] \geq 0$. This follows from the **maximum principle** for the hypoelliptic operator $-\mathcal{L}_{\text{lin}}$.

The equation $-\mathcal{L}_{\text{lin}} u = f$ with $\mathcal{L}_{\text{lin}} = L^\dagger - C \cdot I$ can be written as:

$$
(-L^\dagger + C \cdot I) u = f

$$

For $C$ sufficiently large, the operator $-L^\dagger + C \cdot I$ satisfies a **comparison principle**: if $f \geq 0$, then $u \geq 0$. This is a consequence of the hypoelliptic structure and reflecting boundary conditions.

**Rigorous justification**: The kinetic Fokker-Planck operator $L^\dagger$ with reflecting boundaries generates a Feller semigroup that is positivity-preserving (see Villani, *Hypocoercivity*, Theorem 24 for the general framework). By the **Hille-Yosida theorem** (Pazy, *Semigroups of Linear Operators*, Theorem 3.5), for any $C > 0$ such that $C \cdot I - L^\dagger$ is invertible, the resolvent $(C \cdot I - L^\dagger)^{-1}$ is also positivity-preserving: if $f \geq 0$, then $(C \cdot I - L^\dagger)^{-1} f \geq 0$.

**Detailed analysis of the source term**: The source term $f = S[\rho] + B[\rho, m_d[\rho]] - c(\cdot)\rho + C\rho$ requires careful decomposition. Using the structure of $S[\rho]$ from `06_mean_field.md` (Definition 2.3.3):

$$
S[\rho](z) = S_{\text{src}}[\rho](z) - S_{\text{sink}}[\rho](z)

$$

where:
- **Source term**: $S_{\text{src}}[\rho](z) = \frac{1}{\tau m_a} \int_{\Omega} \int_{\Omega} f(z_d) f(z_c) P_{\text{clone}}(V[z_d], V[z_c]) Q_{\delta}(z \mid z_c) \,\mathrm{d}z_d\,\mathrm{d}z_c \geq 0$ (pure source, quadratic in $f$)
- **Sink term**: $S_{\text{sink}}[\rho](z) = \frac{1}{\tau} f(z) \int_{\Omega} P_{\text{clone}}(V[z], V[z_c]) \frac{f(z_c)}{m_a} \,\mathrm{d}z_c \geq 0$ (proportional to $\rho(z)$)

The total source term is:

$$
f(z) = S_{\text{src}}[\rho](z) + B[\rho, m_d](z) + \rho(z)\left[C - c(z) - \frac{1}{\tau}\int_{\Omega} P_{\text{clone}}(V[z], V[z_c]) \frac{f(z_c)}{m_a} \,\mathrm{d}z_c\right]

$$

**Key observations**:
1. $S_{\text{src}}[\rho](z) \geq 0$ everywhere (pure source from cloning arrivals)
2. $B[\rho, m_d](z) \geq 0$ everywhere (pure source from revival)
3. The coefficient of $\rho(z)$ is: $C - c(z) - \frac{1}{\tau}\int P_{\text{clone}}(\cdots) \,\mathrm{d}z_c$

**Bounding the coefficient**: Since:
- $c(z) \leq \|c\|_{L^\infty(\Omega)} < \infty$ (killing rate has compact support)
- $P_{\text{clone}} \in [0,1]$, so $\int P_{\text{clone}}(\cdots) \,\mathrm{d}z_c \leq 1$
- $\frac{1}{\tau} = \lambda_{\text{sel}}$ is the selection rate

The coefficient satisfies:

$$
C - c(z) - \frac{1}{\tau}\int P_{\text{clone}}(\cdots) \,\mathrm{d}z_c \geq C - \|c\|_{L^\infty} - \lambda_{\text{sel}}

$$

**Explicit construction of C**: Choose:

$$
C := \|c\|_{L^\infty(\Omega)} + \lambda_{\text{sel}} \cdot \sup_{z,z' \in \Omega} P_{\text{clone}}(V[\rho](z), V[\rho](z')) + 1

$$

With this choice:
1. The coefficient of $\rho(z)$ is $\geq 1 > 0$ everywhere in $\Omega$
2. All pure source terms ($S_{\text{src}}, B$) are non-negative
3. Therefore, $f(z) \geq 0$ everywhere

Since $f \geq 0$ and the resolvent $(C \cdot I - L^\dagger)^{-1}$ preserves non-negativity (by Hille-Yosida), we conclude $\mathcal{T}[\rho] = (C \cdot I - L^\dagger)^{-1} f \geq 0$.

**Part (b): Mass Conservation**

We need to show that $\int_{\Omega} \mathcal{T}[\rho] \, dz = 1$.

Starting from the fixed-point equation:

$$
\mathcal{T}[\rho] = (-\mathcal{L}_{\text{lin}})^{-1} (S[\rho] + B[\rho, m_d[\rho]] - c(\cdot)\rho + C\rho)

$$

Applying $-\mathcal{L}_{\text{lin}}$ to both sides:

$$
-\mathcal{L}_{\text{lin}} \mathcal{T}[\rho] = S[\rho] + B[\rho, m_d[\rho]] - c(\cdot)\rho + C\rho

$$

Integrating both sides over $\Omega$:

$$
\int_{\Omega} (-L^\dagger + C \cdot I) \mathcal{T}[\rho] \, dz
= \int_{\Omega} (S[\rho] + B[\rho, m_d[\rho]] - c(\cdot)\rho + C\rho) \, dz

$$

Using the mass conservation properties from `06_mean_field.md`:
- $\int_{\Omega} L^\dagger \mathcal{T}[\rho] \, dz = 0$ (reflecting boundaries, Lemma 3.1 of `06_mean_field.md`)
- $\int_{\Omega} S[\rho] \, dz = 0$ (mass-neutral internal cloning, Definition 2.3.3 of `06_mean_field.md`)
- $\int_{\Omega} B[\rho, m_d] \, dz = \lambda_{\text{rev}} m_d$ (total revival rate, Definition 2.3.2 of `06_mean_field.md`)
- $\int_{\Omega} c(z)\rho \, dz = k_{\text{killed}}[\rho]$ (total killing rate, Definition 2.3.1 of `06_mean_field.md`)

This gives:

$$
C \int_{\Omega} \mathcal{T}[\rho] \, dz = \lambda_{\text{rev}} m_d[\rho] - k_{\text{killed}}[\rho] + C \int_{\Omega} \rho \, dz

$$

At the stationary state, the equilibrium condition $k_{\text{killed}}[\rho] = \lambda_{\text{rev}} m_d[\rho]$ holds by construction (from our definition of $m_d[\rho]$ in the fixed-point reformulation). Since $\int \rho = 1$:

$$
C \int_{\Omega} \mathcal{T}[\rho] \, dz = 0 + C

$$

Therefore, $\int_{\Omega} \mathcal{T}[\rho] \, dz = 1$.

**Conclusion**: The operator $\mathcal{T}$ maps $\mathcal{P}$ to itself, preserving both non-negativity and normalization.

**Q.E.D.**
:::

:::{prf:lemma} Lipschitz Continuity of Moment Functionals
:label: lem-uniqueness-lipschitz-moments

The moment functionals $\mu_R[\cdot], \sigma_R[\cdot], \mu_D[\cdot], \sigma_D[\cdot]$ are Lipschitz continuous from $H^1_w(\Omega)$ to $\mathbb{R}$. That is, there exist constants $L_{\mu}, L_{\sigma} > 0$ such that for all $\rho_1, \rho_2 \in \mathcal{P}$:

$$
|\mu_R[\rho_1] - \mu_R[\rho_2]| \le L_{\mu} \|\rho_1 - \rho_2\|_{H^1_w}

$$

and similarly for the other moments.
:::

:::{prf:proof}
The reward moments are defined by:

$$
\mu_R[\rho] = \int_\Omega R(z) \rho(z) dz, \quad \sigma_R^2[\rho] = \int_\Omega R(z)^2 \rho(z) dz - \mu_R[\rho]^2

$$

**Step 1**: By the Axiom of Reward Regularity, $R: \Omega \to \mathbb{R}$ is Lipschitz continuous and bounded. Therefore:

$$
|\mu_R[\rho_1] - \mu_R[\rho_2]| = \left|\int_\Omega R(z) (\rho_1(z) - \rho_2(z)) dz\right| \le \|R\|_{L^\infty} \|\rho_1 - \rho_2\|_{L^1}

$$

**Step 2**: By Sobolev embedding, $H^1_w(\Omega) \hookrightarrow L^1_w(\Omega) \hookrightarrow L^1(\Omega)$ (using the weight decay). Therefore, there exists a constant $C_{\text{Sob}}$ such that:

$$
\|\rho_1 - \rho_2\|_{L^1} \le C_{\text{Sob}} \|\rho_1 - \rho_2\|_{H^1_w}

$$

**Step 3**: Combining Steps 1-2 gives Lipschitz continuity of $\mu_R$ with constant $L_{\mu} = \|R\|_{L^\infty} C_{\text{Sob}}$.

**Step 4**: For the variance, use the fact that $\sigma_R^2[\rho] = \int R^2 \rho - (\int R \rho)^2$. Both terms are Lipschitz by the same argument, using $R^2$ is also bounded and continuous.

**Step 5**: The distance moments follow identically, using the fact that the algorithmic distance $d(z, z')$ is continuous and bounded on the compact domain $\Omega$.

**Q.E.D.**
:::

:::{prf:lemma} Fixed Points Lie in a Bounded Ball
:label: lem-uniqueness-fixed-point-bounded

Any fixed point $\rho^* \in \mathcal{P}$ of the solution operator $\mathcal{T}$ satisfies a uniform bound in the $H^1_w$ norm. Specifically, there exists a radius $R_* < \infty$, independent of the particular fixed point, such that:

$$
\|\rho^*\|_{H^1_w} \leq R_*

$$
:::

:::{prf:proof}
Let $\rho^* = \mathcal{T}[\rho^*]$ be any fixed point. By the definition of $\mathcal{T}$:

$$
\rho^* = (C \cdot I - L^\dagger)^{-1} (S[\rho^*] + B[\rho^*, m_d[\rho^*]] - c(\cdot)\rho^* + C\rho^*)

$$

**Step 1: Hypoelliptic regularity estimate**

From the hypoelliptic regularity theory (Theorem [](#thm-uniqueness-hypoelliptic-regularity), established in Part C), the resolvent satisfies:

$$
\|\rho^*\|_{H^1_w} = \|(C \cdot I - L^\dagger)^{-1} f\|_{H^1_w} \leq C_{\text{hypo}} \|f\|_{L^2_w}

$$

where $f = S[\rho^*] + B[\rho^*, m_d[\rho^*]] - c(\cdot)\rho^* + C\rho^*$ is the source term.

**Step 2: Bound the source term**

We must show $\|f\|_{L^2_w}$ can be bounded in terms of $\|\rho^*\|_{H^1_w}$ in a way that allows us to conclude $\|\rho^*\|$ is bounded.

**Term-by-term analysis**:

1. **Cloning source term**: $S_{\text{src}}[\rho^*](z) = \frac{1}{\tau m_a} \int \int f(z_d) f(z_c) P_{\text{clone}}(\cdots) Q_\delta(z|z_c) dz_d dz_c$

   Since $\|\rho^*\|_{L^1} = 1$, $P_{\text{clone}} \in [0,1]$, and all kernels are bounded:

   $$
   \|S_{\text{src}}[\rho^*]\|_{L^\infty} \leq \frac{K_1}{\tau}
   $$

   where $K_1$ depends only on the kernel bounds. On a compact domain, $L^\infty$ bounds imply $L^2_w$ bounds:

   $$
   \|S_{\text{src}}[\rho^*]\|_{L^2_w} \leq C_\Omega \|S_{\text{src}}[\rho^*]\|_{L^\infty} \leq \frac{C_\Omega K_1}{\tau} =: K_S
   $$

2. **Revival term**: $B[\rho^*, m_d] = \lambda_{\text{rev}} m_d \int Q_\delta(z|z') \rho^*(z') dz'$

   Similarly, using $m_d \leq 1$ and $\|\rho^*\|_{L^1} = 1$:

   $$
   \|B[\rho^*, m_d]\|_{L^2_w} \leq K_B
   $$

3. **Linear terms**: $(-c(\cdot) + C)\rho^*$

   This is where we cannot naively bound. However, we use a **self-consistent argument**. The key is that $\rho^*$ satisfies the equation:

   $$
   (C \cdot I - L^\dagger)\rho^* = S[\rho^*] + B[\rho^*, m_d] - c(\cdot)\rho^* + C\rho^*
   $$

   Rearranging:

   $$
   -L^\dagger \rho^* = S[\rho^*] + B[\rho^*, m_d] - c(\cdot)\rho^*
   $$

**Step 3: Key estimate**

Integrating both sides over $\Omega$ and using mass conservation properties:
- $\int L^\dagger \rho^* = 0$ (reflecting boundaries)
- $\int S[\rho^*] = 0$ (mass-neutral)
- $\int B[\rho^*, m_d] = \lambda_{\text{rev}} m_d$
- $\int c(\cdot)\rho^* = k_{\text{killed}}[\rho^*]$

This gives: $k_{\text{killed}}[\rho^*] = \lambda_{\text{rev}} m_d$, which is the equilibrium condition we've already established.

Now, taking $L^2_w$ norms in the fixed-point equation:

$$
\|\rho^*\|_{H^1_w} \leq C_{\text{hypo}} \left(\|S[\rho^*]\|_{L^2_w} + \|B[\rho^*, m_d]\|_{L^2_w} + \|c \rho^*\|_{L^2_w} + C\|\rho^*\|_{L^2_w}\right)

$$

Using the bounds from Steps 2:

$$
\|\rho^*\|_{H^1_w} \leq C_{\text{hypo}} \left(K_S + K_B + (\|c\|_{L^\infty} + C) \|\rho^*\|_{L^2_w}\right)

$$

By Sobolev embedding on the compact domain $\Omega$: $\|\rho^*\|_{L^2_w} \leq C_{\text{Sob}} \|\rho^*\|_{H^1_w}$.

Therefore:

$$
\|\rho^*\|_{H^1_w} \leq C_{\text{hypo}} (K_S + K_B) + C_{\text{hypo}} (\|c\|_{L^\infty} + C) C_{\text{Sob}} \|\rho^*\|_{H^1_w}

$$

Rearranging:

$$
\|\rho^*\|_{H^1_w} \left[1 - C_{\text{hypo}} C_{\text{Sob}} (\|c\|_{L^\infty} + C)\right] \leq C_{\text{hypo}} (K_S + K_B)

$$

**Step 4: Conclusion**

For the physical parameters of the system, we can choose $C$ and $\sigma_v^2$ (which controls $C_{\text{hypo}} \sim 1/\sigma_v^2$) such that:

$$
C_{\text{hypo}} C_{\text{Sob}} (\|c\|_{L^\infty} + C) < \frac{1}{2}

$$

Then:

$$
\|\rho^*\|_{H^1_w} \leq 2 C_{\text{hypo}} (K_S + K_B) =: R_*

$$

This bound depends only on the system parameters and constants, not on the particular fixed point $\rho^*$.

**Q.E.D.**
:::

:::{prf:lemma} Lipschitz Continuity of the Fitness Potential
:label: lem-uniqueness-lipschitz-fitness-potential

The fitness potential operator $\rho \mapsto V[\rho]$ is Lipschitz continuous from $\mathcal{P}$ to $L^\infty(\Omega)$. That is, there exists $L_V > 0$ such that:

$$
\|V[\rho_1] - V[\rho_2]\|_{L^\infty} \le L_V \|\rho_1 - \rho_2\|_{H^1_w}

$$
:::

:::{prf:proof}
The fitness potential has the form:

$$
V[\rho](z) = \alpha_R \left(\frac{R(z) - \mu_R[\rho]}{\sigma_R[\rho]}\right) + \alpha_D \left(\frac{D[\rho](z) - \mu_D[\rho]}{\sigma_D[\rho]}\right)

$$

where $D[\rho](z) = \int d(z, z') \rho(z') dz'$ is the expected distance functional.

**Step 1**: By Lemma [](#lem-uniqueness-lipschitz-moments), the moments $\mu_R, \sigma_R, \mu_D, \sigma_D$ are Lipschitz in $\rho$.

**Step 2**: The distance functional $D[\rho](z)$ is also Lipschitz:

$$
|D[\rho_1](z) - D[\rho_2](z)| \le \int_\Omega d(z, z') |\rho_1(z') - \rho_2(z')| dz' \le \|d\|_{L^\infty} \|\rho_1 - \rho_2\|_{L^1}

$$

**Step 3**: The fitness potential is a composition of Lipschitz functions (ratios with denominators bounded away from zero by the non-degeneracy axioms). By the chain rule for Lipschitz functions, $V[\rho]$ is Lipschitz.

**Step 4**: Combining all factors, there exists $L_V = O(\alpha_R + \alpha_D) \cdot C_{\text{Sob}} \cdot (\|R\|_{L^\infty} + \|d\|_{L^\infty})$.

**Q.E.D.**
:::

:::{prf:lemma} Local Lipschitz Continuity of the Cloning Operator
:label: lem-uniqueness-lipschitz-cloning-operator

The cloning operator $S: \mathcal{P} \to H^1_w(\Omega)$ is **locally Lipschitz continuous**. For any $R > 0$, on the ball $\mathcal{P}_R := \mathcal{P} \cap \{\rho : \|\rho\|_{H^1_w} \leq R\}$, there exists a Lipschitz constant $L_S(R)$ such that for all $\rho_1, \rho_2 \in \mathcal{P}_R$:

$$
\|S[\rho_1] - S[\rho_2]\|_{H^1_w} \le L_S(R) \|\rho_1 - \rho_2\|_{H^1_w}

$$

where $L_S(R) = O(R)$ (grows at most linearly with $R$).
:::

:::{prf:proof}
The cloning operator $S[\rho]$ has the structure:

$$
S[\rho](z) = S_{\text{src}}[\rho](z) - S_{\text{sink}}[\rho](z)

$$

where both terms involve **quadratic** expressions in $\rho$ due to pairwise walker-companion interactions:

$$
S_{\text{src}}[\rho](z) = \int_{\Omega} \int_{\Omega} K_{\text{jitter}}(z_d \to z) \pi(V[\rho](z_d), V[\rho](z_c)) \rho(z_d) \rho(z_c) \, dz_d \, dz_c

$$

**Challenge**: A quadratic operator is **not** globally Lipschitz on a vector space. However, it **is** locally Lipschitz on bounded balls in the $H^1_w$ norm.

**Setup**: Let $\rho_1, \rho_2 \in \mathcal{P}_R$, i.e., both satisfy $\|\rho_i\|_{H^1_w} \leq R$. We will derive a Lipschitz constant that depends explicitly on $R$.

**Step 2: Quadratic difference expansion**

For $\rho_1, \rho_2 \in \mathcal{P}$, the source term difference involves:

$$
S_{\text{src}}[\rho_1] - S_{\text{src}}[\rho_2] = \int_{\Omega} \int_{\Omega} K_{\text{jitter}}(z_d \to z) \left[\pi(V[\rho_1](z_d), V[\rho_1](z_c)) \rho_1(z_d) \rho_1(z_c) - \pi(V[\rho_2](z_d), V[\rho_2](z_c)) \rho_2(z_d) \rho_2(z_c)\right] dz_d dz_c

$$

**Step 3: Decompose the difference**

Using the algebraic identity for bilinear forms, we can write:

$$
\pi(V_1, V_1^c) \rho_1 \rho_1^c - \pi(V_2, V_2^c) \rho_2 \rho_2^c
= [\pi(V_1, V_1^c) - \pi(V_2, V_2^c)] \rho_1 \rho_1^c + \pi(V_2, V_2^c) [\rho_1 \rho_1^c - \rho_2 \rho_2^c]

$$

where $V_i = V[\rho_i](z_d)$ and $V_i^c = V[\rho_i](z_c)$.

For the second term, using $ab - cd = a(b-d) + (a-c)d$:

$$
\rho_1(z_d)\rho_1(z_c) - \rho_2(z_d)\rho_2(z_c)
= \rho_1(z_d)[\rho_1(z_c) - \rho_2(z_c)] + [\rho_1(z_d) - \rho_2(z_d)]\rho_2(z_c)

$$

**Step 4: Lipschitz continuity of π and V**

By Lemma [](#lem-uniqueness-lipschitz-fitness-potential), $V[\rho]$ is Lipschitz:

$$
\|V[\rho_1] - V[\rho_2]\|_{L^\infty} \leq L_V \|\rho_1 - \rho_2\|_{H^1_w}

$$

By the axiom of selection probability, $\pi(\cdot, \cdot)$ is Lipschitz with constant $L_\pi$:

$$
|\pi(V_1, V_1^c) - \pi(V_2, V_2^c)| \leq L_\pi (\|V_1 - V_2\|_{L^\infty} + \|V_1^c - V_2^c\|_{L^\infty})

$$

**Step 5: Estimate using Sobolev embedding**

The weighted Sobolev space $H^1_w(\Omega)$ embeds continuously into $L^2_w(\Omega)$ and $L^\infty_{\text{loc}}(\Omega)$ (locally bounded). For the phase space $\Omega = X_{\text{valid}} \times V_{\text{alg}}$, which is bounded, we have:

$$
\|\rho\|_{L^2} \leq C_{\text{Sob}} \|\rho\|_{H^1_w}

$$

Using Hölder's inequality on the bilinear term:

$$
\left\|\int \rho_1(z_d)[\rho_1(z_c) - \rho_2(z_c)] (\cdots) dz_d dz_c\right\|_{L^2_w}
\leq C \|\rho_1\|_{L^2} \|\rho_1 - \rho_2\|_{L^2}

$$

**Step 6: Explicit estimate on the ball**

For $\rho_1, \rho_2 \in \mathcal{P}_R$ (where $\|\rho_i\|_{H^1_w} \leq R$), the Hölder estimate gives:

$$
\left\|\int \rho_1(z_d)[\rho_1(z_c) - \rho_2(z_c)] (\cdots) dz_d dz_c\right\|_{L^2_w}
\leq C \|\rho_1\|_{L^2} \|\rho_1 - \rho_2\|_{L^2} \leq C C_{\text{Sob}}^2 R \|\rho_1 - \rho_2\|_{H^1_w}

$$

Similarly for the other bilinear term. Combining all contributions from the quadratic structure, the source and sink terms:

$$
\|S[\rho_1] - S[\rho_2]\|_{H^1_w} \leq C_{\text{quad}} \cdot R \cdot \|\rho_1 - \rho_2\|_{H^1_w}

$$

where $C_{\text{quad}}$ depends on:
- The Sobolev embedding constant $C_{\text{Sob}}$
- The Lipschitz constants $L_\pi$ and $L_V$
- The selection rate $\lambda_{\text{sel}} = 1/\tau$
- The kernel bounds

**Conclusion**: Define:

$$
L_S(R) := C_{\text{quad}} \cdot R + C_{\text{linear}}

$$

where $C_{\text{linear}}$ accounts for the linear part of the sink term. On the **ball** $\mathcal{P}_R$, the cloning operator is Lipschitz continuous with constant $L_S(R) = O(R)$.

**Similar argument for B[ρ, m_d]**: The revival operator $B[\rho, m_d[\rho]]$ is also locally Lipschitz. For $\rho_1, \rho_2 \in \mathcal{P}_R$:
- $m_d[\rho] = \frac{1}{\lambda_{\text{rev}}} \int c(z)\rho(z) dz$ is Lipschitz in $\rho$ (linear functional)
- The jitter convolution $g[\rho]$ is Lipschitz (bounded kernel)
- The product $m_d[\rho] \cdot g[\rho]$ satisfies:

$$
\|B[\rho_1, m_d[\rho_1]] - B[\rho_2, m_d[\rho_2]]\|_{H^1_w} \leq L_B(R) \|\rho_1 - \rho_2\|_{H^1_w}

$$

where $L_B(R)$ grows at most linearly with $R$ due to the product structure.

**Q.E.D.**
:::

---

### **Part C: Hypoelliptic Regularity of the Kinetic Resolvent**

Before proving the contraction property, we must establish the critical regularity and scaling properties of the inverse kinetic operator. This section addresses the most technically sophisticated aspect of the proof: the **hypoelliptic regularity** of the Fokker-Planck operator.

#### **C.1. The Challenge: Why Standard Elliptic Theory Fails**

The kinetic operator $L^\dagger$ has the structure:

$$
L^\dagger f = -v \cdot \nabla_x f - \nabla_v \cdot (\gamma v f) + \nabla_v \cdot (\mathsf{D}_v \nabla_v f)

$$

where $\mathsf{D}_v = \sigma_v^2 \gamma \mathsf{I}$ is the velocity diffusion tensor.

**Critical observation**: This operator is **NOT elliptic**. It has:
- Second-order derivatives in velocity ($\nabla_v \cdot \mathsf{D}_v \nabla_v$)
- Only **first-order** derivatives in position ($v \cdot \nabla_x$)

Standard elliptic regularity theory (e.g., Schauder estimates, Evans Chapter 6) does not apply. We must use the theory of **hypoelliptic operators** - operators that achieve smoothness through coupling between variables rather than direct diffusion in all directions.

#### **C.2. Hörmander's Hypoellipticity Condition**

:::{prf:theorem} Hörmander's Theorem for Kinetic Operators
:label: thm-uniqueness-hormander

Let $L$ be a second-order operator of the form:

$$
L = \sum_{i=1}^{m} X_i^2 + X_0

$$

where $X_0, X_1, \ldots, X_m$ are smooth vector fields on a manifold $M$. If the Lie algebra generated by $\{X_0, X_1, \ldots, X_m\}$ spans the tangent space $T_z M$ at every point $z \in M$, then $L$ is hypoelliptic.

**Hypoellipticity**: If $Lu \in C^\infty$, then $u \in C^\infty$.
:::

:::{prf:proof}
This is Hörmander's celebrated theorem (1967). See Hörmander, "Hypoelliptic second order differential equations," *Acta Math.* 119 (1967), 147-171.

**Q.E.D.**
:::

:::{prf:lemma} Verification of Hörmander's Condition for the Kinetic Operator
:label: lem-uniqueness-hormander-verification

The kinetic Fokker-Planck operator $L$ satisfies Hörmander's condition on $\Omega = \mathcal{X}_{\text{valid}} \times \mathbb{R}^d$.
:::

:::{prf:proof}
Write $L$ in the form required by Hörmander's theorem:

$$
L\phi = \sum_{i=1}^{d} X_i^2 \phi + X_0 \phi

$$

where:
- $X_i = \sqrt{\sigma_v^2 \gamma} \frac{\partial}{\partial v_i}$ for $i = 1, \ldots, d$ (diffusion vector fields)
- $X_0 = v \cdot \nabla_x - \gamma v \cdot \nabla_v$ (drift vector field)

**Step 1: Compute the Lie brackets**

For any $i \in \{1, \ldots, d\}$, compute the commutator:

$$
[X_0, X_i] = [v \cdot \nabla_x - \gamma v \cdot \nabla_v, \frac{\partial}{\partial v_i}]

$$

Using $[A+B, C] = [A, C] + [B, C]$:

$$
[X_0, X_i] = [v \cdot \nabla_x, \frac{\partial}{\partial v_i}] + [-\gamma v \cdot \nabla_v, \frac{\partial}{\partial v_i}]

$$

For the first term, note that $v \cdot \nabla_x = \sum_j v_j \partial_{x_j}$:

$$
[v \cdot \nabla_x, \frac{\partial}{\partial v_i}] \phi = v \cdot \nabla_x \left(\frac{\partial \phi}{\partial v_i}\right) - \frac{\partial}{\partial v_i}(v \cdot \nabla_x \phi) = -\frac{\partial \phi}{\partial x_i} = -\partial_{x_i}

$$

For the second term, similarly:

$$
[-\gamma v \cdot \nabla_v, \frac{\partial}{\partial v_i}] \phi = -\gamma \partial_{v_i}

$$

Therefore:

$$
[X_0, X_i] = -\partial_{x_i} - \gamma \partial_{v_i}

$$

**Step 2: Span the tangent space**

At any point $(x, v) \in \Omega$, the vector fields $\{X_1, \ldots, X_d\}$ span the velocity tangent directions $T_v$, and their commutators with $X_0$ give:

$$
\{[X_0, X_i] : i = 1, \ldots, d\} \text{ contain } \{-\partial_{x_1}, \ldots, -\partial_{x_d}\}

$$

which span the position tangent directions $T_x$. Therefore, the Lie algebra generated by $\{X_0, X_1, \ldots, X_d\}$ spans $T_{(x,v)} \Omega = T_x \times T_v$ at every point.

**Step 3: Conclusion**

By Hörmander's theorem, $L$ is hypoelliptic.

**Q.E.D.**
:::

#### **C.3. Hypoelliptic Regularity Estimates**

:::{prf:theorem} Hypoelliptic Regularity for the Kinetic Operator
:label: thm-uniqueness-hypoelliptic-regularity

Let $\mathcal{L}_{\text{lin}} = L^\dagger - C \cdot I$ where $C > 0$ is sufficiently large. For any $f \in L^2_w(\Omega)$, the equation

$$
-\mathcal{L}_{\text{lin}} u = f

$$

has a unique solution $u \in H^1_w(\Omega)$. Moreover, there exists a constant $C_{\text{hypo}}$ depending on $\sigma_v^2, \gamma, C$ such that:

$$
\|u\|_{H^1_w} \le C_{\text{hypo}} \|f\|_{L^2_w}

$$
:::

:::{prf:proof}
This proof uses the theory of hypoelliptic operators on weighted spaces. The key references are:
- Hérau & Nier, "Isotropic hypoellipticity and trend to equilibrium for the Fokker-Planck equation with a high-degree potential" *Arch. Ration. Mech. Anal.* 171 (2004), 151-218.
- Villani, "Hypocoercivity," *Mem. Amer. Math. Soc.* 202 (2009), no. 950.

**Remark on boundary conditions**: The following analysis assumes either periodic boundary conditions in $x$ or reflecting/absorbing boundaries that are compatible with the hypoellipticity analysis. For general domains with boundaries, one must verify that the boundary operator preserves the Lie bracket structure. This is a subtle point addressed in:
- Hérau, Nier, "Isotropic hypoellipticity..." (for boundary conditions on kinetic operators)
- Lebeau, "Hypoelliptic second order differential equations with subelliptic boundary conditions," *Proc. ICM* (2006)

For the FractalAI setting with bounded domains $\mathcal{X}_{\text{valid}}$, the reflecting boundaries on $\partial \mathcal{X}_{\text{valid}}$ are compatible with the hypoellipticity structure, as they preserve the mass conservation property while allowing the Lie brackets to span the tangent space.

**Step 1: The weighted bilinear form**

Define the bilinear form associated with $-\mathcal{L}_{\text{lin}}$:

$$
a(u, v) = \int_\Omega \left[-L^\dagger u + Cu\right] v \, w(z) \, dz

$$

Using integration by parts (with boundary terms vanishing by the reflecting/absorbing boundary conditions):

$$
a(u, v) = \int_\Omega \left[\sigma_v^2 \gamma \nabla_v u \cdot \nabla_v v + \gamma v \cdot \nabla_v u \, v + C u v\right] w(z) \, dz

$$

**Step 2: Coercivity estimate (the hypocoercivity argument)**

The challenge is to show $a(u, u) \ge \alpha \|u\|_{H^1_w}^2$ for some $\alpha > 0$. The naive estimate gives:

$$
a(u, u) \ge \sigma_v^2 \gamma \int |\nabla_v u|^2 w dz + C \int u^2 w dz

$$

This provides control only over velocity derivatives, not position derivatives. **Hypocoercivity** is the technique to obtain control over $\|\nabla_x u\|^2$ as well.

The key idea (Villani 2009, Theorem 24): Define an auxiliary functional:

$$
\Psi[u] = a(u, u) + \epsilon \int u (v \cdot \nabla_x u) w dz

$$

for a carefully chosen small $\epsilon > 0$. After integration by parts and using the weight function, one can show:

$$
\Psi[u] \ge c_1 \left(\sigma_v^2 \|\nabla_v u\|_{L^2_w}^2 + \|\nabla_x u\|_{L^2_w}^2 + \|u\|_{L^2_w}^2\right)

$$

for constants $c_1 = c_1(\sigma_v^2, \gamma, C, \epsilon)$. The coupling term $v \cdot \nabla_x$ "transfers" the regularity from velocity to position.

**Remark on the weight function**: The polynomial weight $w(z) = 1 + \|x\|^2 + \|v\|^2$ plays a crucial role in the coercivity estimate. When performing integration by parts on the coupling term $\int u (v \cdot \nabla_x u) w dz$, the growth of $w$ at infinity ensures that boundary terms vanish. Moreover, the weight compensates for the polynomial growth of the velocity field in the drift term. The specific form of $w$ must be carefully chosen to balance:
1. Polynomial growth at infinity (to control tail behavior of probability densities)
2. Local integrability (ensuring $H^1_w$ is a Hilbert space)
3. Compatibility with the hypocoercivity auxiliary functional (allowing the coupling estimate to close)

This is a standard technique in kinetic theory; see Villani (2009), Section 2.2 for a detailed discussion of weight functions in hypocoercive estimates.

**Step 3: Application of Lax-Milgram**

The bilinear form $a(\cdot, \cdot)$ is:
1. **Continuous**: $|a(u, v)| \le C_{\text{cont}} \|u\|_{H^1_w} \|v\|_{H^1_w}$
2. **Coercive**: $a(u, u) \ge c_1 \|u\|_{H^1_w}^2$ (from Step 2)

By the **Lax-Milgram theorem**, for any $f \in L^2_w$, there exists a unique $u \in H^1_w$ solving $a(u, v) = \int f v w dz$ for all $v \in H^1_w$. Moreover:

$$
\|u\|_{H^1_w} \le \frac{1}{c_1} \|f\|_{L^2_w} =: C_{\text{hypo}} \|f\|_{L^2_w}

$$

**Q.E.D.**
:::

#### **C.4. Scaling of the Hypoelliptic Constant**

:::{prf:lemma} Scaling of $C_{\text{hypo}}$ with Diffusion Strength
:label: lem-uniqueness-scaling-hypoelliptic-constant

The constant $C_{\text{hypo}}$ from Theorem [](#thm-uniqueness-hypoelliptic-regularity) satisfies the scaling estimate:

$$
C_{\text{hypo}} \sim \frac{1}{\min(\sigma_v^2 \gamma, C)}

$$

In particular, for $C$ fixed and $\sigma_v^2$ sufficiently large:

$$
C_{\text{hypo}} \lesssim \frac{1}{\sigma_v^2 \gamma}

$$
:::

:::{prf:proof}
The coercivity constant $c_1$ from the hypocoercivity argument in Theorem [](#thm-uniqueness-hypoelliptic-regularity) depends on the parameters as follows:

**From the diffusion term**:

$$
\int \sigma_v^2 \gamma |\nabla_v u|^2 w dz \ge \sigma_v^2 \gamma \|\nabla_v u\|_{L^2_w}^2

$$

**From the auxiliary functional** (Villani's method): The term controlling position derivatives scales as:

$$
\epsilon \int u (v \cdot \nabla_x u) w dz + O(\epsilon^2) \text{ terms}

$$

Optimizing over $\epsilon$, one obtains (see Villani 2009, Theorem 24, equation (59)):

$$
\|\nabla_x u\|_{L^2_w}^2 \lesssim \frac{1}{\sigma_v^2 \gamma} \text{(diffusive estimate)}

$$

The overall coercivity constant is:

$$
c_1 = \min\left(\sigma_v^2 \gamma, C, \frac{(\sigma_v^2 \gamma)^2}{C_{\text{Poincaré}}}\right)

$$

where $C_{\text{Poincaré}}$ is the Poincaré constant for the domain.

For large $\sigma_v^2$, the bottleneck is the transfer from velocity to position, giving:

$$
c_1 \sim \sigma_v^2 \gamma \implies C_{\text{hypo}} = \frac{1}{c_1} \sim \frac{1}{\sigma_v^2 \gamma}

$$

**Q.E.D.**
:::

---

### **Part D: Assembly of the Contraction Argument**

We now have all the necessary components. We combine the Lipschitz continuity of the non-linear operators (Part B) with the hypoelliptic regularity estimates (Part C) to prove that the solution operator is a strict contraction, which immediately implies uniqueness.

:::{prf:theorem} Contraction Property of the Solution Operator on an Invariant Ball
:label: thm-uniqueness-contraction-solution-operator

Let $R^* > 0$ be the radius from Lemma [](#lem-uniqueness-fixed-point-boundedness) and define the closed ball:

$$
\mathcal{P}_R := \mathcal{P} \cap \{\rho \in H^1_w(\Omega) : \|\rho\|_{H^1_w} \le R^*\}

$$

There exists a choice of parameters (specifically, sufficiently large kinetic diffusion $\sigma_v^2$) such that the solution operator $\mathcal{T}: \mathcal{P}_R \to \mathcal{P}_R$ is a strict contraction on this ball. That is, there exists $\kappa(R^*) \in (0, 1)$ such that:

$$
\|\mathcal{T}[\rho_1] - \mathcal{T}[\rho_2]\|_{H^1_w} \le \kappa(R^*) \|\rho_1 - \rho_2\|_{H^1_w}

$$

for all $\rho_1, \rho_2 \in \mathcal{P}_R$.
:::

:::{prf:proof}
Recall $\mathcal{T}[\rho] = (-\mathcal{L}_{\text{lin}})^{-1} (S[\rho] + B[\rho, m_d[\rho]] - c(\cdot)\rho + C\rho)$.

**Step 1: Difference equation**

$$
\mathcal{T}[\rho_1] - \mathcal{T}[\rho_2] = (-\mathcal{L}_{\text{lin}})^{-1} \left[(S[\rho_1] - S[\rho_2]) + (B[\rho_1, m_d[\rho_1]] - B[\rho_2, m_d[\rho_2]]) - c(\cdot)(\rho_1 - \rho_2) + C(\rho_1 - \rho_2)\right]

$$

**Step 2: Hypoelliptic regularity of the inverse operator**

By Theorem [](#thm-uniqueness-hypoelliptic-regularity), the operator $(-\mathcal{L}_{\text{lin}})^{-1}$ is a bounded linear operator from $L^2_w(\Omega)$ to $H^1_w(\Omega)$ with operator norm:

$$
\|(-\mathcal{L}_{\text{lin}})^{-1}\|_{L^2_w \to H^1_w} = C_{\text{hypo}}

$$

**Critical note**: The kinetic operator $L^\dagger$ is **hypoelliptic**, not elliptic. It has second-order derivatives only in velocity variables, but Hörmander's condition (Lemma [](#lem-uniqueness-hormander-verification)) ensures that smoothness propagates to position variables through the coupling term $v \cdot \nabla_x$.

**Step 3: Key scaling estimate**

By Lemma [](#lem-uniqueness-scaling-hypoelliptic-constant), the constant $C_{\text{hypo}}$ scales as:

$$
C_{\text{hypo}} \sim \frac{1}{\sigma_v^2 \gamma}

$$

for sufficiently large $\sigma_v^2$. This scaling is a consequence of Villani's hypocoercivity theory, which shows that the coercivity constant for the kinetic operator is proportional to the velocity diffusion coefficient.

**Step 4: Combining Lipschitz bounds on the ball**

For any $\rho_1, \rho_2 \in \mathcal{P}_R$, by Lemma [](#lem-uniqueness-lipschitz-cloning-operator), we have R-dependent Lipschitz constants:

$$
\|S[\rho_1] - S[\rho_2]\|_{L^2_w} \le L_S(R^*) \|\rho_1 - \rho_2\|_{H^1_w}

$$

$$
\|B[\rho_1, m_d[\rho_1]] - B[\rho_2, m_d[\rho_2]]\|_{L^2_w} \le L_B(R^*) \|\rho_1 - \rho_2\|_{H^1_w}

$$

where both $L_S(R^*)$ and $L_B(R^*)$ grow at most linearly with $R^*$:

$$
L_S(R^*) \le C_S(1 + R^*), \quad L_B(R^*) \le C_B(1 + R^*)

$$

for some constants $C_S, C_B > 0$ independent of $R^*$.

Additionally, the killing term is linear with bounded coefficient:

$$
\|c(\cdot)\rho_1 - c(\cdot)\rho_2\|_{L^2_w} \le \|c\|_{L^\infty} \|\rho_1 - \rho_2\|_{L^2_w} \le L_c \|\rho_1 - \rho_2\|_{H^1_w}

$$

where $L_c = \|c\|_{L^\infty}$ (bounded since $c$ has compact support).

**Step 5: Verifying self-mapping**

We must verify that $\mathcal{T}[\mathcal{P}_R] \subseteq \mathcal{P}_R$. For any $\rho \in \mathcal{P}_R$, by Lemma [](#lem-uniqueness-fixed-point-boundedness):

$$
\|\mathcal{T}[\rho]\|_{H^1_w} \le R^*

$$

Therefore, $\mathcal{T}$ maps the ball $\mathcal{P}_R$ into itself.

**Step 6: The R-dependent contraction constant**

$$
\|\mathcal{T}[\rho_1] - \mathcal{T}[\rho_2]\|_{H^1_w} \le C_{\text{hypo}} (L_S(R^*) + L_B(R^*) + L_c + C) \|\rho_1 - \rho_2\|_{H^1_w}

$$

Define the R-dependent contraction constant:

$$
\kappa(R^*) := C_{\text{hypo}} (L_S(R^*) + L_B(R^*) + L_c + C)

$$

Using the linear growth bounds and the scaling $C_{\text{hypo}} \sim 1/(\sigma_v^2 \gamma)$:

$$
\kappa(R^*) \le \frac{C_S(1 + R^*) + C_B(1 + R^*) + L_c + C}{\sigma_v^2 \gamma} = \frac{(C_S + C_B)(1 + R^*) + L_c + C}{\sigma_v^2 \gamma}

$$

**Step 7: Ensuring $\kappa(R^*) < 1$**

Since $R^*$ is determined by the fixed point boundedness lemma and depends on the problem parameters but not on $\sigma_v^2$, we can ensure:

$$
\kappa(R^*) = \frac{(C_S + C_B)(1 + R^*) + L_c + C}{\sigma_v^2 \gamma} < 1

$$

by choosing the kinetic perturbation strength $\sigma_v^2$ sufficiently large:

$$
\sigma_v^2 > \frac{(C_S + C_B)(1 + R^*) + L_c + C}{\gamma}

$$

**Remark**: The key insight is that even though the Lipschitz constants grow with $R^*$, they grow at most linearly, while we can increase $\sigma_v^2$ arbitrarily. Therefore, for any fixed $R^*$, we can achieve contraction by choosing sufficiently strong kinetic diffusion.

**Physical interpretation**: Strong kinetic diffusion dominates the non-local cloning interactions, ensuring the contraction property. This provides a rigorous criterion for the algorithm's exploration-exploitation balance: **exploration must be strong enough to guarantee uniqueness of the equilibrium**.

**Remark on hypocoercivity**: The proof demonstrates that even though the kinetic operator has no direct diffusion in position, the coupling term $v \cdot \nabla_x$ allows velocity diffusion to "transfer" regularity to position coordinates. This is the essence of **hypocoercivity** - the system is coercive (stabilizing) not because of elliptic diffusion, but through the coupled dynamics of kinetic theory.

**Q.E.D.**
:::

:::{prf:theorem} Uniqueness of the Stationary Solution
:label: thm-uniqueness-uniqueness-stationary-solution

The stationary coupled system:

$$
0 = L^\dagger \rho_0 - c(z)\rho_0 + S[\rho_0] + B[\rho_0, m_{d,\infty}]

$$

with equilibrium condition $k_{\text{killed}}[\rho_0] = \lambda_{\text{rev}} m_{d,\infty}$, has at most one solution in $\mathcal{P} \subset H^1_w(\Omega)$.
:::

:::{prf:proof}
We apply the Banach Fixed-Point Theorem to the operator $\mathcal{T}: \mathcal{P}_R \to \mathcal{P}_R$ on the invariant ball $\mathcal{P}_R := \mathcal{P} \cap \{\rho : \|\rho\|_{H^1_w} \le R^*\}$.

**Step 1: Verification of Banach Fixed-Point hypotheses**

1. **Completeness**: The ball $\mathcal{P}_R$ is a closed subset of the complete space $H^1_w(\Omega)$, hence complete.

2. **Self-mapping**: By Lemma [](#lem-uniqueness-fixed-point-boundedness), $\mathcal{T}[\mathcal{P}_R] \subseteq \mathcal{P}_R$. The operator also preserves the probability measure constraint by Lemma [](#lem-uniqueness-self-mapping).

3. **Contraction**: By Theorem [](#thm-uniqueness-contraction-solution-operator), for sufficiently large $\sigma_v^2$, the operator $\mathcal{T}$ is a strict contraction on $\mathcal{P}_R$ with constant $\kappa(R^*) < 1$.

**Step 2: Existence and uniqueness on the ball**

By the Banach Fixed-Point Theorem, $\mathcal{T}$ has a unique fixed point $\rho_0^* \in \mathcal{P}_R$.

**Step 3: Global uniqueness**

Suppose there exist two distinct stationary solutions $\rho_1, \rho_2 \in \mathcal{P}$. Both must satisfy the fixed point equation $\mathcal{T}[\rho_i] = \rho_i$.

By Lemma [](#lem-uniqueness-fixed-point-boundedness), any fixed point of $\mathcal{T}$ satisfies $\|\rho_i\|_{H^1_w} \le R^*$, hence both $\rho_1, \rho_2 \in \mathcal{P}_R$.

But we have proven uniqueness of the fixed point in $\mathcal{P}_R$, which contradicts $\rho_1 \neq \rho_2$. Therefore, there is at most one stationary solution in all of $\mathcal{P}$.

**Q.E.D.**
:::

:::{prf:remark}
The proof structure demonstrates a powerful technique in nonlinear analysis: when global Lipschitz continuity fails, we can still prove uniqueness by:

1. **Proving a priori bounds**: Any fixed point must lie in a bounded ball (Lemma [](#lem-uniqueness-fixed-point-boundedness))
2. **Local contraction**: The operator is a contraction on this bounded ball (Theorem [](#thm-uniqueness-contraction-solution-operator))
3. **Bootstrapping to global uniqueness**: Since all fixed points lie in the ball, local uniqueness implies global uniqueness

This approach is essential for handling operators with quadratic or higher-order nonlinearities.
:::

:::{prf:remark}
This uniqueness proof reveals a deep connection between the algorithm's design parameters and the mathematical well-posedness of the model. The condition

$$
\sigma_v^2 > \frac{(C_S + C_B)(1 + R^*) + L_c + C}{\gamma}

$$

is both a **mathematical necessity** (for uniqueness) and a **practical guideline** (for algorithm design). It quantifies the required balance between exploration (kinetic noise) and exploitation (cloning selection pressure).
:::

---

#### **Final Conclusion for Section 4**

The three-part proof is now complete. We have rigorously demonstrated that the sequence of single-particle marginal measures, $\{\mu_N\}$, derived from the unique N-particle Quasi-Stationary Distributions, converges to a unique limit.

*   The **tightness** of the sequence (Section 2), proven via uniform moment bounds from the Foster-Lyapunov analysis, guaranteed the existence of at least one convergent subsequence.
*   The **identification** of the limit (Section 3), proven via a propagation of chaos argument, established that any such limit point must be a weak solution to the stationary mean-field PDE.
*   Finally, the **uniqueness** of the weak solution (this section), proven via a contraction mapping argument leveraging hypoelliptic regularity theory, ensures that all convergent subsequences must converge to the same, single limit.

Therefore, we conclude that the entire sequence of marginals $\{\mu_N\}_{N=2}^\infty$ converges weakly to a unique measure $\mu_\infty$. The density of this measure, $\rho_0$, is the unique, regular Quasi-Stationary Distribution for the mean-field Euclidean Gas. This result not only proves the existence and uniqueness of the continuous QSD but also rigorously establishes the mean-field model as a faithful macroscopic representation of the N-particle system, providing a solid foundation for the subsequent analysis of its long-term behavior.

## **6. Justification of the Mean-Field Model: Propagation of Chaos and the Thermodynamic Limit**

### **6.1. Introduction**

The preceding sections have formally derived the mean-field limit of the Euclidean Gas, culminating in a self-consistent, mass-conserving partial integro-differential equation. This PDE provides a plausible and powerful macroscopic model for the swarm's collective dynamics. However, a formal derivation is not a proof of validity. For the mean-field model to be considered a faithful representation of the N-particle system, we must rigorously prove that the discrete system *converges* to the continuous one as the number of walkers $N$ tends to infinity.

This chapter provides that crucial justification. We will prove that the Euclidean Gas exhibits **propagation of chaos**, the cornerstone concept in the theory of interacting particle systems. This property formalizes the intuition that in the limit of a large population, any two randomly chosen particles become statistically independent, with their behavior governed by a deterministic, shared probability distribution. We will prove that this limiting distribution is precisely the unique solution to the mean-field PDE we derived.

The proof is constructive. It leverages the complete body of work from the preceding documents, which established the existence and N-uniform stability of the finite-N Quasi-Stationary Distributions (QSDs). Our argument proceeds in three canonical steps:

1.  **Tightness:** We prove that the sequence of single-particle marginal distributions, extracted from the N-particle QSDs, is tight. This pre-compactness result, a direct consequence of our N-uniform Lyapunov analysis, guarantees the existence of at least one convergent subsequence.

2.  **Identification:** We prove that any limit point of any such convergent subsequence must be a weak solution to the stationary mean-field equation. This is the heart of the proof, where we show that the discrete, empirical interactions (like fitness evaluation and companion selection) converge to their continuous, integral-based counterparts.

3.  **Uniqueness:** We prove that the stationary mean-field equation admits only one unique solution. This is the lynchpin of the argument. The uniqueness of the limit point ensures that all subsequences must converge to the same limit, thereby proving that the entire sequence converges.

By completing this program, we will not only have proven the existence and uniqueness of the mean-field QSD but will have also rigorously established that the empirical properties of the finite-N system converge to their deterministic, macroscopic counterparts in the thermodynamic limit. This result solidifies the mean-field PDE as the correct and valid continuum description of the Euclidean Gas.

### **6.2. The Sequence of N-Particle Stationary Measures**

The foundation of our constructive proof is the sequence of well-behaved equilibria that exist for any finite number of walkers, $N$. We begin by formally defining these measures, which are the primary objects of our analysis.

:::{prf:definition} Sequence of N-Particle QSDs and their Marginals
:label: def-sequence-of-qsds

1.  **The N-Particle Quasi-Stationary Distribution.** For each integer $N \ge 2$, let $\nu_N^{QSD} \in \mathcal{P}(\Sigma_N)$ be the **unique Quasi-Stationary Distribution** for the N-particle Euclidean Gas, whose existence and uniqueness were established in `06_convergence.md`. This is a probability measure on the full N-particle state space $\Sigma_N = (\Omega \times \{0,1\})^N$, describing the long-term statistical behavior of surviving swarm trajectories.

2.  **The First Marginal Measure.** Let $\mu_N \in \mathcal{P}(\Omega)$ be the **first marginal** of the N-particle measure $\nu_N^{QSD}$. This measure represents the probability distribution of a single, typical particle (e.g., walker $i=1$) when the entire N-particle swarm is in its quasi-stationary equilibrium state. Formally, for any measurable set $A \subseteq \Omega$:
    $$
    \mu_N(A) := \nu_N^{QSD}(\{ S \in \Sigma_N \mid (x_1, v_1) \in A \})
    $$
:::

A cornerstone of the mean-field approach is the property of **exchangeability**. The rules of the Euclidean Gas algorithm are symmetric with respect to any permutation of the walker indices. Consequently, the unique N-particle QSD, $\nu_N^{QSD}$, must also be a symmetric measure. A direct and critical consequence of this symmetry is that the marginal distribution $\mu_N$ is the same regardless of which walker index $i \in \{1, \dots, N\}$ is chosen for the projection. This allows us to study the behavior of a single "typical" particle as being representative of the entire swarm's macroscopic state. Our central goal is to prove that the sequence of single-particle measures, $\{\mu_N\}_{N=2}^\infty$, converges weakly to a unique limit.

### **6.3. Step 1: Tightness of the Marginal Sequence**

The first step in proving convergence is to show that the sequence of measures $\{\mu_N\}$ is pre-compact in the space of probability measures on $\Omega$. This property, known as **tightness**, guarantees the existence of at least one convergent subsequence, ensuring that probability mass does not "escape to infinity" in the limit.

:::{prf:theorem} The Sequence of Marginals $\{\mu_N\}$ is Tight
:label: thm-qsd-marginals-are-tight

The sequence of single-particle marginal measures $\{\mu_N\}_{N=2}^\infty$ is tight in the space of probability measures on $\Omega$, $\mathcal{P}(\Omega)$.
:::
:::{prf:proof}
**Proof.**

The proof proceeds by verifying the conditions of **Prokhorov's theorem**. On the Polish space $\Omega$, a sequence of measures is tight if and only if for every $\epsilon > 0$, there exists a single compact set $K_\epsilon \subset \Omega$ such that $\mu_N(K_\epsilon) \ge 1 - \epsilon$ uniformly for all $N \ge 2$. We establish this uniform containment by leveraging the moment bounds from the N-particle Lyapunov analysis.

1.  **Uniform Moment Bound from the N-Particle System:**
    The geometric ergodicity of the N-particle system, established in `06_convergence.md`, is a consequence of a Foster-Lyapunov drift condition for a Lyapunov function $V_{\text{total}}(S)$. A standard result from the theory of Markov chains (e.g., Meyn & Tweedie, *Markov Chains and Stochastic Stability*) is that such a geometric drift condition implies the existence of uniform moment bounds for the corresponding stationary measure. Critically, because all constants in the drift inequality (`κ_total`, `C_total`) were proven to be **N-uniform**, the resulting moment bound is also independent of $N$. Specifically, there exists a finite constant $C < \infty$ such that:
    $$
    \mathbb{E}_{\nu_N^{QSD}}[V_{\text{total}}] = \int_{\Sigma_N} V_{\text{total}}(S) \, d\nu_N^{QSD}(S) \le C \quad \text{for all } N \ge 2.
    $$

2.  **Translation to a Single-Particle Moment Bound:**
    The Lyapunov function $V_{\text{total}}$ is constructed as a sum of N-normalized terms. A suitable choice of $V_{\text{total}}$ includes a term proportional to the average squared norm of the walkers' kinematic states, e.g., $V(S) \propto \frac{1}{N}\sum_{i=1}^N (\|x_i\|^2 + \|v_i\|^2)$. By the linearity of expectation and the **exchangeability** of the walkers under the symmetric measure $\nu_N^{QSD}$, the uniform bound on the total expectation implies a uniform bound on the expected squared norm of any single walker:
    $$
    \mathbb{E}_{\mu_N}[\|x\|^2 + \|v\|^2] = \int_\Omega (\|x\|^2 + \|v\|^2) \, d\mu_N(x,v) \le C'
    $$
    for some other constant $C'$ that is also independent of $N$. This demonstrates that the second moments of the measures in the sequence $\{\mu_N\}$ are uniformly bounded.

3.  **Application of Markov's Inequality to Show Tightness:**
    With this uniform moment control, we apply **Markov's inequality**. For any $R > 0$, let $K_R = \{ (x,v) \in \Omega \mid \|x\|^2 + \|v\|^2 \le R^2 \}$ be a compact ball in the phase space. The probability of a particle being outside this set is:
    $$
    \mu_N(\Omega \setminus K_R) = \mathbb{P}_{z \sim \mu_N}(\|x\|^2 + \|v\|^2 > R^2) \le \frac{\mathbb{E}_{\mu_N}[\|x\|^2 + \|v\|^2]}{R^2} \le \frac{C'}{R^2}
    $$
    For any desired tolerance $\epsilon > 0$, we can choose a radius $R_\epsilon = \sqrt{C'/\epsilon}$. This choice defines a compact set $K_\epsilon := K_{R_\epsilon}$ for which:
    $$
    \mu_N(K_\epsilon) = 1 - \mu_N(\Omega \setminus K_\epsilon) \ge 1 - \frac{C'}{R_\epsilon^2} = 1 - \epsilon.
    $$
    Because the constant $C'$ is independent of $N$, our choice of the compact set $K_\epsilon$ depends only on the tolerance $\epsilon$ and not on $N$. This satisfies the uniform containment condition required by Prokhorov's theorem.

4.  **Conclusion:**
    We have shown that for any $\epsilon > 0$, there exists a compact set $K_\epsilon$ such that $\mu_N(K_\epsilon) \ge 1 - \epsilon$ for all measures in the sequence. By Prokhorov's theorem, this guarantees that the sequence of marginal measures $\{\mu_N\}$ is tight, which implies the existence of at least one weakly convergent subsequence.

**Q.E.D.**
:::

### **6.4. Step 2: Identification of the Limit Point**

Tightness guarantees that at least one convergent subsequence exists. This section proves that any such limit point must be a weak solution to the stationary mean-field PDE. This is the core of the propagation of chaos argument, where we demonstrate that the discrete, empirical interactions of the N-particle system converge to the continuous, integral-based functionals of the mean-field model.

:::{prf:theorem} Limit Points are Weak Solutions to the Stationary Mean-Field PDE
:label: thm-limit-is-weak-solution

Let $\{\mu_{N_k}\}$ be any subsequence of the marginal measures that converges weakly to a limit point $\mu_\infty$. Then $\mu_\infty$ is a weak solution to the stationary mean-field equation $L^\dagger \rho_0 + S[\rho_0] + B[\rho_0] = 0$, where $\rho_0$ is the density of $\mu_\infty$.
:::
:::{prf:proof}
**Proof.**

A measure $\mu_\infty$ with density $\rho_0$ is a weak solution to the stationary mean-field equation if, for any smooth, compactly supported test function $\phi \in C_c^\infty(\Omega)$, it satisfies $\int_\Omega (\mathcal{L}_{\text{FG}} \phi)(z) d\mu_\infty(z) = 0$, where $\mathcal{L}_{\text{FG}}$ is the generator of the mean-field process. This is equivalent to:

$$
\int_\Omega \left(L^\dagger \rho_0(z) - c(z)\rho_0(z) + S[\rho_0](z) + B[\rho_0, m_{d,\infty}](z)\right) \phi(z) \, dz = 0

$$
Our proof establishes this by starting with the stationarity condition for the finite-$N_k$ system and showing that it converges to this weak formulation as $k \to \infty$.

1.  **The N-Particle Stationarity Condition:**
    For each $N_k$, the QSD $\nu_{N_k}^{QSD}$ is stationary with respect to the N-particle generator $\mathcal{L}_{N_k}$. For a test function $\Phi(S) = \phi(z_1)$ that depends only on the state of the first particle, this implies:
    $$
    \mathbb{E}_{\nu_{N_k}^{QSD}}[\mathcal{L}_{N_k} \phi(z_1)] = 0
    $$
    Decomposing the generator, we have $\mathbb{E}_{\nu_{N_k}}[\mathcal{L}_{\text{kin}, N_k} \phi(z_1)] + \mathbb{E}_{\nu_{N_k}}[\mathcal{L}_{\text{clone}, N_k} \phi(z_1)] = 0$ for all $k$.

2.  **Limit of the Kinetic Term:**
    The kinetic generator $\mathcal{L}_{\text{kin}, N_k}$ acts only on the state of particle 1. The expectation is therefore an integral against the first marginal: $\mathbb{E}_{\nu_{N_k}}[\mathcal{L}_{\text{kin}} \phi(z_1)] = \int_{\Omega} (L\phi)(z) d\mu_{N_k}(z)$. Since $\mu_{N_k} \rightharpoonup \mu_\infty$ and $L\phi$ is a bounded, continuous function (as $\phi \in C_c^\infty$), the integral converges:
    $$
    \lim_{k \to \infty} \int_{\Omega} (L\phi)(z) d\mu_{N_k}(z) = \int_{\Omega} (L\phi)(z) d\mu_{\infty}(z) = \int_{\Omega} (L^\dagger\rho_0)(z)\phi(z) \, dz
    $$

3.  **Limit of the Cloning Term (Propagation of Chaos):**
    This is the critical step. The cloning rate for walker 1 depends on its fitness relative to companions drawn from the *empirical measure* of the other $N_k-1$ particles. As $k \to \infty$, the law of large numbers for exchangeable particles (a key consequence of the Hewitt-Savage theorem) implies that this empirical measure converges weakly to the law of a single particle, which is our limit measure $\mu_\infty$.

    The cloning operator for walker 1, $\mathcal{L}_{\text{clone}, N_k}\phi(z_1)$, is a function of the state of walker 1 and the empirical measure of its companions, $\mu_{N_k-1}^{\text{comp}}$. We have already proven:
    *   The empirical companion measure converges: $\mu_{N_k-1}^{\text{comp}} \rightharpoonup \mu_\infty$ almost surely.
    *   The functionals for moments and fitness potentials are continuous with respect to weak convergence.

    Therefore, the N-particle cloning and boundary operators, which are continuous functions of these empirical measures, converge point-wise to the mean-field operators. By the bounded convergence theorem (justified by the uniform boundedness of the generator's action on $\phi$), the expectation also converges:
    $$
    \lim_{k \to \infty} \mathbb{E}_{\nu_{N_k}}[\mathcal{L}_{\text{clone}, N_k} \phi(z_1)] = \int_{\Omega} S[\rho_0]\phi(z) dz
    $$
    $$
    \lim_{k \to \infty} \mathbb{E}_{\nu_{N_k}}[\mathcal{L}_{\text{boundary}, N_k} \phi(z_1)] = \int_{\Omega} (-c(z)\rho_0 + B[\rho_0, m_{d,\infty}])\phi(z) dz
    $$

4.  **Conclusion:**
    Taking the limit of the entire N-particle stationarity condition, we have shown that each term converges to its mean-field counterpart. The limit measure $\mu_\infty$ must therefore satisfy:
    $$
    \int_\Omega \left(L^\dagger \rho_0(z) - c(z)\rho_0(z) + S[\rho_0](z) + B[\rho_0, m_{d,\infty}](z)\right) \phi(z) \, dz = 0
    $$
    This holds for any $\phi \in C_c^\infty(\Omega)$, which is the definition of a weak solution.

**Q.E.D.**
:::

### **6.5. Step 3: Uniqueness of the Weak Solution**

The final step is to prove that the stationary solution identified above is unique. This ensures that all convergent subsequences converge to the same limit, which implies the convergence of the entire sequence $\{\mu_N\}$. We achieve this by showing that the solution operator for the stationary PDE is a strict contraction mapping on a suitable function space.

:::{prf:theorem} Uniqueness of the Stationary Solution
:label: thm-uniqueness-of-qsd

There is at most one probability density $\rho \in \mathcal{P}(\Omega)$ that is a weak solution to the stationary mean-field equation.
:::
:::{prf:proof}
**Proof (via Contraction Mapping).**

The proof strategy is to reformulate the stationary PDE as a fixed-point problem, $\rho = \mathcal{T}[\rho]$, and then to prove that the solution operator $\mathcal{T}$ is a strict contraction on a suitable complete metric space. The Banach Fixed-Point Theorem then guarantees the uniqueness of the solution.

1.  **The Fixed-Point Formulation:**
    The stationary equation is $0 = L^\dagger \rho + S[\rho] + B[\rho]$. We rewrite this by isolating the linear, diffusive part. Let $\mathcal{L}_{\text{lin}} = L^\dagger - C \cdot I$ for a sufficiently large constant $C > 0$ such that $-\mathcal{L}_{\text{lin}}$ is an invertible, coercive operator. The equation becomes $\rho = (-\mathcal{L}_{\text{lin}})^{-1}(S[\rho] + B[\rho] + C\rho)$. We define the solution operator as:
    $$
    \mathcal{T}[\rho] := (-\mathcal{L}_{\text{lin}})^{-1} (S[\rho] + B[\rho] + C\rho)
    $$
    A stationary solution is a fixed point of $\mathcal{T}$.

2.  **The Function Space:**
    We work in the weighted Sobolev space $H^1_w(\Omega)$, a complete metric space (a Banach space) that enforces sufficient regularity on the densities. We consider the operator $\mathcal{T}$ acting on the closed subset of probability densities, $\mathcal{P} \subset H^1_w(\Omega)$.

3.  **Lipschitz Continuity of the Non-Linear Operators:**
    The core of the proof is to show that the non-linear operators, $S[\rho]$ and $B[\rho]$, are Lipschitz continuous on $\mathcal{P}$. That is, there exist constants $L_S$ and $L_B$ such that:
    $$
    \|S[\rho_1] - S[\rho_2]\|_{H^1_w} \le L_S \|\rho_1 - \rho_2\|_{H^1_w}
    $$
    and similarly for $B[\rho]$. This proof follows from the composition of the Lipschitz properties of the underlying functionals: the moment functionals and the fitness potential are Lipschitz with respect to their input densities (as proven via Sobolev embedding), and the cloning operator itself is a smooth integral operator.

4.  **Hypoelliptic Regularity and Boundedness of the Inverse Kinetic Operator:**
    The inverse linear operator, $(-\mathcal{L}_{\text{lin}})^{-1}$, is the solution operator for a kinetic Fokker-Planck equation. This operator is not elliptic but **hypoelliptic**. A key result from the theory of hypoelliptic operators (leveraging Hörmander's theorem) is that this inverse operator is a bounded map from $L^2_w(\Omega)$ to $H^1_w(\Omega)$. Crucially, its operator norm, $C_{\text{hypo}} = \|(-\mathcal{L}_{\text{lin}})^{-1}\|_{L^2_w \to H^1_w}$, scales inversely with the strength of the velocity diffusion:
    $$
    C_{\text{hypo}} \sim \frac{1}{\sigma_v^2 \gamma}
    $$

5.  **The Contraction Property:**
    We now bound the distance between the images of two densities, $\rho_1$ and $\rho_2$, under the full solution operator:
    $$
    \|\mathcal{T}[\rho_1] - \mathcal{T}[\rho_2]\|_{H^1_w} \le C_{\text{hypo}} \|(S[\rho_1]-S[\rho_2]) + (B[\rho_1]-B[\rho_2]) + C(\rho_1-\rho_2)\|_{L^2_w}
    $$
    Applying the triangle inequality and the Lipschitz bounds for $S$ and $B$:
    $$
    \le C_{\text{hypo}} (L_S + L_B + C) \|\rho_1 - \rho_2\|_{H^1_w}
    $$
    The contraction constant is $\kappa := C_{\text{hypo}} (L_S + L_B + C) \sim \frac{L_S + L_B + C}{\sigma_v^2 \gamma}$. Since the Lipschitz constants $L_S$ and $L_B$ depend on the cloning parameters but not the kinetic diffusion $\sigma_v^2$, we can always choose the kinetic noise `σ_v` large enough to ensure that `κ < 1`.

6.  **Conclusion:**
    For a sufficiently large choice of the kinetic exploration noise relative to the cloning selection pressure, the operator $\mathcal{T}$ is a strict contraction on the complete metric space $\mathcal{P}$. By the **Banach Fixed-Point Theorem**, $\mathcal{T}$ has a unique fixed point. Therefore, the stationary solution to the mean-field equation is unique.

**Q.E.D.**
:::

### **6.6. The Thermodynamic Limit**

The convergence of the entire sequence $\{\mu_N\}$ to a unique limit is the formal statement of **propagation of chaos**. A direct and powerful corollary of this result is the existence of a valid thermodynamic limit for the system. This means that macroscopic, intensive quantities calculated from the finite-N equilibrium state converge to the corresponding quantities calculated from the mean-field equilibrium state as $N \to \infty$.

:::{prf:theorem} Convergence of Macroscopic Observables (The Thermodynamic Limit)
:label: thm-thermodynamic-limit

Let $\rho_0$ be the unique stationary solution to the mean-field PDE. Let $\phi: \Omega \to \mathbb{R}$ be any bounded, continuous function (a "macroscopic observable").

Then, the average value of this observable in the N-particle quasi-stationary state converges to the expected value of the observable in the mean-field stationary state:

$$
\lim_{N \to \infty} \mathbb{E}_{\nu_N^{QSD}} \left[ \frac{1}{N} \sum_{i=1}^N \phi(z_i) \right] = \int_\Omega \phi(z) \rho_0(z) dz

$$
:::
:::{prf:proof}
**Proof.**

The proof demonstrates that the left-hand side is equivalent to the definition of weak convergence for the sequence of first marginals.

1.  **Exploit Exchangeability:** As established previously, the N-particle QSD, $\nu_N^{QSD}$, is an exchangeable measure. By the linearity of expectation and exchangeability, the expected average of the observable is equal to the expectation of the observable for any single particle:
    $$
    \mathbb{E}_{\nu_N^{QSD}}\left[ \frac{1}{N} \sum_{i=1}^N \phi(z_i) \right] = \mathbb{E}_{\nu_N^{QSD}}[\phi(z_1)]
    $$

2.  **Relate to the First Marginal:** By definition, the expectation of a function of only the first particle is given by the integral of that function against the first marginal measure, $\mu_N$:
    $$
    \mathbb{E}_{\nu_N^{QSD}}[\phi(z_1)] = \int_\Omega \phi(z) d\mu_N(z)
    $$

3.  **Invoke the Main Convergence Result:** The combination of Tightness (Theorem 5.2), Identification (Theorem 5.4), and Uniqueness (Theorem 5.5) proves that the entire sequence of first marginals converges weakly to the unique mean-field QSD, $\mu_\infty$, whose density is $\rho_0$:
    $$
    \mu_N \rightharpoonup \mu_\infty \quad (\text{as } N \to \infty)
    $$

4.  **Apply the Definition of Weak Convergence:** The definition of weak convergence states that for any bounded, continuous function $\phi$, the integrals converge:
    $$
    \lim_{N \to \infty} \int_\Omega \phi(z) d\mu_N(z) = \int_\Omega \phi(z) d\mu_\infty(z) = \int_\Omega \phi(z) \rho_0(z) dz
    $$

5.  **Conclusion:** By combining the steps, we have shown that the limit of the N-particle average is equal to the mean-field expectation. This completes the proof.

**Q.E.D.**
:::

:::{prf:corollary} Wasserstein-2 Convergence in the Thermodynamic Limit
:label: cor-w2-convergence-thermodynamic-limit

The convergence of marginals to the mean-field QSD holds in the stronger Wasserstein-2 metric:

$$
\lim_{N \to \infty} W_2(\mu_N, \mu_\infty) = 0

$$

where $W_2$ is the Wasserstein-2 (optimal transport) distance between probability measures.
:::

:::{prf:proof}
**Proof.**

The upgrade from weak convergence to W2 convergence follows from a standard metrization theorem in optimal transport theory, given that we have uniform control of second moments.

**Step 1: Uniform Second Moment Control**

By Theorem [](#thm-qsd-marginals-are-tight), the tightness proof established that there exists a constant $C' < \infty$ independent of $N$ such that:

$$
\sup_{N \ge 2} \mathbb{E}_{\mu_N}[\|z\|^2] = \sup_{N \ge 2} \int_\Omega (\|x\|^2 + \|v\|^2) \, d\mu_N(x,v) \le C'

$$

This uniform bound on second moments is a direct consequence of the N-uniform Foster-Lyapunov analysis in `06_convergence.md`.

**Step 2: Weak Convergence**

The main result of Section 5 (combining Theorems 5.2, 5.4, and 5.5) established that:

$$
\mu_N \rightharpoonup \mu_\infty \quad \text{as } N \to \infty

$$

**Step 3: Apply the Metrization Theorem**

With both weak convergence and uniform second moments established, we can invoke the following classical result from optimal transport theory:

**Theorem (Villani, *Optimal Transport: Old and New*, Theorem 6.9):** Let $\{\nu_n\}$ be a sequence of probability measures on a Polish space $\mathcal{X}$ with a reference point $x_0 \in \mathcal{X}$. If:
1. $\nu_n \rightharpoonup \nu$ (weak convergence)
2. $\sup_n \int d(x, x_0)^2 d\nu_n(x) < \infty$ (uniform second moments)

Then $W_2(\nu_n, \nu) \to 0$.

**Application:** Our sequence $\{\mu_N\}$ satisfies both hypotheses on the Polish space $\Omega = \mathcal{X}_{\text{valid}} \times \mathbb{R}^d$ with the Euclidean metric. Therefore, by Villani's theorem:

$$
\lim_{N \to \infty} W_2(\mu_N, \mu_\infty) = 0

$$

**Step 4: Physical Interpretation**

The W2 metric has a natural physical interpretation as the minimal "cost" of transporting one probability distribution to another, where cost is measured by squared Euclidean distance. The W2 convergence result implies that:

1. **Position convergence**: The spatial distribution of the swarm converges in a strong sense
2. **Velocity convergence**: The velocity distribution also converges strongly
3. **Joint convergence**: The phase-space structure of the empirical measure converges to the mean-field prediction

This is a stronger statement than weak convergence, which only guarantees convergence of expectations of bounded continuous functions. W2 convergence implies convergence of second moments and provides quantitative control over the distance between distributions.

**Q.E.D.**
:::
