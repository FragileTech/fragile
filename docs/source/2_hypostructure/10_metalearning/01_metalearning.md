# The Theory of Learning

*The Agent, The Loss, and The Solver.*

:::{div} feynman-prose
Now we come to what I think is one of the most beautiful parts of this whole story. Up until now, we have been treating the axioms of hypostructure as fixed things---given to us by the structure of the problem. But here is the question that any honest person must eventually ask: how do we *find* these axioms in the first place?

Think about what happens when you encounter a new physical system. You observe trajectories, you measure energy flows, you see patterns. Somewhere in all that data, there is structure waiting to be discovered. The question is: can we build a machine that discovers it automatically?

The answer is yes, and the method is surprisingly simple. We turn the axioms themselves into learnable parameters, and we train them by minimizing how badly they fail to hold. If a candidate energy functional does not decrease along trajectories, that is a measurable error. If the dissipation does not match the entropy production, that is another error. Stack up all these errors, and you have a loss function. Minimize it, and the axioms emerge.

This is not just a clever trick. It is a fundamental statement about what scientific theories *are*. A theory is whatever minimizes the gap between prediction and observation, subject to structural constraints. The hypostructure axioms are those constraints. Meta-learning is the process of discovering them.
:::

(ch-meta-learning)=

## Meta-Learning Axioms (The L-Layer)

In previous chapters, each soft axiom $A$ was associated with a defect functional $K_A : \mathcal{U} \to [0,\infty]$ defined on a class $\mathcal{U}$ of trajectories. The value $K_A(u)$ quantifies the extent to which axiom $A$ fails along trajectory $u$, and vanishes when the axiom is exactly satisfied.

In this chapter, the axioms themselves are treated as objects to be chosen: each axiom is specified by a family of global parameters, and these parameters are determined as minimizers of defect functionals. Global axioms are obtained as minimizers of the defects of their local soft counterparts.

### Parametric families of axioms

:::{prf:definition} Parameter space
:label: def-parameter-space

Let $\Theta$ be a metric space (typically a subset of a finite-dimensional vector space $\mathbb{R}^d$). A **parametric axiom family** is a collection $\{A_\theta\}_{\theta \in \Theta}$ where each $A_\theta$ is a soft axiom instantiated by global data depending on $\theta$.
:::

:::{prf:definition} Parametric hypostructure components
:label: def-parametric-hypostructure-components

For each $\theta \in \Theta$, define the following *co-equal* components of the hypostructure (none is auxiliary; boundary data is on the same footing as $\Phi_\theta$ or $\mathfrak{D}_\theta$):

- **Parametric height functional:** $\Phi_\theta : X \to \mathbb{R}$
- **Parametric dissipation:** $\mathfrak{D}_\theta : X \to [0,\infty]$
- **Parametric symmetry group:** $G_\theta \subset \mathrm{Aut}(X)$
- **Parametric local structures:** metrics, norms, or capacities depending on $\theta$
- **Boundary interface:** a boundary object $\mathcal{B}_\theta$, trace morphism $\mathrm{Tr}_\theta: X \to \mathcal{B}_\theta$, flux morphism $\mathcal{J}_\theta: \mathcal{B}_\theta \to \underline{\mathbb{R}}$, and reinjection kernel $\mathcal{R}_\theta: \mathcal{B}_\theta \to \mathcal{P}(X)$ as in {prf:ref}`def-categorical-hypostructure` and {prf:ref}`def-thin-objects`

:::

The tuple $\mathbb{H}_\theta = (X, S_t, \Phi_\theta, \mathfrak{D}_\theta, G_\theta, \mathcal{B}_\theta, \mathrm{Tr}_\theta, \mathcal{J}_\theta, \mathcal{R}_\theta)$ is a **parametric hypostructure**, with boundary data treated as first-class structure rather than a peripheral constraint.

:::{prf:remark} Boundary object consistency
The boundary interface here uses the categorical hypostructure framework: $\mathcal{B}_\theta$ is the boundary data object, $\mathrm{Tr}_\theta$ is the trace/restriction morphism (categorically the counit of $\iota_! \dashv \iota^*$), $\mathcal{J}_\theta$ measures boundary flux, and $\mathcal{R}_\theta$ encodes reinjection/feedback. This is the meta-learning counterpart of the boundary axiom {prf:ref}`ax-boundary` and the open-system check {prf:ref}`def-node-boundary`.
:::

:::{prf:remark} Boundary parity principle
All reconstruction, identifiability, and risk statements in this chapter treat boundary structure on equal footing with bulk structure. In particular, two parameter choices that agree on $(\Phi_\theta, \mathfrak{D}_\theta, G_\theta)$ but disagree on $(\mathcal{B}_\theta, \mathrm{Tr}_\theta, \mathcal{J}_\theta, \mathcal{R}_\theta)$ are **distinct hypostructures** and remain distinguishable by $K_{Bound}$.
:::

:::{prf:definition} Parametric defect functional
:label: def-parametric-defect-functional

For each $\theta \in \Theta$ and each soft axiom label $A \in \mathcal{A} = \{\text{C}, \text{D}, \text{SC}, \text{Cap}, \text{LS}, \text{TB}, \text{Bound}\}$, define the defect functional:

$$
K_A^{(\theta)} : \mathcal{U} \to [0,\infty]
$$

constructed from the hypostructure $\mathbb{H}_\theta$ and the local definition of axiom $A$.
:::

:::{prf:lemma} Defect characterization
:label: lem-defect-characterization

For all $\theta \in \Theta$ and $u \in \mathcal{U}$:

$$
K_A^{(\theta)}(u) = 0 \quad \Longleftrightarrow \quad \text{trajectory } u \text{ satisfies } A_\theta \text{ exactly.}
$$

Small values of $K_A^{(\theta)}(u)$ correspond to small violations of axiom $A_\theta$.
:::

:::{prf:proof}
We verify the characterization for each axiom $A \in \mathcal{A}$:

**(C) Compatibility:** $K_C^{(\theta)}(u) := \|S_t(u(s)) - u(s+t)\|$ for appropriate $s, t \in T$. This equals zero if and only if $u$ is a trajectory of the semiflow.

**(D) Dissipation:** $K_D^{(\theta)}(u) := \int_T \max(0, \partial_t \Phi_\theta(u(t)) + \mathfrak{D}_\theta(u(t))) dt$. This equals zero if and only if $\partial_t \Phi_\theta + \mathfrak{D}_\theta \leq 0$ holds pointwise along $u$.

**(SC) Symmetry Compatibility:** $K_{SC}^{(\theta)}(u) := \sup_{g \in G_\theta} \sup_{t \in T} d(g \cdot u(t), S_t(g \cdot u(0)))$. This equals zero if and only if the semiflow commutes with the $G_\theta$-action along $u$.

**(Cap) Capacity Bounds:** $K_{Cap}^{(\theta)}(u) := \int_T |\text{cap}(\{u(t)\}) - \mathfrak{D}_\theta(u(t))| dt$ (or analogous comparison). Vanishes when capacity and dissipation agree.

**(LS) Local Structure:** $K_{LS}^{(\theta)}(u)$ measures deviations from local metric, norm, or regularity assumptions as specified in previous chapters.

**(TB) Thermodynamic Bounds:** $K_{TB}^{(\theta)}(u)$ measures violations of data processing inequalities or entropy bounds.

**(Bound) Boundary interface:** $K_{Bound}^{(\theta)}(u)$ measures violations of boundary compatibility in the sense of the boundary object $\mathcal{B}_\theta$, trace morphism $\mathrm{Tr}_\theta$, flux $\mathcal{J}_\theta$, and reinjection kernel $\mathcal{R}_\theta$. Concretely, $K_{Bound}^{(\theta)}(u) = 0$ when:
1. The trace $\mathrm{Tr}_\theta(u(t))$ is well-defined for $\mu$-a.e. $t$ and matches the boundary restriction encoded by $\mathcal{B}_\theta$,
2. The boundary flux balances dissipation (e.g., $\frac{d}{dt}\Phi_\theta(u(t)) + \mathfrak{D}_\theta(u(t)) = -\mathcal{J}_\theta(\mathrm{Tr}_\theta(u(t)))$ in the classical setting), and
3. Any reinjection is represented by $\mathcal{R}_\theta$ (Dirichlet/Neumann/Feller cases as in {prf:ref}`def-node-boundary`).

In each case, $K_A^{(\theta)}(u) \geq 0$ with equality if and only if the constraint is satisfied exactly.
:::

### Global defect functionals and defect risk

:::{prf:definition} Trajectory measure
:label: def-trajectory-measure

Let $\mu$ be a $\sigma$-finite measure on the trajectory space $\mathcal{U}$. This measure describes how trajectories are sampled or weighted—for instance, a law induced by initial conditions and the evolution $S_t$, or an empirical distribution of observed trajectories.
:::

:::{prf:definition} Expected defect
:label: def-expected-defect

For each axiom $A \in \mathcal{A}$ and parameter $\theta \in \Theta$, define the **expected defect**:

$$
\mathcal{R}_A(\theta) := \int_{\mathcal{U}} K_A^{(\theta)}(u) \, d\mu(u)
$$

whenever the integral is well-defined and finite.
:::

:::{prf:definition} Worst-case defect
:label: def-worst-case-defect

For an admissible class $\mathcal{U}_{\text{adm}} \subset \mathcal{U}$, define:

$$
\mathcal{K}_A(\theta) := \sup_{u \in \mathcal{U}_{\text{adm}}} K_A^{(\theta)}(u).
$$

:::

:::{prf:definition} Joint defect risk
:label: def-joint-defect-risk

For a finite family of soft axioms $\mathcal{A}$ with nonnegative weights $(w_A)_{A \in \mathcal{A}}$, define the **joint defect risk**:

$$
\mathcal{R}(\theta) := \sum_{A \in \mathcal{A}} w_A \, \mathcal{R}_A(\theta).
$$

:::

:::{prf:lemma} Interpretation of defect risk
:label: lem-interpretation-of-defect-risk

The quantity $\mathcal{R}_A(\theta)$ measures the global quality of axiom $A_\theta$:

- Small values indicate that, on average with respect to $\mu$, axiom $A_\theta$ is nearly satisfied.
- Large values indicate frequent or severe violations on a set of nontrivial $\mu$-measure.

:::

:::{prf:proof}
By {prf:ref}`def-expected-defect`, $\mathcal{R}_A(\theta) = \int_{\mathcal{U}} K_A^{(\theta)}(u) \, d\mu(u)$. Since $K_A^{(\theta)}(u) \geq 0$ with equality precisely when trajectory $u$ satisfies axiom $A$ under parameter $\theta$ ({prf:ref}`def-parametric-defect-functional`), we have:

1. **Small $\mathcal{R}_A(\theta)$:** For any $\varepsilon > 0$, Markov's inequality gives

   $$
   \mu\big(\{u : K_A^{(\theta)}(u) > \varepsilon\}\big) \leq \frac{\mathcal{R}_A(\theta)}{\varepsilon}.
   $$

   Thus small $\mathcal{R}_A(\theta)$ forces the set of trajectories with defect above $\varepsilon$ to have small $\mu$-measure, i.e., the axiom is nearly satisfied in average and in measure.

2. **Large $\mathcal{R}_A(\theta)$:** If $K_A^{(\theta)}$ is bounded below by a positive constant on a set of positive $\mu$-measure (frequent or severe violations), then the integral is large. More generally, if violations are frequent or large in magnitude, the integral grows.

The interpretation follows from nonnegativity of $K_A^{(\theta)}$ and standard measure bounds; no pointwise guarantees are claimed without stronger assumptions.
:::

#### The Meta-Objective Functional

The joint defect risk $\mathcal{R}(\theta)$ admits a variational interpretation. We introduce the **Meta-Objective Functional** and the **Principle of Least Structural Defect**.

:::{prf:definition} Meta-Objective Functional
:label: def-meta-action-functional

Define the **Meta-Objective** $\mathcal{S}_{\text{meta}}: \Theta \to \mathbb{R}$ as:
$$
\mathcal{S}_{\text{meta}}(\theta) := \int_{\text{System Space}} \left(
\underbrace{\mathcal{L}_{\text{fit}}(\theta, u)}_{\text{Data Fit Term}} +
\underbrace{\lambda \sum_{A \in \mathcal{A}} w_A K_A^{(\theta)}(u)^2}_{\text{Structural Penalty Term}}
\right) d\mu_{\text{sys}}(u)
$$
where:

- $\mathcal{L}_{\text{fit}}(\theta, u)$ measures empirical fit (data fitting term),
- $K_A^{(\theta)}(u)^2$ measures structural violation (structural penalty term),
- $\lambda > 0$ is a regularization constant balancing fit and structure.

:::

:::{prf:observation} Principle of Least Structural Defect
:label: obs-least-structural-defect

The optimal axiom parameters $\theta^*$ minimize the Meta-Objective:
$$
\theta^* = \arg\min_{\theta \in \Theta} \mathcal{S}_{\text{meta}}(\theta).
$$
:::

*Interpretation:* The learning process converges to a **stable configuration in theory space**---a parameter setting where structural constraints are satisfied while fitting the observed data.

:::{prf:proposition} Variational Characterization
:label: prop-variational-characterization

Under the assumptions of {prf:ref}`mt-existence-of-defect-minimizers`, the global defect minimizer $\theta^*$ satisfies the variational equation:
$$
\nabla_\theta \mathcal{S}_{\text{meta}}(\theta^*) = 0.
$$
Moreover, if $\mathcal{S}_{\text{meta}}$ is strictly convex, $\theta^*$ is unique.
:::

:::{prf:proof}
By {prf:ref}`mt-existence-of-defect-minimizers`, $\theta^*$ exists. If $\theta^*$ is an interior point of $\Theta$, the first-order necessary condition is $\nabla_\theta \mathcal{S}_{\text{meta}}(\theta^*) = 0$. Strict convexity implies uniqueness by standard arguments.
:::

### Trainable global permits

:::{prf:definition} Global defect minimizer
:label: def-global-defect-minimizer

A point $\theta^* \in \Theta$ is a **global defect minimizer** if:

$$
\mathcal{R}(\theta^*) = \inf_{\theta \in \Theta} \mathcal{R}(\theta).
$$

:::

:::{prf:metatheorem} Existence of Defect Minimizers
:label: mt-existence-of-defect-minimizers

Assume:

1. The parameter space $\Theta$ is compact and metrizable.
2. For each $A \in \mathcal{A}$ and each $u \in \mathcal{U}$, the map $\theta \mapsto K_A^{(\theta)}(u)$ is continuous on $\Theta$.
3. There exists an integrable majorant $M_A \in L^1(\mu)$ such that $0 \leq K_A^{(\theta)}(u) \leq M_A(u)$ for all $\theta \in \Theta$ and $\mu$-a.e. $u$.

Then, for each $A \in \mathcal{A}$, the expected defect $\mathcal{R}_A(\theta)$ is finite and continuous on $\Theta$. Consequently, the joint risk $\mathcal{R}(\theta)$ is continuous and attains its infimum on $\Theta$. There exists at least one global defect minimizer $\theta^* \in \Theta$.
:::

:::{prf:proof}
**Step 1 (Setup).** Let $\theta_n \to \theta$ in $\Theta$. We must show $\mathcal{R}_A(\theta_n) \to \mathcal{R}_A(\theta)$.

**Step 2 (Pointwise convergence).** By assumption (2), for each $u \in \mathcal{U}$:

$$
K_A^{(\theta_n)}(u) \to K_A^{(\theta)}(u).
$$

**Step 3 (Dominated convergence).** By assumption (3), $|K_A^{(\theta_n)}(u)| \leq M_A(u)$ with $M_A \in L^1(\mu)$. The dominated convergence theorem yields:

$$
\mathcal{R}_A(\theta_n) = \int_{\mathcal{U}} K_A^{(\theta_n)}(u) \, d\mu(u) \to \int_{\mathcal{U}} K_A^{(\theta)}(u) \, d\mu(u) = \mathcal{R}_A(\theta).
$$

**Step 4 (Continuity of joint risk).** Since $\mathcal{R}(\theta) = \sum_{A \in \mathcal{A}} w_A \mathcal{R}_A(\theta)$ is a finite sum of continuous functions, it is continuous.

**Step 5 (Existence).** By the extreme value theorem, a continuous function on a compact set attains its infimum. Hence there exists $\theta^* \in \Theta$ with $\mathcal{R}(\theta^*) = \inf_{\theta \in \Theta} \mathcal{R}(\theta)$.
:::

:::{prf:corollary} Characterization of exact minimizers
:label: cor-characterization-of-exact-minimizers

If $\mathcal{R}_A(\theta^*) = 0$ for all $A \in \mathcal{A}$, then all axioms in $\mathcal{A}$ hold $\mu$-almost surely under $A_{\theta^*}$. The hypostructure $\mathbb{H}_{\theta^*}$ satisfies all soft axioms globally.
:::

:::{prf:proof}
If $\mathcal{R}_A(\theta^*) = \int K_A^{(\theta^*)} d\mu = 0$ and $K_A^{(\theta^*)} \geq 0$, then $K_A^{(\theta^*)}(u) = 0$ for $\mu$-a.e. $u$. By {prf:ref}`lem-defect-characterization`, axiom $A_{\theta^*}$ holds $\mu$-almost surely.
:::

### Gradient-based approximation

Assume $\Theta \subset \mathbb{R}^d$ is open and convex.

:::{prf:lemma} Leibniz rule for defect risk
:label: lem-leibniz-rule-for-defect-risk

Assume:

1. For each $A \in \mathcal{A}$ and each $u \in \mathcal{U}$, the map $\theta \mapsto K_A^{(\theta)}(u)$ is differentiable on $\Theta$ with gradient $\nabla_\theta K_A^{(\theta)}(u)$.
2. There exists an integrable majorant $M_A \in L^1(\mu)$ such that $|\nabla_\theta K_A^{(\theta)}(u)| \leq M_A(u)$ for all $\theta \in \Theta$ and $\mu$-a.e. $u$.

Then the gradient of $\mathcal{R}_A$ admits the integral representation:

$$
\nabla_\theta \mathcal{R}_A(\theta) = \int_{\mathcal{U}} \nabla_\theta K_A^{(\theta)}(u) \, d\mu(u).
$$

:::

:::{prf:proof}
**Step 1 (Difference quotient).** For $h \in \mathbb{R}^d$ with $|h|$ small:

$$
\frac{\mathcal{R}_A(\theta + h) - \mathcal{R}_A(\theta)}{|h|} = \int_{\mathcal{U}} \frac{K_A^{(\theta + h)}(u) - K_A^{(\theta)}(u)}{|h|} \, d\mu(u).
$$

**Step 2 (Mean value theorem).** By differentiability, for each $u$:

$$
\frac{K_A^{(\theta + h)}(u) - K_A^{(\theta)}(u)}{|h|} \to \nabla_\theta K_A^{(\theta)}(u) \cdot \frac{h}{|h|}
$$

as $|h| \to 0$.

**Step 3 (Dominated convergence).** The mean value theorem gives:

$$
\left|\frac{K_A^{(\theta + h)}(u) - K_A^{(\theta)}(u)}{|h|}\right| \leq \sup_{\xi \in [\theta, \theta+h]} |\nabla_\theta K_A^{(\xi)}(u)| \leq M_A(u).
$$

By dominated convergence, differentiation passes through the integral.
:::

:::{prf:corollary} Gradient of joint risk
:label: cor-gradient-of-joint-risk

Under the assumptions of {prf:ref}`lem-leibniz-rule-for-defect-risk`:

$$
\nabla_\theta \mathcal{R}(\theta) = \sum_{A \in \mathcal{A}} w_A \int_{\mathcal{U}} \nabla_\theta K_A^{(\theta)}(u) \, d\mu(u).
$$

:::

:::{prf:corollary} Gradient descent convergence
:label: cor-gradient-descent-convergence

Consider the gradient descent iteration:

$$
\theta_{k+1} = \theta_k - \eta_k \nabla_\theta \mathcal{R}(\theta_k)
$$

with step sizes $\eta_k > 0$ satisfying $\sum_k \eta_k = \infty$ and $\sum_k \eta_k^2 < \infty$. Assume in addition that the iterates remain in a compact sublevel set of $\mathcal{R}$ (or, equivalently, that $\mathcal{R}$ has compact sublevel sets and $\mathcal{R}(\theta_k)$ is nonincreasing).
:::

Under the assumptions of {prf:ref}`lem-leibniz-rule-for-defect-risk`, together with Lipschitz continuity of $\nabla_\theta \mathcal{R}$, the sequence $(\theta_k)$ has accumulation points, and every accumulation point is a stationary point of $\mathcal{R}$.

If additionally $\mathcal{R}$ is convex, every accumulation point is a global defect minimizer.

:::{prf:proof}
We apply the Robbins-Monro theorem.

**Step 1 (Descent property).** For $L$-Lipschitz continuous gradients:

$$
\mathcal{R}(\theta_{k+1}) \leq \mathcal{R}(\theta_k) - \eta_k \|\nabla \mathcal{R}(\theta_k)\|^2 + \frac{L\eta_k^2}{2}\|\nabla \mathcal{R}(\theta_k)\|^2.
$$

**Step 2 (Summability).** Summing over $k$ and using $\sum_k \eta_k^2 < \infty$:

$$
\sum_{k=0}^\infty \eta_k(1 - L\eta_k/2)\|\nabla \mathcal{R}(\theta_k)\|^2 \leq \mathcal{R}(\theta_0) - \inf \mathcal{R} < \infty.
$$

Since $\sum_k \eta_k = \infty$ and $\eta_k \to 0$, we have $\liminf_{k \to \infty} \|\nabla \mathcal{R}(\theta_k)\| = 0$.

**Step 3 (Accumulation points).** By the compact sublevel-set assumption, $(\theta_k)$ is precompact and hence has accumulation points. Continuity of $\nabla \mathcal{R}$ implies any accumulation point $\theta^*$ satisfies $\nabla \mathcal{R}(\theta^*) = 0$ (stationary).

**Step 4 (Convex case).** If $\mathcal{R}$ is convex, stationary points satisfy $\nabla \mathcal{R}(\theta^*) = 0$ if and only if $\theta^*$ is a global minimizer.
:::

### Joint training of axioms and extremizers

:::{prf:definition} Two-level parameterization
:label: def-two-level-parameterization

Consider:

- **Hypostructure parameters:** $\theta \in \Theta$ defining $\Phi_\theta, \mathfrak{D}_\theta, G_\theta$
- **Extremizer parameters:** $\vartheta \in \Upsilon$ parametrizing candidate trajectories $u_\vartheta \in \mathcal{U}$

:::

:::{prf:definition} Joint training objective
:label: def-joint-training-objective

Define:

$$
\mathcal{L}(\theta, \vartheta) := \sum_{A \in \mathcal{A}} w_A \, \mathbb{E}[K_A^{(\theta)}(u_\vartheta)] + \sum_{B \in \mathcal{B}} v_B \, \mathbb{E}[F_B^{(\theta)}(u_\vartheta)]
$$

where:

- $\mathcal{A}$ indexes axioms whose defects are minimized
- $\mathcal{B}$ indexes extremal problems whose values $F_B^{(\theta)}(u_\vartheta)$ are optimized

:::

:::{prf:metatheorem} Joint Training Dynamics
:label: mt-joint-training-dynamics

Under differentiability assumptions analogous to {prf:ref}`lem-leibniz-rule-for-defect-risk` for both $\theta$ and $\vartheta$, the objective $\mathcal{L}$ is differentiable in $(\theta, \vartheta)$. The joint gradient descent:

$$
(\theta_{k+1}, \vartheta_{k+1}) = (\theta_k, \vartheta_k) - \eta_k \nabla_{(\theta, \vartheta)} \mathcal{L}(\theta_k, \vartheta_k)
$$

converges to stationary points under standard conditions.
:::

:::{prf:proof}
**Step 1 (Differentiability).** Both $\theta \mapsto K_A^{(\theta)}(u_\vartheta)$ and $\vartheta \mapsto u_\vartheta$ are differentiable by assumption. Chain rule gives differentiability of the composition.

**Step 2 (Integral exchange).** Dominated convergence (as in {prf:ref}`lem-leibniz-rule-for-defect-risk`) allows differentiation under the expectation.

**Step 3 (Convergence).** The same Robbins-Monro analysis as in {prf:ref}`cor-gradient-descent-convergence` applies to the joint iteration on $(\theta, \vartheta) \in \Theta \times \Upsilon$. Under Lipschitz continuity of $\nabla_{(\theta, \vartheta)} \mathcal{L}$ and compactness of $\Theta \times \Upsilon$, the descent inequality holds in the product space. The step size conditions ensure convergence to stationary points of $\mathcal{L}$.
:::

:::{prf:corollary} Interpretation
:label: cor-interpretation

In this scheme:

- The global axioms $\theta$ are **learned** to minimize defects of local soft axioms.
- The extremal profiles $\vartheta$ are simultaneously tuned to probe and saturate the variational problems defined by these axioms.
- The resulting pair $(\theta^*, \vartheta^*)$ consists of a globally adapted hypostructure and representative extremal trajectories within it.

:::

### Thermodynamic Interpretation: Fisher Information and Otto Calculus

The dissipation defect $K_D^{(\theta)}$ admits a rigorous thermodynamic interpretation via **Fisher Information** and **Wasserstein Gradient Flows** (Otto Calculus). This section provides the geometric-thermodynamic foundation for the meta-learning dynamics.

:::{prf:definition} Fisher Information Metric
:label: def-fisher-information

Let $(\mathcal{P}(X), W_2)$ be the Wasserstein space of probability measures on a metric-measure space $(X, d, \mathfrak{m})$. For a curve $\rho_t$ in $\mathcal{P}(X)$ with density $\rho_t(x) = \frac{d\mu_t}{d\mathfrak{m}}(x)$ relative to the reference measure $\mathfrak{m}$, the **Fisher Information** is:

$$
\text{Fisher}(\rho_t \,|\, \mathfrak{m}) := \int_X \left|\nabla \log \frac{\rho_t}{\mathfrak{m}}\right|^2 d\mu_t = \int_X \frac{|\nabla \rho_t|^2}{\rho_t} d\mathfrak{m}
$$

This defines a **Riemannian metric** on $\mathcal{P}(X)$ called the **Wasserstein metric** or **Otto metric**:

$$
g_{\rho}(v, w) = \int_X \langle v, w \rangle d\rho
$$

for tangent vectors $v, w \in T_\rho \mathcal{P}(X)$.

**Interpretation:** The Fisher Information measures the "kinetic energy" of probability flow in the Wasserstein manifold.

**Literature:** {cite}`Otto01` (Wasserstein geometry); {cite}`Villani09` (Optimal transport)
:::

:::{prf:theorem} JKO Scheme and Dissipation
:label: thm-jko-dissipation

Let $\Phi: \mathcal{P}(X) \to \mathbb{R}$ be a free energy functional (e.g., $\Phi[\rho] = \int \rho V d\mathfrak{m} + \int \rho \log \rho d\mathfrak{m}$ for potential $V$). The **Jordan-Kinderlehrer-Otto (JKO) scheme** defines the gradient flow via:

$$
\rho_{t+\tau} = \arg\min_{\rho \in \mathcal{P}(X)} \left\{\Phi[\rho] + \frac{1}{2\tau}W_2^2(\rho, \rho_t)\right\}
$$

where $W_2$ is the Wasserstein-2 distance.

**Dissipation Identity:** The dissipation rate along the gradient flow satisfies:

$$
\frac{d}{dt}\Phi[\rho_t] = -\text{Fisher}(\rho_t \,|\, \mathfrak{m})
$$

This provides the **rigorous link** between:
- **Geometry:** Geodesic motion in $(\mathcal{P}(X), W_2)$
- **Thermodynamics:** Entropy dissipation $\dot{S} = -\text{Fisher}$

**Consequence for Meta-Learning:** The dissipation defect $K_D^{(\theta)}$ should be formulated as:

$$
K_D^{(\theta)}(u) = \left|\frac{d}{dt}\Phi_\theta[u(t)] + \text{Fisher}(u(t) \,|\, \mathfrak{m}_\theta)\right|
$$

This measures the deviation from the "natural" thermodynamic evolution.

**Literature:** {cite}`JordanKinderlehrerOtto98` (JKO scheme); {cite}`AmbrosioGigliSavare08` (Gradient flows in metric spaces)
:::

:::{prf:remark} Upgraded Loss for Learning Agents
:label: rem-upgraded-loss

The user's critique identifies that current "Physicist" agents minimize $\|\Delta z\|^2$ (kinetic energy) without accounting for the **drift induced by measure concentration**. The corrected loss should be:

**Current (Incomplete):**

$$
\mathcal{L}_{\text{old}} = \frac{1}{2\tau}\|\rho_{t+\tau} - \rho_t\|_{L^2}^2 + \text{KL}(\rho_{t+\tau} \,\|\, \mathfrak{m})
$$

**Upgraded (Metric-Measure Correct):**

$$
\mathcal{L}_{\text{new}} = \frac{1}{2\tau}W_2^2(\rho_{t+\tau}, \rho_t) + \Phi[\rho_{t+\tau}]
$$

where the **Wasserstein distance** $W_2$ accounts for both metric geometry and measure concentration.

**Explicit Gradient (Otto Calculus):**
The gradient of $\Phi$ in the Wasserstein manifold is:

$$
\nabla_{W_2}\Phi[\rho] = -\nabla \cdot \left(\rho \nabla \frac{\delta \Phi}{\delta \rho}\right)
$$

For $\Phi[\rho] = \int \rho V + \int \rho \log \rho$, this gives:

$$
\nabla_{W_2}\Phi[\rho] = -\nabla \cdot (\rho \nabla (V + \log \rho))
$$

**Agent Implementation:** The "Physicist" state vector $z_{\text{macro}}$ must include:
1. **Position:** $x \in X$
2. **Density potential:** $S = \log \rho$ (entropy)
3. **Fisher Information:** $\text{Fisher} = \|\nabla S\|^2$

The agent loss becomes:

$$
\mathcal{L}_{\text{Physicist}} = \frac{1}{2\tau}W_2^2(\rho_{t+\tau}, \rho_t) + \Phi[\rho_{t+\tau}] + \lambda_{\text{LSI}}(K_{\text{LSI}}^{-1} - \text{target variance})^2
$$

where the LSI penalty prevents "melting" (measure dispersion).
:::

:::{prf:theorem} No-Melt Theorem (Exponential Convergence)
:label: thm-no-melt

Let $(X, d, \mathfrak{m})$ satisfy $\mathrm{RCD}(K, N)$ with $K > 0$. Let $\rho_t$ be the gradient flow of $\Phi[\rho] = \text{KL}(\rho || \mathfrak{m})$ under the JKO scheme.

**Claim:** The relative entropy decays exponentially:

$$
\text{KL}(\rho_t \,\|\, \mathfrak{m}) \leq e^{-2Kt}\text{KL}(\rho_0 \,\|\, \mathfrak{m})
$$

:::{prf:proof}

By the EVI (Evolution Variational Inequality, Theorem {prf:ref}`thm-rcd-dissipation-link`):

$$
\frac{d}{dt}\text{KL}(\rho_t \,\|\, \mathfrak{m}) + K W_2^2(\rho_t, \mathfrak{m}) + \text{Fisher}(\rho_t \,|\, \mathfrak{m}) \leq 0
$$

Using the **Talagrand inequality** $W_2^2(\rho, \mathfrak{m}) \geq \frac{2}{K}\text{KL}(\rho \,\|\, \mathfrak{m})$ (which holds under $\mathrm{RCD}(K, N)$):

$$
\frac{d}{dt}\text{KL}(\rho_t \,\|\, \mathfrak{m}) + 2K \,\text{KL}(\rho_t \,\|\, \mathfrak{m}) \leq 0
$$

This is a differential inequality with solution:

$$
\text{KL}(\rho_t \,\|\, \mathfrak{m}) \leq e^{-2Kt}\text{KL}(\rho_0 \,\|\, \mathfrak{m})
$$

:::

**Consequence:** An agent satisfying the $\mathrm{RCD}(K, N)$ condition with $K > 0$ **cannot drift indefinitely**. The probability of delusional states (large Wasserstein distance from equilibrium) decays exponentially with compute time.

**Landauer Efficiency:** The thermodynamic cost of maintaining this convergence is:

$$
\Delta S_{\text{min}} = k_B T \ln(2) \cdot K^{-1} \cdot \text{(bits erased)}
$$

This is the **Landauer bound** with constant $K^{-1}$: stronger curvature (larger $K$) enables more efficient computation.

**Literature:** {cite}`OttoVillani00` (Talagrand inequality); {cite}`AmbrosioGigliSavare14` (EVI for RCD spaces)
:::

:::{prf:theorem} Metric Evolution Law (Ricci Flow Analogue for Meta-Learning)
:label: thm-metric-evolution

**Purpose:** This theorem closes the "Dissipation = Curvature Tautology" by proving that geometry **evolves dynamically** in response to dissipation, rather than being defined to equal it.

**Setting:** Let $g_t$ be a time-dependent Riemannian metric on the parameter space $\Theta$, and let $\mathfrak{D}_t$ be the dissipation 2-form measuring entropy production. We seek a **dynamic coupling law** that governs how $g_t$ responds to $\mathfrak{D}_t$.

**The Coupling Law (Discrete-Time Metaregulator Update):**

The meta-learning algorithm updates the metric according to the **Wasserstein gradient flow** of the relative entropy functional:

$$
g_{t+\tau} = \arg\min_{g} \left\{ \text{KL}(\rho_{g} \,\|\, \mathfrak{m}) + \frac{1}{2\tau}W_2^2(g, g_t) + \lambda \int_\Theta \text{Ric}(g) \wedge \mathfrak{D}_t \right\}
$$

where:
- $\text{KL}(\rho_g || \mathfrak{m})$ is the relative entropy of the induced measure under metric $g$
- $W_2(g, g_t)$ is the Wasserstein distance between metrics (in the space of Riemannian structures)
- $\text{Ric}(g)$ is the Ricci curvature 2-form of $g$
- $\mathfrak{D}_t$ is the measured dissipation 2-form
- $\lambda > 0$ is the coupling strength

**Continuum Limit (Ricci Flow):**

Taking $\tau \to 0$ and computing the Euler-Lagrange equation yields:

$$
\frac{\partial g}{\partial t} = -2 \,\text{Ric}(g) - \lambda \mathfrak{D}
$$

This is the **Ricci Flow equation** with a **dissipation-driven forcing term**:
- The first term $-2 \text{Ric}(g)$ is Hamilton's Ricci Flow, which smooths the metric toward constant curvature
- The second term $-\lambda \mathfrak{D}$ couples geometry to thermodynamics: high dissipation regions contract (reducing metric size), forcing the system to "learn" more efficient paths

**Physical Interpretation:**
- **Geometry → Thermodynamics:** The curvature $\text{Ric}(g)$ determines dissipation rate via the Jacobi metric $g_{\mathfrak{D}} = \mathfrak{D} \cdot g$
- **Thermodynamics → Geometry:** The measured dissipation $\mathfrak{D}_t$ feeds back to **deform the metric** $g_t$, creating a self-consistent dynamical system

**This is NOT a tautology:** The metric and dissipation are coupled through a **derived differential equation**, not through definition. The geometry evolves to minimize entropy production, which is a variational principle (like Einstein's field equations coupling spacetime geometry to matter stress-energy).

**For Discrete Systems (Simplicial Complex):**

On a simplicial complex $G = (V, E, F)$, the metric evolution becomes a **graph rewiring / edge weight update**:

$$
W_{ij}^{t+1} = W_{ij}^t - \tau \left( \frac{\partial \mathfrak{D}}{\partial W_{ij}} + \lambda \sum_{f \ni (i,j)} \kappa_f \right)
$$

where:
- $W_{ij}$ are edge weights (discrete metric)
- $\mathfrak{D}$ is the discrete dissipation (1-cochain on edges)
- $\kappa_f$ is the discrete curvature of face $f$ (from simplicial cohomology)

**This grounds the Ricci flow in concrete linear algebra**, avoiding infinite-dimensional PDE machinery.

**Literature:** Hamilton's Ricci Flow {cite}`Hamilton82`; Perelman's entropy functionals {cite}`Perelman02`; Discrete Ricci Flow on graphs {cite}`Chow03`; Bakry-Émery Ricci curvature {cite}`BakryEmery85`.
:::

:::{prf:remark} Universal Sieve Applicability
:label: rem-universal-sieve

The RCD formalism works for **non-smooth spaces** (graphs, discrete logic, singular geometries). The Cheeger Energy definition ({prf:ref}`thm-cheeger-dissipation`) applies to:
- **Continuous Physics:** Manifolds with Riemannian metrics
- **Discrete Logic:** Weighted graphs with discrete Laplacian
- **Hybrid Systems:** Stratified spaces with singularities

**Implication:** The **same Sieve** (with Metric-Measure upgrade) can verify:
- **Neural AI:** VAE/LLM latent spaces as Wasserstein manifolds
- **Symbolic AI:** Proof graphs as discrete metric-measure spaces
- **Robotics:** Configuration spaces with obstacles (Alexandrov spaces)

No separate framework is needed—RCD theory **unifies** geometry and thermodynamics across all modalities.

**Practical Verification via LSI Thin Permit ({prf:ref}`permit-lsi-thin`):** For discrete systems (Markov chains, graph neural networks, discretized trajectories), the Log-Sobolev Inequality and exponential convergence (No-Melt Theorem) can be verified **without hard analysis** by:
1. Extracting the weighted graph $G = (V, E, W)$ from the Thin State Object
2. Computing the spectral gap $\lambda_2(L) > 0$ of the graph Laplacian (finite linear algebra)
3. Invoking RCD stability theory to lift the discrete LSI to the continuum limit via the Expansion Adjunction $\mathcal{F} \dashv U$

This **discrete-to-continuum lifting** bypasses infinite-dimensional PDE analysis entirely, making LSI verification tractable for real ML systems. See {prf:ref}`def-node-stiffness` for the full protocol.
:::

### Imported Learning Metatheorems

The following metatheorems are imported from the core hypostructure framework and provide the foundational identifiability and reconstruction results required for trainable hypostructures.

:::{prf:metatheorem} SV-09: Meta-Identifiability
:label: mt-sv-09-meta-identifiability

**[Sieve Signature]**

- **Weakest Precondition**: $K_5^+$ (Parameters stable) AND $K_7^+$ (Log-Sobolev)
- **Produces**: $K_{\text{SV09}}$ (Local Injectivity)
- **Invalidated By**: $K_5^-$ (degenerate parametrization)


Permits: $\mathcal{P}_{\text{full}}$ (default; specialize if fewer permits are needed).

**Statement**: Parameters are learnable under persistent excitation and nondegenerate parametrization.

*Algorithmic Class:* Parameter Estimation. *Convergence:* Local Injectivity.
:::

:::{prf:metatheorem} Functional Reconstruction
:label: mt-functional-reconstruction

**[Sieve Signature]**
Permits: $\mathcal{P}_{\text{full}}$ (default; specialize if fewer permits are needed).


- **Weakest Precondition**: $K_{12}^+$ (gradient consistency) AND $\{K_{11}^+ \lor K_{\text{Epi}}^{\text{blk}}\}$ (finite dictionary)
- **Consumes**: Context $\Gamma$ with GradientCheck and ComplexCheck certificates
- **Produces**: $K_{\text{Reconstruct}}$ (explicit Lyapunov functional)
- **Invalidated By**: $K_{12}^-$ (gradient inconsistency) or $K_{\text{Epi}}^{\text{br}}$ (semantic horizon)


**Statement**: If the local Context $\Gamma$ contains gradient consistency and finite dictionary certificates, the Lyapunov functional is explicitly recoverable as the geodesic distance in a Jacobi metric, or as the solution to a Hamilton–Jacobi equation. No prior knowledge of an energy functional is required.
:::

:::{prf:metatheorem} Algorithmic Thermodynamics of the Sieve
:label: mt-algorithmic-thermodynamics

**[Sieve Signature]**
Permits: $K_{\text{Geom}}$, $K_{\text{Spec}}$, $K_{\text{Horizon}}$ (Geometric Structure, Spectral Resonance, Thermodynamic Limit)

- **Weakest Precondition**: Thin Kernel defined with finite computational budget $\mathcal{S}_{\max}$ (Bekenstein bound)
- **Consumes**: Verification trace $\tau$ with Levin Complexity $Kt(\tau) = |\tau| + \log(\text{steps}(\tau))$
- **Produces**: Phase classification $\{\text{Solid}, \text{Liquid}, \text{Gas}\}$ with certificate $K_{\text{Phase}}^+$
- **Invalidated By**: $Kt(\tau) > \mathcal{S}_{\max}$ → **HORIZON** verdict

**Statement**: The Structural Sieve $\mathcal{S}$ induces a **Renormalization Group (RG) Flow** on the space of input systems. The limit points of this flow classify the computational complexity of the input into thermodynamic phases.

**Classification (Phase Diagram)**:

1. **Solid Phase (Decidable/Crystal)**:
   - **RG Behavior**: Flow converges to low-entropy fixed point ($Kt \ll |x|$)
   - **Certificates**: $K_{\text{Geom}}^{+}(\text{Poly})$ (Polynomial growth) OR $K_{\text{Geom}}^{+}(\text{CAT0})$ (Structured)
   - **Physical Analog**: Crystal / Integrable System
   - **Verdict**: **REGULAR**
   - **Examples**: Euclidean lattices $\mathbb{Z}^d$, Nilpotent groups, Higher-rank lattices $SL(n,\mathbb{Z})$ ($n \geq 3$)

2. **Liquid Phase (Critical/Compressible)**:
   - **RG Behavior**: Flow remains scale-invariant but structured ($Kt \sim \log |x|$)
   - **Certificates**: $K_{\text{Geom}}^{+}(\text{Hyp})$ (Hyperbolic) OR $K_{\text{Spec}}^{+}$ (Spectral Resonance)
   - **Physical Analog**: Self-Organized Criticality / Quantum Chaos
   - **Verdict**: **PARTIAL**
   - **Examples**: Free groups, Logic trees, Riemann zeros, Quantum graphs, Arithmetic chaos

3. **Gas Phase (Undecidable/Random)**:
   - **RG Behavior**: Flow diverges to maximum entropy ($Kt \sim |x|$)
   - **Certificates**: $K_{\text{Horizon}}^{\text{blk}}$ (Levin Limit exceeded) OR $K_{\text{Geom}}^{-} \land K_{\text{Spec}}^{-}$ (Expander without resonance)
   - **Physical Analog**: Thermal Equilibrium / Randomness
   - **Verdict**: **HORIZON**
   - **Examples**: Halting Problem, Random matrices, Generic expanders, Chaitin's $\Omega$

**Proof Strategy**:

*Step 1 (Levin-Schnorr Foundation):* By the **Levin-Schnorr Theorem** ({cite}`Levin73`, {cite}`Schnorr71`), algorithmic incompressibility (Kolmogorov complexity $K(x) \approx |x|$) implies unpredictability (Martin-Löf randomness). Inputs in the Gas Phase have $Kt(\tau) \approx |\tau|$ — no effective theory shorter than themselves.

*Step 2 (RG Flow Dynamics):* Define the renormalization operator $\mathcal{R}_\ell$ as coarse-graining by scale $\ell$:

$$
\mathcal{R}_\ell(\mathcal{I}) := \{\text{structural features visible at scale } \ell\}
$$

- **Solid**: $\mathcal{R}_\ell(\mathcal{I}) \to \mathcal{I}_{\text{simple}}$ (converges to finite representation)
- **Liquid**: $\mathcal{R}_\ell(\mathcal{I})$ remains self-similar across scales (power-law decay, no characteristic scale)
- **Gas**: $\mathcal{R}_\ell(\mathcal{I}) \to$ maximum entropy (no structure at any scale)

*Step 3 (Phase Transition Detection):* The Sieve correctly identifies phases via:
- **Geometric Tests** ({prf:ref}`ax-geom-tits`): Polynomial/Hyperbolic/CAT(0) vs. Expander
- **Spectral Tests** ({prf:ref}`ax-spectral-resonance`): Arithmetic correlations vs. Random matrix statistics
- **Resource Bounds** ({prf:ref}`def-thermodynamic-horizon`): $Kt(\tau) > \mathcal{S}_{\max}$ → Gas Phase

*Step 4 (Thermodynamic Budget):* The Bekenstein Bound $\mathcal{S}_{\max} = \frac{2\pi k_B ER}{\hbar c}$ for a finite computational system imposes fundamental limits. When $Kt(\tau) > \mathcal{S}_{\max}$, the verification trace exceeds physical capacity → honest **HORIZON** verdict.

*Step 5 (Correctness):* The Sieve does not claim to "solve undecidable problems" — it **classifies** them as thermodynamically inaccessible (Gas Phase), maintaining soundness.

**Universal Coverage Table**:

| Input Class | Geometric Test | Spectral Test | Phase | Verdict | Certificate |
|------------|----------------|---------------|-------|---------|-------------|
| Polynomial Growth | $K_{\text{Geom}}^{+}(\text{Poly})$ | N/A | Solid | **REGULAR** | Finite group/manifold |
| Hyperbolic/Logic | $K_{\text{Geom}}^{+}(\text{Hyp})$ | N/A | Liquid | **PARTIAL** | Tree/Free group encoding |
| CAT(0)/Lattices | $K_{\text{Geom}}^{+}(\text{CAT0})$ | N/A | Solid | **REGULAR** | Building/symmetric space |
| Arithmetic Chaos | $K_{\text{Geom}}^{-}$ | $K_{\text{Spec}}^{+}$ | Liquid | **PARTIAL** | Trace formula/L-function |
| Random/Thermal | $K_{\text{Geom}}^{-}$ | $K_{\text{Spec}}^{-}$ | Gas | **HORIZON** | $Kt > \mathcal{S}_{\max}$ |

**Significance**: This metatheorem elevates the Sieve from a heuristic diagnostic to a **rigorous phase transition detector** grounded in:
- **Algorithmic Information Theory** (Kolmogorov complexity, Levin complexity)
- **Geometric Group Theory** (Tits Alternative, CAT(0) spaces)
- **Random Matrix Theory** (Spectral statistics, Trace formulas)
- **Physical Thermodynamics** (Bekenstein bound, resource-bounded computation)

**Literature:** {cite}`Levin73`; {cite}`Schnorr71`; {cite}`Chaitin75`; {cite}`Tits72`; {cite}`Gromov87`; {cite}`Selberg56`; {cite}`Bekenstein81`; {cite}`Lloyd00`
:::

---

### Trainable Hypostructure Consistency

The preceding sections established that axiom defects can be minimized via gradient descent. This section proves the central metatheorem: under identifiability conditions, defect minimization provably recovers the true hypostructure and its structural predictions.

**Setting.** Fix a dynamical system $S$ with state space $X$, semiflow $S_t$, and trajectory class $\mathcal{U}$. Suppose there exists a "true" hypostructure

$$
\mathcal{H}_{\Theta^*} = (X, S_t, \Phi_{\Theta^*}, \mathfrak{D}_{\Theta^*}, G_{\Theta^*}, \mathcal{B}_{\Theta^*}, \mathrm{Tr}_{\Theta^*}, \mathcal{J}_{\Theta^*}, \mathcal{R}_{\Theta^*})
$$

satisfying the axioms. Consider a parametric family $\{\mathcal{H}_\theta\}_{\theta \in \Theta_{\mathrm{adm}}}$ containing $\mathcal{H}_{\Theta^*}$, with joint defect risk:

$$
\mathcal{R}(\theta) := \sum_{A \in \mathcal{A}} w_A \, \mathcal{R}_A(\theta), \quad \mathcal{R}_A(\theta) := \int_{\mathcal{U}} K_A^{(\theta)}(u) \, d\mu(u).
$$

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom Validity at $\Theta^*$:** The target hypostructure $\mathcal{H}_{\Theta^*}$ satisfies axioms (C, D, SC, Cap, LS, TB, Bound, Reg, GC)
>     *   [ ] **Well-Behaved Defect Functionals:** Compact $\Theta$, continuous $\theta \mapsto K_A^{(\theta)}(u)$, integrable majorants ({prf:ref}`lem-leibniz-rule-for-defect-risk`)
>     *   [ ] **Structural Identifiability:** Persistent excitation (C1), nondegenerate parametrization (C2), regular parameter space (C3) ({prf:ref}`mt-sv-09-meta-identifiability`)
>     *   [ ] **Defect Reconstruction:** Reconstruction of $(\Phi_\theta, \mathfrak{D}_\theta, S_t, \mathcal{B}_\theta, \mathrm{Tr}_\theta, \mathcal{J}_\theta, \mathcal{R}_\theta, \text{barriers}, M)$ from defects up to Hypo-isomorphism ({prf:ref}`mt-defect-reconstruction-2`)
> *   **Output (Structural Guarantee):**
>     *   Global minimizer $\Theta^*$ satisfies $\mathcal{R}(\Theta^*) = 0$; any global minimizer $\hat{\theta}$ with $\mathcal{R}(\hat{\theta}) = 0$ yields $\mathcal{H}_{\hat{\theta}} \cong \mathcal{H}_{\Theta^*}$
>     *   Local quadratic identifiability: $c|\theta - \tilde{\Theta}|^2 \leq \mathcal{R}(\theta) \leq C|\theta - \tilde{\Theta}|^2$
>     *   Gradient descent converges to true hypostructure with Robbins-Monro step sizes
>     *   Barrier constants and failure-mode classifications converge
> *   **Failure Condition (Debug):**
>     *   If **Axiom Validity** fails → **Mode misspecification** (wrong axiom target)
>     *   If **Identifiability** fails → **Mode parameter degeneracy** (multiple equivalent minima)
>     *   If **Defect Reconstruction** fails → **Mode reconstruction ambiguity** (structural non-uniqueness)

:::{prf:metatheorem} Trainable Hypostructure Consistency
:label: mt-trainable-hypostructure-consistency

Let $S$ be a dynamical system with a hypostructure representation $\mathcal{H}_{\Theta^*}$ inside a parametric family $\{\mathcal{H}_\theta\}_{\theta \in \Theta_{\mathrm{adm}}}$. Assume:

1. **(Axiom validity at $\Theta^*$.)** The hypostructure $\mathcal{H}_{\Theta^*}$ satisfies axioms (C, D, SC, Cap, LS, TB, Bound, Reg, GC). Consequently, $K_A^{(\Theta^*)}(u) = 0$ for $\mu$-a.e. trajectory $u \in \mathcal{U}$ and all $A \in \mathcal{A}$.

2. **(Well-behaved defect functionals.)** The assumptions of {prf:ref}`lem-leibniz-rule-for-defect-risk` hold: $\Theta$ compact and metrizable, $\theta \mapsto K_A^{(\theta)}(u)$ continuous and differentiable with integrable majorants.

3. **(Structural identifiability.)** The family satisfies the conditions of {prf:ref}`mt-sv-09-meta-identifiability`: persistent excitation (C1), nondegenerate parametrization (C2), and regular parameter space (C3).

4. **(Defect reconstruction.)** The Defect Reconstruction Theorem ({prf:ref}`mt-defect-reconstruction-2`) holds: from $\{K_A^{(\theta)}\}_{A \in \mathcal{A}}$ on $\mathcal{U}$, one reconstructs $(\Phi_\theta, \mathfrak{D}_\theta, S_t, \mathcal{B}_\theta, \mathrm{Tr}_\theta, \mathcal{J}_\theta, \mathcal{R}_\theta, \text{barriers}, M)$ up to Hypo-isomorphism.

Consider gradient descent with step sizes $\eta_k > 0$ satisfying $\sum_k \eta_k = \infty$, $\sum_k \eta_k^2 < \infty$:

$$
\theta_{k+1} = \theta_k - \eta_k \nabla_\theta \mathcal{R}(\theta_k).
$$

Then:

1. **(Correctness of global minimizer.)** $\Theta^*$ is a global minimizer of $\mathcal{R}$ with $\mathcal{R}(\Theta^*) = 0$. Conversely, any global minimizer $\hat{\theta}$ with $\mathcal{R}(\hat{\theta}) = 0$ satisfies $\mathcal{H}_{\hat{\theta}} \cong \mathcal{H}_{\Theta^*}$ (Hypo-isomorphic).

2. **(Local quantitative identifiability.)** There exist $c, C, \varepsilon_0 > 0$ such that for $|\theta - \Theta^*| < \varepsilon_0$:

   $$
   c \, |\theta - \tilde{\Theta}|^2 \leq \mathcal{R}(\theta) \leq C \, |\theta - \tilde{\Theta}|^2
   $$

   where $\tilde{\Theta}$ is a representative of $[\Theta^*]$. In particular: $\mathcal{R}(\theta) \leq \varepsilon \Rightarrow |\theta - \tilde{\Theta}| \leq \sqrt{\varepsilon/c}$.

3. **(Convergence to true hypostructure.)** Every accumulation point of $(\theta_k)$ is stationary. Under the local strong convexity of (2), any sequence initialized sufficiently close to $[\Theta^*]$ converges to some $\tilde{\Theta} \in [\Theta^*]$.

4. **(Barrier and failure-mode convergence.)** As $\theta_k \to \tilde{\Theta}$, barrier constants converge to those of $\mathcal{H}_{\Theta^*}$, and for all large $k$, $\mathcal{H}_{\theta_k}$ forbids exactly the same failure modes as $\mathcal{H}_{\Theta^*}$.
:::

:::{prf:proof}
**Step 1 ($\Theta^*$ is correct global minimizer).** By assumption (1), $K_A^{(\Theta^*)}(u) = 0$ for $\mu$-a.e. $u$ and all $A$. Thus $\mathcal{R}_A(\Theta^*) = 0$ for all $A$, hence $\mathcal{R}(\Theta^*) = 0$. Since $K_A^{(\theta)} \geq 0$, we have $\mathcal{R}(\theta) \geq 0$ for all $\theta$, so $\Theta^*$ achieves the global minimum.

Conversely, if $\mathcal{R}(\hat{\theta}) = 0$, then $\mathcal{R}_A(\hat{\theta}) = 0$ for all $A$, so $K_A^{(\hat{\theta})}(u) = 0$ for $\mu$-a.e. $u$. By the Defect Reconstruction Theorem, both $\mathcal{H}_{\hat{\theta}}$ and $\mathcal{H}_{\Theta^*}$ reconstruct to the same structural data on the support of $\mu$. By structural identifiability ({prf:ref}`mt-sv-09-meta-identifiability`), $\mathcal{H}_{\hat{\theta}} \cong \mathcal{H}_{\Theta^*}$.

**Step 2 (Local quadratic bounds).** By Defect Reconstruction and structural identifiability, the map $\theta \mapsto \mathsf{Sig}(\theta)$ is locally injective around $[\Theta^*]$ up to gauge. Since $\mathcal{R}(\Theta^*) = 0$ and $\nabla \mathcal{R}(\Theta^*) = 0$ (all defects vanish), Taylor expansion gives:

$$
\mathcal{R}(\theta) = \frac{1}{2}(\theta - \tilde{\Theta})^\top H (\theta - \tilde{\Theta}) + o(|\theta - \tilde{\Theta}|^2)
$$

where $H = \sum_A w_A H_A$ is the Hessian. Identifiability implies $H$ is positive definite on $\Theta_{\mathrm{adm}}/{\sim}$ (directions that leave all defects unchanged correspond to pure gauge). Thus for small $|\theta - \tilde{\Theta}|$:

$$
c \, |\theta - \tilde{\Theta}|^2 \leq \mathcal{R}(\theta) \leq C \, |\theta - \tilde{\Theta}|^2.
$$

**Step 3 (Gradient descent convergence).** By {prf:ref}`cor-gradient-descent-convergence`, accumulation points are stationary. The local strong convexity from Step 2 implies: on $B(\tilde{\Theta}, \varepsilon_0)$, $\mathcal{R}$ is strongly convex (modulo gauge) with unique stationary point $\tilde{\Theta}$. Standard optimization theory for strongly convex functions with Robbins-Monro step sizes yields convergence of $(\theta_k)$ to $\tilde{\Theta}$ when initialized in this basin.

**Step 4 (Barrier convergence).** Barrier constants and failure-mode classifications are continuous in the structural data $(\Phi, \mathfrak{D}, \alpha, \beta, \ldots)$ by {prf:ref}`mt-sv-09-meta-identifiability`. Since $\theta_k \to \tilde{\Theta}$, structural data converges, hence barriers converge and failure-mode predictions stabilize.
:::

**Key Insight (Structural parameter estimation).** This theorem elevates the meta-learning framework from "we can optimize a loss" to a metatheorem: under identifiability, **structural parameters are estimable**. The parameter manifold $\Theta$ is equipped with the Fisher-Rao metric, following Amari's Information Geometry {cite}`Amari00`, treating learning as a projection onto a statistical manifold. The minimization of defect risk $\mathcal{R}(\theta)$ converges to the unique hypostructure compatible with the trajectory distribution $\mu$, and all high-level structural predictions (barrier constants, forbidden failure modes) converge with it.

---

:::{prf:remark} What the metatheorem says

In plain language:

1. If a system admits a hypostructure satisfying the axioms for some $\Theta^*$,
2. and the parametric family + data is rich enough to make that hypostructure identifiable,
3. then defect minimization is a **consistent learning principle**:

   - The global minimum corresponds exactly to $\Theta^*$ (mod gauge)
   - Small risk means ``almost recovered the true axioms''
   - Gradient descent converges to the correct hypostructure
   - All structural predictions (barriers, forbidden modes) converge


:::

:::{prf:corollary} Verification via training
:label: cor-verification-via-training

A trained hypostructure with $\mathcal{R}(\theta_k) < \varepsilon$ provides:

1. **Approximate axiom satisfaction:** Each axiom holds with defect at most $\varepsilon/w_A$
2. **Approximate structural recovery:** Parameters within $\sqrt{\varepsilon/c}$ of truth
3. **Correct qualitative predictions:** For $\varepsilon$ small enough, barrier signs and failure-mode classifications match the true system

This connects the trainable framework to the diagnostic and verification goals of the hypostructure program.
:::

### Meta-Error Localization

The previous section established that defect minimization recovers the true hypostructure. This section addresses a finer question: when training yields nonzero residual risk, **which axiom block is misspecified?** We prove that the pattern of residual risks under blockwise retraining uniquely identifies the error location.

#### Parameter block structure

:::{prf:definition} Block decomposition
:label: def-block-decomposition

Decompose the parameter space into axiom-aligned blocks:
$$\theta = (\theta^{\mathrm{dyn}}, \theta^{\mathrm{cap}}, \theta^{\mathrm{sc}}, \theta^{\mathrm{top}}, \theta^{\mathrm{ls}}) \in \Theta_{\mathrm{adm}}$$
where:

- $\theta^{\mathrm{dyn}}$: parallel transport/dynamics parameters (C, D axioms)
- $\theta^{\mathrm{cap}}$: capacity and barrier constants (Cap, TB axioms)
- $\theta^{\mathrm{sc}}$: scaling exponents and structure (SC axiom)
- $\theta^{\mathrm{top}}$: topological sector data (TB, topological aspects of Cap)
- $\theta^{\mathrm{ls}}$: Łojasiewicz exponents and symmetry-breaking data (LS axiom)

:::

Let $\mathcal{B} := \{\mathrm{dyn}, \mathrm{cap}, \mathrm{sc}, \mathrm{top}, \mathrm{ls}\}$ denote the set of block labels.

:::{prf:definition} Block-restricted reoptimization
:label: def-block-restricted-reoptimization

For block $b \in \mathcal{B}$ and current parameter $\theta$, define:

1. **Feasible set:** $\Theta^b(\theta) := \{\tilde{\theta} \in \Theta_{\mathrm{adm}} : \tilde{\theta}^c = \theta^c \text{ for all } c \neq b\}$
2. **Block-restricted minimal risk:** $\mathcal{R}_b^*(\theta) := \inf_{\tilde{\theta} \in \Theta^b(\theta)} \mathcal{R}(\tilde{\theta})$

This represents "retrain only block $b$" while freezing all other blocks.
:::

:::{prf:definition} Response signature
:label: def-response-signature

The **response signature** at $\theta$ is:
$$\rho(\theta) := \big(\mathcal{R}_b^*(\theta)\big)_{b \in \mathcal{B}} \in \mathbb{R}_{\geq 0}^{|\mathcal{B}|}$$
:::

:::{prf:definition} Error support
:label: def-error-support

Given true parameter $\Theta^* = (\Theta^{*,b})_{b \in \mathcal{B}}$ and current parameter $\theta$, the **error support** is:
$$E(\theta) := \{b \in \mathcal{B} : \theta^b \not\sim \Theta^{*,b}\}$$
where $\sim$ denotes gauge equivalence within Hypo-isomorphism classes.
:::

#### Localization assumptions

:::{prf:definition} Block-orthogonality conditions
:label: def-block-orthogonality-conditions

The parametric family satisfies **block-orthogonality** if in a neighborhood $\mathcal{N}$ of $[\Theta^*]$:

1. **(Smooth risk.)** $\mathcal{R}$ is $C^2$ on $\mathcal{N}$ with Hessian $H := \nabla^2 \mathcal{R}(\Theta^*)$ positive definite modulo gauge.

2. **(Block-diagonal Hessian.)** $H$ decomposes as:
$$H = \bigoplus_{b \in \mathcal{B}} H_b$$
where each $H_b$ is positive definite on its block. Cross-Hessian blocks $H_{bc} = 0$ for $b \neq c$ (modulo gauge).

3. **(Quadratic approximation.)** There exists $\delta > 0$ such that for $|\theta - \Theta^*| < \delta$:
$$\mathcal{R}(\theta) = \frac{1}{2}(\theta - \Theta^*)^\top H (\theta - \Theta^*) + O(|\theta - \Theta^*|^3)$$
:::

:::{prf:remark} Interpretation of block-orthogonality

Condition (2) means: perturbations in different axiom blocks contribute additively and independently to the risk at second order. No combination of ``wrong capacity'' and ``wrong scaling'' can cancel in the expected defect. This holds when the parametrization is factorized by axiom family without hidden re-encodings.
:::

#### The localization theorem

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Block-Orthogonality Conditions:** Smooth risk $\mathcal{R}$ with positive-definite Hessian $H = \bigoplus_b H_b$, block-diagonal modulo gauge
>     *   [ ] **Quadratic Approximation:** $\mathcal{R}(\theta) = \frac{1}{2}(\theta - \Theta^*)^\top H (\theta - \Theta^*) + O(|\theta - \Theta^*|^3)$
>     *   [ ] **Parameter Block Decomposition:** $\theta = (\theta^{\mathrm{dyn}}, \theta^{\mathrm{cap}}, \theta^{\mathrm{sc}}, \theta^{\mathrm{top}}, \theta^{\mathrm{ls}})$
> *   **Output (Structural Guarantee):**
>     *   Single-block error: uniquely smallest $\mathcal{R}_b^*(\theta)$ identifies misspecified block
>     *   Multiple-block error: response signature discriminates error support
>     *   Signature injectivity: $b \in E(\theta) \iff \mathcal{R}_b^*(\theta) \leq \gamma \cdot \min_{c \notin E(\theta)} \mathcal{R}_c^*(\theta)$
> *   **Failure Condition (Debug):**
>     *   If **Block-Orthogonality** fails → **Mode cross-coupling** (blocks interfere, false positives)
>     *   If **Quadratic Approximation** fails → **Mode higher-order dominance** (cubic terms mask signal)

:::{prf:metatheorem} Meta-Error Localization
:label: mt-meta-error-localization

Assume the block-orthogonality conditions ({prf:ref}`def-block-orthogonality-conditions`). There exist $\mathcal{N}$, $c$, $C$, $\varepsilon_0 > 0$ such that for $\theta \in \mathcal{N}$ with $|\theta - \Theta^*| < \varepsilon_0$:

1. **(Single-block error.)** If $E(\theta) = \{b^*\}$ (exactly one misspecified block), then:
   - For block $b^*$: $\mathcal{R}_{b^*}^*(\theta) \leq C |\theta - \Theta^*|^3$
   - For $b \neq b^*$: $\mathcal{R}_b^*(\theta) \geq c |\theta - \Theta^*|^2$

   The uniquely smallest $\mathcal{R}_b^*(\theta)$ identifies the misspecified block.

2. **(Multiple-block error.)** For arbitrary nonempty $E(\theta) \subseteq \mathcal{B}$:
   - If $b \notin E(\theta)$: $\mathcal{R}_b^*(\theta) \geq c \sum_{c \in E(\theta)} |\theta^c - \Theta^{*,c}|^2$
   - If $b \in E(\theta)$: $\mathcal{R}_b^*(\theta) \approx \frac{1}{2} \sum_{c \in E(\theta) \setminus \{b\}} (\theta^c - \Theta^{*,c})^\top H_c (\theta^c - \Theta^{*,c})$

3. **(Signature injectivity.)** There exists $\gamma > 0$ such that:
$$b \in E(\theta) \iff \mathcal{R}_b^*(\theta) \leq \gamma \cdot \min_{c \notin E(\theta)} \mathcal{R}_c^*(\theta)$$

The map $E \mapsto \rho(\theta)$ is injective and stable: the response signature uniquely encodes the error support.
:::

:::{prf:proof}
Let $\delta\theta := \theta - \Theta^*$ with block decomposition $\delta\theta = (\delta\theta^b)_{b \in \mathcal{B}}$.

**Step 1 (Quadratic structure).** By assumption, $\mathcal{R}(\theta) = \frac{1}{2}\delta\theta^\top H \delta\theta + O(|\delta\theta|^3)$. Block-diagonality gives:
$$\delta\theta^\top H \delta\theta = \sum_{b \in \mathcal{B}} (\delta\theta^b)^\top H_b \delta\theta^b.$$
Since each $H_b$ is positive definite, there exist $m_b, M_b > 0$ with:
$$m_b |\delta\theta^b|^2 \leq (\delta\theta^b)^\top H_b \delta\theta^b \leq M_b |\delta\theta^b|^2.$$

**Step 2 (Block-restricted optimization).** For block $b$, the restricted optimization varies only $\delta\theta^b$ while fixing $\delta\theta^c$ for $c \neq b$. The quadratic approximation:
$$Q(\delta\theta) = \frac{1}{2} \sum_{c \in \mathcal{B}} (\delta\theta^c)^\top H_c \delta\theta^c$$
splits by block. The minimum over $\delta\theta^b$ is achieved at $\delta\theta^b = 0$, giving:
$$Q_b^*(\delta\theta) := \inf_{\tilde{\delta\theta}^b} Q = \frac{1}{2} \sum_{c \neq b} (\delta\theta^c)^\top H_c \delta\theta^c.$$
The true minimal risk satisfies $|\mathcal{R}_b^*(\theta) - Q_b^*(\delta\theta)| \leq C_1 |\delta\theta|^3$.

**Step 3 (Single-block case).** If $E(\theta) = \{b^*\}$, then $\delta\theta^c = 0$ for $c \neq b^*$.

For $b = b^*$: $Q_{b^*}^* = \frac{1}{2}\sum_{c \neq b^*} (\delta\theta^c)^\top H_c \delta\theta^c = 0$, so $\mathcal{R}_{b^*}^* \leq C|\delta\theta|^3$.

For $b \neq b^*$: $Q_b^* \geq \frac{1}{2} m_{b^*} |\delta\theta^{b^*}|^2 \geq c|\delta\theta|^2$, so $\mathcal{R}_b^* \geq c|\delta\theta|^2 - C_1|\delta\theta|^3 \geq \frac{c}{2}|\delta\theta|^2$ for small $|\delta\theta|$.

**Step 4 (Multiple-block case).** For general $E(\theta)$:

If $b \notin E(\theta)$: The sum $Q_b^* = \frac{1}{2}\sum_{c \neq b} (\delta\theta^c)^\top H_c \delta\theta^c$ includes all error blocks $c \in E(\theta)$, giving the lower bound.

If $b \in E(\theta)$: The sum excludes block $b$, so $Q_b^* = \frac{1}{2}\sum_{c \in E(\theta) \setminus \{b\}} (\delta\theta^c)^\top H_c \delta\theta^c$.

**Step 5 (Signature discrimination).** Blocks in $E(\theta)$ have systematically smaller $\mathcal{R}_b^*$ than blocks not in $E(\theta)$, by a multiplicative margin depending on the spectra of $H_c$. Taking $\gamma$ as the ratio of spectral bounds yields the equivalence.
:::

---

**Key Insight (Built-in debugger).** A trainable hypostructure comes with principled error diagnosis:

1. Train the full model to reduce $\mathcal{R}(\theta)$
2. If residual risk remains, compute $\mathcal{R}_b^*$ for each block by retraining only that block
3. The pattern $\rho(\theta) = (\mathcal{R}_b^*)_b$ provably identifies which axiom blocks are wrong

:::{prf:corollary} Diagnostic protocol
:label: cor-diagnostic-protocol

Given trained parameters $\theta$ with $\mathcal{R}(\theta) > 0$:

1. **Compute response signature:** For each $b \in \mathcal{B}$, solve $\mathcal{R}_b^*(\theta) = \min_{\tilde{\theta}^b} \mathcal{R}(\theta^{-b}, \tilde{\theta}^b)$
2. **Identify error support:** $\hat{E} = \{b : \mathcal{R}_b^*(\theta) \text{ is anomalously small}\}$
3. **Interpret:** The blocks in $\hat{E}$ are misspecified; blocks not in $\hat{E}$ are correct
:::

:::{prf:remark} Error types and remediation

The error support $E(\theta)$ indicates:

| Error Support | Interpretation | Remediation |
|--------------|----------------|-------------|
| $\{\mathrm{dyn}\}$ | Dynamics model wrong | Revise connection/transport ansatz |
| $\{\mathrm{cap}\}$ | Capacity/barriers wrong | Adjust geometric estimates |
| $\{\mathrm{sc}\}$ | Scaling exponents wrong | Recompute dimensional analysis |
| $\{\mathrm{top}\}$ | Topological sectors wrong | Check sector decomposition |
| $\{\mathrm{ls}\}$ | Łojasiewicz data wrong | Verify equilibrium structure |
| Multiple | Combined misspecification | Address each block |

This connects the trainable framework to systematic model debugging and refinement.
:::

### Block Factorization Axiom

The Meta-Error Localization Theorem ({prf:ref}`mt-meta-error-localization`) assumes that when we restrict reoptimization to a single parameter block $\theta^b$, the result meaningfully tests whether that block is correct. This requires that the axiom defects factorize cleanly across parameter blocks—a structural condition we now formalize.

:::{prf:definition} Axiom-Support Set
:label: def-axiom-support-set

For each axiom $A \in \mathcal{A}$, define its **axiom-support set** $\mathrm{Supp}(A) \subseteq \mathcal{B}$ as the minimal collection of blocks such that:
$$K_A^{(\theta)}(u) = K_A^{(\theta|_{\mathrm{Supp}(A)})}(u)$$
for all trajectories $u$ and all parameters $\theta$. That is, $\mathrm{Supp}(A)$ contains exactly the blocks that the defect functional $K_A$ actually depends on.
:::

:::{prf:definition} Semantic Block via Axiom Support
:label: def-semantic-block-via-axiom-support

A partition $\mathcal{B}$ of the parameter space $\theta = (\theta^b)_{b \in \mathcal{B}}$ is **semantically aligned** if each block $b$ corresponds to a coherent set of axiom dependencies:
$$b \in \mathrm{Supp}(A) \implies \text{all parameters in } \theta^b \text{ influence } K_A$$
:::

**Block Factorization Axiom (BFA).** We say the hypostructure training problem satisfies the **Block Factorization Axiom** if:

**(BFA-1) Sparse support:** Each axiom depends on few blocks:
$$|\mathrm{Supp}(A)| \leq k \quad \text{for all } A \in \mathcal{A}$$
for some constant $k \ll |\mathcal{B}|$.

**(BFA-2) Block coverage:** Each block is responsible for at least one axiom:
$$\forall b \in \mathcal{B}, \exists A \in \mathcal{A}: b \in \mathrm{Supp}(A)$$

**(BFA-3) Separability:** The joint risk decomposes additively across axiom families:
$$\mathcal{R}(\theta) = \sum_{A \in \mathcal{A}} w_A \mathcal{R}_A(\theta)$$
where each $\mathcal{R}_A$ depends only on blocks in $\mathrm{Supp}(A)$.

**(BFA-4) Independence of irrelevant alternatives:** For blocks $b \notin \mathrm{Supp}(A)$:
$$\frac{\partial \mathcal{R}_A}{\partial \theta^b} = 0$$
That is, blocks outside an axiom's support have zero gradient contribution to that axiom's risk.

:::{prf:remark} Interpretation

BFA formalizes the intuition that:

- **Dynamics parameters** ($\theta^{\mathrm{dyn}}$) govern D, R, C---the core semiflow structure
- **Capacity parameters** ($\theta^{\mathrm{cap}}$) govern Cap, TB---geometric barriers
- **Scaling parameters** ($\theta^{\mathrm{sc}}$) govern SC---dimensional analysis
- **Topological parameters** ($\theta^{\mathrm{top}}$) govern GC---sector structure
- **Łojasiewicz parameters** ($\theta^{\mathrm{ls}}$) govern LS---equilibrium geometry

When BFA holds, testing whether $\theta^{\mathrm{cap}}$ is correct (by computing $\mathcal{R}_{\mathrm{cap}}^*$) cannot be confounded by errors in $\theta^{\mathrm{sc}}$, because capacity axioms do not depend on scaling parameters.
:::

:::{prf:lemma} Stability of Block Factorization under Composition
:label: lem-stability-of-block-factorization-under-composition

Let $(\mathcal{A}_1, \mathcal{B}_1)$ and $(\mathcal{A}_2, \mathcal{B}_2)$ be two axiom-block systems satisfying BFA with constants $k_1$ and $k_2$. If the systems have disjoint parameter spaces, then the combined system $(\mathcal{A}_1 \cup \mathcal{A}_2, \mathcal{B}_1 \cup \mathcal{B}_2)$ satisfies BFA with constant $\max(k_1, k_2)$.
:::

:::{prf:proof}
We verify each clause:

**Step 1 (BFA-1).** For $A \in \mathcal{A}_1$, $\mathrm{Supp}(A) \subseteq \mathcal{B}_1$ with $|\mathrm{Supp}(A)| \leq k_1$. Similarly for $\mathcal{A}_2$. Thus all axioms satisfy sparse support with constant $\max(k_1, k_2)$.

**Step 2 (BFA-2).** Each block in $\mathcal{B}_1$ is covered by some axiom in $\mathcal{A}_1$ (by BFA-2 for system 1). Similarly for $\mathcal{B}_2$. Union preserves coverage.

**Step 3 (BFA-3).** Since parameter spaces are disjoint, $\mathcal{R}_A(\theta_1, \theta_2) = \mathcal{R}_A(\theta_1)$ for $A \in \mathcal{A}_1$. Additive decomposition extends to the union.

**Step 4 (BFA-4).** For $A \in \mathcal{A}_1$ and $b \in \mathcal{B}_2$, the gradient $\partial \mathcal{R}_A / \partial \theta^b = 0$ because $\mathcal{R}_A$ does not depend on $\mathcal{B}_2$ parameters. Combined with original BFA-4 within each system, independence holds globally.
:::

:::{prf:remark} Role in Meta-Error Localization

The Meta-Error Localization Theorem ({prf:ref}`mt-meta-error-localization`) requires BFA implicitly:

- **Response signature well-defined:** $\mathcal{R}_b^*(\theta)$ tests block $b$ in isolation only if BFA-4 ensures other-block gradients do not interfere
- **Error support meaningful:** The set $E(\theta) = \{b : \mathcal{R}_b^*(\theta) < \mathcal{R}(\theta)\}$ identifies the *actual* error blocks only if BFA-1 ensures axiom-block correspondences are sparse
- **Diagnostic protocol valid:** {prf:ref}`cor-diagnostic-protocol`'s remediation table assumes the semantic alignment of {prf:ref}`def-semantic-block-via-axiom-support`

When BFA fails---for example, if capacity and scaling parameters are entangled---then $\mathcal{R}_{\mathrm{cap}}^*$ might decrease even when capacity is correct (because reoptimizing $\theta^{\mathrm{cap}}$ partially compensates for $\theta^{\mathrm{sc}}$ errors). This would produce false positives in error localization.
:::

> **Key Insight:** The Block Factorization Axiom is a *design constraint* on hypostructure parametrizations, not a theorem about dynamics. When constructing trainable hypostructures, one should choose parameter blocks that satisfy BFA—ensuring the Meta-Error Localization machinery works as intended.

### Meta-Generalization Across Systems

In the previous sections we considered a single system $S$ and a parametric family of hypostructures $\{\mathcal{H}_{\Theta,S}\}_{\Theta \in \Theta_{\mathrm{adm}}}$ with axiom-defect risk $\mathcal{R}_S(\Theta)$ (see {prf:ref}`def-joint-defect-risk`). We now move to a *distribution of systems* and show that defect-minimizing hypostructure parameters learned on a training distribution $\mathcal{S}_{\mathrm{train}}$ generalize to new systems drawn from the same structural class.

We write $\mathcal{S}$ for a probability measure on a class of systems, and for each $S$ in the support of $\mathcal{S}$, we assume a hypostructure family $\{\mathcal{H}_{\Theta,S}\}_{\Theta \in \Theta_{\mathrm{adm}}}$ and defect-risk functionals $\mathcal{R}_S(\Theta)$ as in the meta-learning axioms framework.

#### Setting

- Let $\mathcal{S}$ be a distribution over systems $S$ (e.g. PDEs, ODEs, control systems, RL environments) each admitting a hypostructure representation in the same parametric family $\{\mathcal{H}_{\Theta,S}\}_{\Theta \in \Theta_{\mathrm{adm}}}$.

- For each system $S$, the joint defect-risk $\mathcal{R}_S(\Theta)$ is defined via the defect functionals:
$$\mathcal{R}_S(\Theta) := \sum_{A \in \mathcal{A}} w_A \mathcal{R}_{A,S}(\Theta), \qquad \mathcal{R}_{A,S}(\Theta) := \int_{\mathcal{U}_S} K_{A,S}^{(\Theta)}(u) \, d\mu_S(u),$$
where $\mathcal{U}_S$ is the trajectory class for $S$, $\mu_S$ a trajectory distribution, and $K_{A,S}^{(\Theta)}$ are the axiom defects ({prf:ref}`def-parametric-defect-functional`).

- The **average defect risk** over a distribution $\mathcal{S}$ is:
$$\mathcal{R}_{\mathcal{S}}(\Theta) := \mathbb{E}_{S \sim \mathcal{S}}[\mathcal{R}_S(\Theta)].$$

- We consider two distributions $\mathcal{S}_{\mathrm{train}}$ and $\mathcal{S}_{\mathrm{test}}$. For simplicity we first treat the $\mathcal{S}_{\mathrm{train}} = \mathcal{S}_{\mathrm{test}}$ case, then note the extension to covariant shifts.

#### Structural manifold of true hypostructures

We assume that for each system $S$ in the support of $\mathcal{S}$, there exists a "true" parameter $\Theta^*(S) \in \Theta_{\mathrm{adm}}$ such that:

- $\mathcal{H}_{\Theta^*(S),S}$ satisfies the hypostructure axioms (C, D, SC, Cap, LS, TB, Bound, Reg, GC) for that system, including boundary trace/flux/reinjection compatibility;

- all axiom defects vanish for the true parameter:
$$\mathcal{R}_S(\Theta^*(S)) = 0, \qquad K_{A,S}^{(\Theta^*(S))}(u) = 0 \quad \mu_S\text{-a.e. for all } A \in \mathcal{A};$$

- $\Theta^*(S)$ is uniquely determined up to Hypo-isomorphism by the structural data $(\Phi_{\Theta^*(S),S}, \mathfrak{D}_{\Theta^*(S),S}, \mathcal{B}_{\Theta^*(S),S}, \mathrm{Tr}_{\Theta^*(S),S}, \mathcal{J}_{\Theta^*(S),S}, \mathcal{R}_{\Theta^*(S),S}, \ldots)$ (structural identifiability, as in {prf:ref}`mt-sv-09-meta-identifiability`).

We further assume that the map $S \mapsto \Theta^*(S)$ takes values in a compact $C^1$ submanifold $\mathcal{M} \subset \Theta_{\mathrm{adm}}$, which we call the **structural manifold**. Intuitively, $\mathcal{M}$ collects all true hypostructure parameters realized by systems in the support of $\mathcal{S}$.

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **True Hypostructures on Compact Structural Manifold:** $\mathcal{R}_S(\Theta^*(S)) = 0$ for $\mathcal{S}$-a.e. $S$, $\mathcal{M} \subset \Theta_{\mathrm{adm}}$ compact $C^1$ submanifold
>     *   [ ] **Uniform Local Strong Convexity:** $c \, \mathrm{dist}(\Theta, \mathcal{M})^2 \leq \mathcal{R}_S(\Theta) \leq C \, \mathrm{dist}(\Theta, \mathcal{M})^2$ near $\mathcal{M}$
>     *   [ ] **Lipschitz Continuity:** $|\mathcal{R}_S(\Theta) - \mathcal{R}_{S'}(\Theta')| \leq L(d_{\mathcal{S}}(S, S') + |\Theta - \Theta'|)$
>     *   [ ] **Approximate Empirical Minimization:** $\widehat{\mathcal{R}}_N(\widehat{\Theta}_N) \leq \inf_{\Theta} \widehat{\mathcal{R}}_N(\Theta) + \varepsilon_N$
> *   **Output (Structural Guarantee):**
>     *   Generalization: $\mathcal{R}_{\mathcal{S}}(\widehat{\Theta}_N) \leq C_1(\varepsilon_N + \sqrt{\log(1/\delta)/N})$
>     *   Structural recovery: $\mathbb{E}[\mathrm{dist}(\widehat{\Theta}_N, \Theta^*(S))] \leq C_2\sqrt{\varepsilon_N + \sqrt{\log(1/\delta)/N}}$
>     *   Asymptotic consistency as $N \to \infty$, $\varepsilon_N \to 0$
> *   **Failure Condition (Debug):**
>     *   If **Structural Manifold** non-compact → **Mode overfitting** (learned structure specific to training systems)
>     *   If **Lipschitz** fails → **Mode distribution shift** (test systems structurally different)

:::{prf:metatheorem} Meta-Generalization
:label: mt-meta-generalization

Let $\mathcal{S}$ be a distribution over systems $S$, and suppose that:

1. **True hypostructures on a compact structural manifold.** For $\mathcal{S}$-a.e. $S$, there exists $\Theta^*(S) \in \Theta_{\mathrm{adm}}$ such that:
   - $\mathcal{R}_S(\Theta^*(S)) = 0$;
   - $\mathcal{H}_{\Theta^*(S),S}$ satisfies the hypostructure axioms (C, D, SC, Cap, LS, TB, Bound, Reg, GC);
   - $\Theta^*(S)$ is structurally identifiable up to Hypo-isomorphism.

   The image $\mathcal{M} := \{\Theta^*(S) : S \in \mathrm{supp}(\mathcal{S})\}$ is contained in a compact $C^1$ submanifold of $\Theta_{\mathrm{adm}}$.

2. **Uniform local strong convexity near the structural manifold.** There exist constants $c, C, \rho > 0$ such that for all $S$ and all $\Theta$ with $\mathrm{dist}(\Theta, \mathcal{M}) \leq \rho$:
$$c \, \mathrm{dist}(\Theta, \mathcal{M})^2 \leq \mathcal{R}_S(\Theta) \leq C \, \mathrm{dist}(\Theta, \mathcal{M})^2.$$
(Here $\mathrm{dist}$ is taken modulo gauge; this is the multi-task version of the local quadratic bounds from {prf:ref}`mt-trainable-hypostructure-consistency` for a single system.)

3. **Lipschitz continuity of risk in $\Theta$ and $S$.** There exists $L > 0$ such that for all $S, S'$ and $\Theta, \Theta'$ in a neighborhood of $\mathcal{M}$:
$$|\mathcal{R}_S(\Theta) - \mathcal{R}_{S'}(\Theta')| \leq L \big( d_{\mathcal{S}}(S, S') + |\Theta - \Theta'| \big),$$
where $d_{\mathcal{S}}$ is a metric on the space of systems compatible with $\mathcal{S}$ and controls boundary mismatch (e.g. the induced distance between boundary interfaces in the thin-interface sense).

4. **Approximate empirical minimization on training systems.** Let $S_1, \ldots, S_N$ be i.i.d. samples from $\mathcal{S}$. Define the empirical average risk:
$$\widehat{\mathcal{R}}_N(\Theta) := \frac{1}{N} \sum_{i=1}^N \mathcal{R}_{S_i}(\Theta).$$
Suppose $\widehat{\Theta}_N \in \Theta_{\mathrm{adm}}$ satisfies:
$$\widehat{\mathcal{R}}_N(\widehat{\Theta}_N) \leq \inf_{\Theta} \widehat{\mathcal{R}}_N(\Theta) + \varepsilon_N,$$
for some optimization accuracy $\varepsilon_N \geq 0$.

Then, with probability at least $1 - \delta$ over the draw of the $S_i$, the following hold for $N$ large enough:

1. **(Average generalization of defect risk.)** There exists a constant $C_1$, depending only on the structural manifold and the Lipschitz/convexity constants in (2)–(3), such that:
$$\mathcal{R}_{\mathcal{S}}(\widehat{\Theta}_N) := \mathbb{E}_{S \sim \mathcal{S}}[\mathcal{R}_S(\widehat{\Theta}_N)] \leq C_1 \left( \varepsilon_N + \sqrt{\frac{\log(1/\delta)}{N}} \right).$$

2. **(Average closeness to true hypostructures.)** There exists a constant $C_2 > 0$ such that:
$$\mathbb{E}_{S \sim \mathcal{S}} \big[ \mathrm{dist}(\widehat{\Theta}_N, \Theta^*(S)) \big] \leq C_2 \sqrt{ \varepsilon_N + \sqrt{\tfrac{\log(1/\delta)}{N}} }.$$

3. **(Convergence as $N \to \infty$.)** In particular, if $\varepsilon_N \to 0$ as $N \to \infty$, then:
$$\lim_{N \to \infty} \mathcal{R}_{\mathcal{S}}(\widehat{\Theta}_N) = 0, \qquad \lim_{N \to \infty} \mathbb{E}_{S \sim \mathcal{S}} \big[ \mathrm{dist}(\widehat{\Theta}_N, \Theta^*(S)) \big] = 0,$$
i.e. the learned parameter $\widehat{\Theta}_N$ yields hypostructures that are asymptotically axiom-consistent and structurally correct on average across systems drawn from $\mathcal{S}$.
:::

:::{prf:proof}
By assumption (1), zero-risk parameters for each system lie on the manifold $\mathcal{M}$. For any $\Theta$ close to $\mathcal{M}$, the uniform quadratic bound (2) implies:
$$c \, \mathrm{dist}(\Theta, \mathcal{M})^2 \leq \mathcal{R}_S(\Theta) \leq C \, \mathrm{dist}(\Theta, \mathcal{M})^2 \quad \text{for all } S.$$

Taking expectations over $S \sim \mathcal{S}$ gives:
$$c \, \mathrm{dist}(\Theta, \mathcal{M})^2 \leq \mathcal{R}_{\mathcal{S}}(\Theta) \leq C \, \mathrm{dist}(\Theta, \mathcal{M})^2.$$

Thus small average risk and small average distance to $\mathcal{M}$ are equivalent up to constants.

Next, $\mathcal{R}_S(\Theta)$ is bounded and Lipschitz in $\Theta$ and $S$ by (3), so standard uniform convergence arguments (e.g. covering number or Rademacher complexity bounds on the function class $\{\mathcal{R}_S(\cdot) : S \in \mathrm{supp}(\mathcal{S})\}$) imply that, with probability at least $1 - \delta$:
$$\sup_{\Theta \in \Theta_{\mathrm{adm}}} \left| \widehat{\mathcal{R}}_N(\Theta) - \mathcal{R}_{\mathcal{S}}(\Theta) \right| \leq C_3 \sqrt{\frac{\log(1/\delta)}{N}},$$
for some constant $C_3$ depending on the Lipschitz constants and the metric entropy of $\Theta_{\mathrm{adm}}$.

By the approximate minimization condition:
$$\widehat{\mathcal{R}}_N(\widehat{\Theta}_N) \leq \widehat{\mathcal{R}}_N(\Theta_{\mathcal{M}}^*) + \varepsilon_N,$$
where $\Theta_{\mathcal{M}}^* \in \mathcal{M}$ is any selector (e.g. minimizing $\mathcal{R}_{\mathcal{S}}$ over $\mathcal{M}$, which is zero by (1)). Using uniform convergence, we get:
$$\mathcal{R}_{\mathcal{S}}(\widehat{\Theta}_N) \leq \widehat{\mathcal{R}}_N(\widehat{\Theta}_N) + C_3 \sqrt{\tfrac{\log(1/\delta)}{N}} \leq \widehat{\mathcal{R}}_N(\Theta_{\mathcal{M}}^*) + \varepsilon_N + C_3 \sqrt{\tfrac{\log(1/\delta)}{N}} \leq \mathcal{R}_{\mathcal{S}}(\Theta_{\mathcal{M}}^*) + 2C_3 \sqrt{\tfrac{\log(1/\delta)}{N}} + \varepsilon_N.$$

But $\mathcal{R}_{\mathcal{S}}(\Theta_{\mathcal{M}}^*) = 0$ by construction, so:
$$\mathcal{R}_{\mathcal{S}}(\widehat{\Theta}_N) \leq \varepsilon_N + 2C_3 \sqrt{\tfrac{\log(1/\delta)}{N}}.$$
This gives (1), up to renaming constants.

Applying the lower bound in (2) to $\Theta = \widehat{\Theta}_N$:
$$c \, \mathrm{dist}(\widehat{\Theta}_N, \mathcal{M})^2 \leq \mathcal{R}_{\mathcal{S}}(\widehat{\Theta}_N),$$
and combining with the upper bound just obtained yields:
$$\mathrm{dist}(\widehat{\Theta}_N, \mathcal{M}) \leq C_4 \sqrt{ \varepsilon_N + \sqrt{\tfrac{\log(1/\delta)}{N}} },$$
for some constant $C_4$. Since for each $S$ the minimizer set $\{\Theta^*(S)\} \subset \mathcal{M}$, the distance to $\Theta^*(S)$ is bounded by the distance to $\mathcal{M}$, giving (2).

The convergence statements in (3) follow immediately when $\varepsilon_N \to 0$ and $N \to \infty$.
:::

:::{prf:remark} Interpretation

The theorem shows that **average defect minimization over a distribution of systems** is a consistent procedure: if each system admits a hypostructure in the parametric family and the structural manifold is well-behaved, then a trainable hypostructure that approximately minimizes empirical defect risk on finitely many training systems will, with high probability, yield **globally good** hypostructures for new systems drawn from the same structural class.
:::

:::{prf:remark} Covariate shift

Extensions to a **covariately shifted test distribution** $\mathcal{S}_{\mathrm{test}}$ (e.g. different but structurally equivalent systems) follow by the same argument, provided the map $S \mapsto \Theta^*(S)$ is Lipschitz between the supports of $\mathcal{S}_{\mathrm{train}}$ and $\mathcal{S}_{\mathrm{test}}$.
:::

:::{prf:remark} Motivic Interpretation

In the $\infty$-categorical framework ({prf:ref}`def-categorical-hypostructure`), Meta-Generalization admits a deeper interpretation via **Motivic Integration** {cite}`Kontsevich95` and {cite}`DenefLoeser01`. The learner does not merely fit parameters; it extracts the **Motive** of the system---an object in the Grothendieck ring of varieties $K_0(\text{Var}_k)$.
:::

Specifically, define the **error variety** for parameter $\Theta$ over a field $k$:
$$\mathcal{E}_\Theta := \{(S, u) \in \text{Syst} \times \mathcal{X} : \mathcal{R}_S(\Theta)(u) > 0\} \subset \text{Syst} \times \mathcal{X}$$

The "loss function" $\mathcal{R}_{\mathcal{S}}(\Theta)$ is then the **motivic volume**:
$$\mathcal{R}_{\mathcal{S}}(\Theta) = \int_{\mathcal{E}_\Theta} \mathbb{L}^{-\dim} \, d\chi = \chi(\mathcal{E}_\Theta) \cdot \mathbb{L}^{-n}$$
where $\chi$ is the Euler characteristic and $\mathbb{L} = [\mathbb{A}^1]$ is the Lefschetz motive.

**Key Property (Motivic Invariance):** The learned structure is **base-change invariant**: by Grothendieck's trace formula, if $\widehat{\Theta}_N$ minimizes the motivic volume over $\mathbb{R}$, it also minimizes over $\mathbb{C}$, $\mathbb{Q}_p$, and finite fields $\mathbb{F}_q$. This provides:
- **Transfer learning**: Structure learned over reals transfers to complex systems
- **Field-independence**: The motive $[\mathcal{M}] \in K_0(\text{Var})$ is an absolute invariant
- **Categorical universality**: The structural manifold $\mathcal{M}$ is defined by its functor of points, not by equations in a specific field

The convergence $\widehat{\Theta}_N \to \mathcal{M}$ is thus **motivically universal**—it holds regardless of the base field, guaranteeing that learned hypostructures are not artifacts of the training domain but genuine structural invariants.

> **Key Insight:** This gives the meta-learning framework a rigorous "meta-generalization" layer: trainable hypostructures do not just fit one system, but converge (in risk and in parameter space) to the correct structural manifold across a whole family of systems. In the motivic interpretation, the learner extracts a **universal motive**---the abstract "essence" of the system class that transcends any particular instantiation.

### Expressivity of Trainable Hypostructures

Up to now we have assumed that the "true" hypostructure for a given system $S$ lives *inside* our parametric family $\{\mathcal{H}_\Theta\}_{\Theta \in \Theta_{\mathrm{adm}}}$. In practice, this is an idealization: the true structure might lie outside our chosen parametrization, but we still expect to approximate it arbitrarily well.

In this section we formalize this as an **expressivity / approximation** property: the parametric hypostructure family is rich enough that any admissible hypostructure satisfying the axioms can be approximated (in structural data) to arbitrary accuracy, and the **axiom-defect risk** then goes to zero.

#### Structural metric on hypostructures

Fix a system $S$ with state space $X$ and semiflow $S_t$. Let $\mathfrak{H}(S)$ denote the class of hypostructures on $S$ of the form:
$$\mathcal{H} = (X, S_t, \Phi, \mathfrak{D}, G, \mathcal{B}, \mathrm{Tr}, \mathcal{J}, \mathcal{R})$$
satisfying the axioms (C, D, SC, Cap, LS, TB, Bound, Reg, GC) and a uniform regularity condition (e.g. Lipschitz bounds on $\Phi, \mathfrak{D}$, bounded barrier constants, and bounded boundary flux).

We define a **structural metric**:
$$d_{\mathrm{struct}} : \mathfrak{H}(S) \times \mathfrak{H}(S) \to [0, \infty)$$
by choosing a reference measure $\nu$ on $X$ (e.g. invariant or finite-energy measure) and setting:
$$d_{\mathrm{struct}}(\mathcal{H}, \mathcal{H}') := \|\Phi - \Phi'\|_{L^\infty(X, \nu)} + \|\mathfrak{D} - \mathfrak{D}'\|_{L^\infty(X, \nu)} + \mathrm{dist}_G(G, G') + \mathrm{dist}_{\partial}((\mathcal{B}, \mathrm{Tr}, \mathcal{J}, \mathcal{R}), (\mathcal{B}', \mathrm{Tr}', \mathcal{J}', \mathcal{R}')),$$
where $\mathrm{dist}_G$ is any metric on the structural data $G$ (capacities, sectors, barrier constants, exponents) compatible with the topology used in Parts VI–X, and $\mathrm{dist}_{\partial}$ is any metric on boundary data that controls trace/flux/reinjection (e.g. $L^\infty$ bounds on $\mathcal{J}$, operator norms for $\mathrm{Tr}$, and a Wasserstein/KL distance on $\mathcal{R}$ where defined). Two hypostructures that differ only by a Hypo-isomorphism are identified in this metric (i.e. we work modulo gauge).

#### Universal structural approximation

Let $\{\mathcal{H}_\Theta\}_{\Theta \in \Theta_{\mathrm{adm}}}$ be a parametric family of hypostructures on $S$:
$$\mathcal{H}_\Theta = (X, S_t, \Phi_\Theta, \mathfrak{D}_\Theta, G_\Theta, \mathcal{B}_\Theta, \mathrm{Tr}_\Theta, \mathcal{J}_\Theta, \mathcal{R}_\Theta).$$

We say this family is **universally structurally approximating** on $\mathfrak{H}(S)$ if (this generalizes the Stone-Weierstrass theorem to dynamical functionals, similar to the universality of flow approximation in {cite}`Ornstein74`):

> For every $\mathcal{H}^* = (X, S_t, \Phi^*, \mathfrak{D}^*, G^*) \in \mathfrak{H}(S)$ and every $\delta > 0$, there exists $\Theta \in \Theta_{\mathrm{adm}}$ such that:
> $$d_{\mathrm{struct}}(\mathcal{H}_\Theta, \mathcal{H}^*) < \delta.$$

Intuitively, $\{\mathcal{H}_\Theta\}$ can approximate any admissible hypostructure arbitrarily well in energy, dissipation, and barrier data.

#### Continuity of defects with respect to structure

Recall that for each axiom $A \in \mathcal{A}$ and trajectory $u \in \mathcal{U}_S$, the defect functional $K_A^{(\Theta)}(u)$ is defined in terms of $(\Phi_\Theta, \mathfrak{D}_\Theta, G_\Theta, \mathcal{B}_\Theta, \mathrm{Tr}_\Theta, \mathcal{J}_\Theta, \mathcal{R}_\Theta)$ and the axioms (C, D, SC, Cap, LS, TB, Bound). Denote by $K_A^{(\mathcal{H})}(u)$ the corresponding defect when computed from a general hypostructure $\mathcal{H} \in \mathfrak{H}(S)$.

We assume:

> **Defect continuity.** There exists a constant $L_A > 0$ such that for all hypostructures $\mathcal{H}, \mathcal{H}' \in \mathfrak{H}(S)$, all trajectories $u \in \mathcal{U}_S$, and all $A \in \mathcal{A}$:
> $$\big| K_A^{(\mathcal{H})}(u) - K_A^{(\mathcal{H}')}(u) \big| \leq L_A \, d_{\mathrm{struct}}(\mathcal{H}, \mathcal{H}').$$
>
> Equivalently, the mapping $\mathcal{H} \mapsto K_A^{(\mathcal{H})}(u)$ is Lipschitz with respect to the structural metric, uniformly over $u$ in the support of the trajectory measure $\mu_S$.

This is a natural assumption given the explicit integral definitions of the defects (e.g. $K_D$ is an integral of the positive part of $\partial_t \Phi + \mathfrak{D}$, capacities/barriers enter via continuous inequalities, and $K_{Bound}$ is controlled by trace/flux/reinjection deviations measured by $\mathrm{dist}_{\partial}$).

:::{prf:remark} Boundary-sensitive expressivity
Because $d_{\mathrm{struct}}$ includes $\mathrm{dist}_{\partial}$, universal approximation in this section requires the parametric family to approximate boundary interfaces as well as bulk dynamics. In particular, $K_{Bound}$ can only be driven to zero if $(\mathcal{B}_\Theta, \mathrm{Tr}_\Theta, \mathcal{J}_\Theta, \mathcal{R}_\Theta)$ converge in the boundary metric.
:::

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **True Admissible Hypostructure:** $\mathcal{H}^* \in \mathfrak{H}(S)$ satisfying axioms with $K_A^{(\mathcal{H}^*)}(u) = 0$ $\mu_S$-a.e.
>     *   [ ] **Universally Structurally Approximating Family:** For all $\mathcal{H}^* \in \mathfrak{H}(S)$, $\forall \delta > 0$, $\exists \Theta$: $d_{\mathrm{struct}}(\mathcal{H}_\Theta, \mathcal{H}^*) < \delta$
>     *   [ ] **Defect Continuity:** $|K_A^{(\mathcal{H})}(u) - K_A^{(\mathcal{H}')}(u)| \leq L_A d_{\mathrm{struct}}(\mathcal{H}, \mathcal{H}')$ uniformly in $u$
> *   **Output (Structural Guarantee):**
>     *   Approximate realizability: $\forall \varepsilon > 0$, $\exists \Theta_\varepsilon$: $\mathcal{R}_S(\Theta_\varepsilon) \leq \varepsilon$
>     *   Infimum: $\inf_{\Theta} \mathcal{R}_S(\Theta) = 0$
>     *   Quantitative bound: $d_{\mathrm{struct}}(\mathcal{H}_\Theta, \mathcal{H}^*) \leq \delta \Rightarrow \mathcal{R}_S(\Theta) \leq (\sum_A w_A L_A)\delta$
> *   **Failure Condition (Debug):**
>     *   If **Universal Approximation** fails → **Mode expressivity gap** (parametric family too restrictive)
>     *   If **Defect Continuity** fails → **Mode discontinuous sensitivity** (small structural changes cause large defect jumps)

:::{prf:metatheorem} Axiom-Expressivity
:label: mt-axiom-expressivity

Let $S$ be a fixed system with trajectory distribution $\mu_S$ and trajectory class $\mathcal{U}_S$. Let $\mathfrak{H}(S)$ be the class of admissible hypostructures on $S$ as above. Suppose:

1. **(True admissible hypostructure.)** There exists a "true" hypostructure $\mathcal{H}^* \in \mathfrak{H}(S)$ which exactly satisfies the axioms (C, D, SC, Cap, LS, TB, Bound, Reg, GC) for $S$. Thus, for $\mu_S$-a.e. trajectory $u$:
$$K_A^{(\mathcal{H}^*)}(u) = 0 \quad \forall A \in \mathcal{A}.$$

2. **(Universally structurally approximating family.)** The parametric family $\{\mathcal{H}_\Theta\}_{\Theta \in \Theta_{\mathrm{adm}}}$ is universally structurally approximating on $\mathfrak{H}(S)$ in the sense above.

3. **(Defect continuity.)** Each defect functional $K_A^{(\mathcal{H})}(u)$ is Lipschitz in $\mathcal{H}$ with respect to $d_{\mathrm{struct}}$, uniformly in $u$ (defect continuity).

Define the joint defect risk of parameter $\Theta$ on system $S$ by:
$$\mathcal{R}_S(\Theta) := \sum_{A \in \mathcal{A}} w_A \int_{\mathcal{U}_S} K_A^{(\Theta)}(u) \, d\mu_S(u),$$
where $K_A^{(\Theta)} := K_A^{(\mathcal{H}_\Theta)}$ and $w_A \geq 0$ are fixed weights.

Then:

1. **(Approximate realizability of zero-risk.)** For every $\varepsilon > 0$ there exists $\Theta_\varepsilon \in \Theta_{\mathrm{adm}}$ such that:
$$\mathcal{R}_S(\Theta_\varepsilon) \leq \varepsilon.$$
In particular:
$$\inf_{\Theta \in \Theta_{\mathrm{adm}}} \mathcal{R}_S(\Theta) = 0.$$

2. **(Quantitative bound.)** More precisely, if for some $\delta > 0$ we pick $\Theta$ such that:
$$d_{\mathrm{struct}}(\mathcal{H}_\Theta, \mathcal{H}^*) \leq \delta,$$
then:
$$\mathcal{R}_S(\Theta) \leq \left( \sum_{A \in \mathcal{A}} w_A L_A \right) \delta.$$
In particular, $\mathcal{R}_S(\Theta_\varepsilon) \leq \varepsilon$ holds whenever:
$$d_{\mathrm{struct}}(\mathcal{H}_{\Theta_\varepsilon}, \mathcal{H}^*) \leq \frac{\varepsilon}{\sum_A w_A L_A}.$$

In words: **any admissible true hypostructure can be approximated arbitrarily well by the trainable family, and the corresponding defect risk can be driven arbitrarily close to zero**.
:::

:::{prf:proof}
Fix $\varepsilon > 0$. Let $L := \sum_{A \in \mathcal{A}} w_A L_A$, where the $L_A$'s are the Lipschitz constants from defect continuity.

By universal structural approximation (assumption 2), there exists $\Theta_\varepsilon \in \Theta_{\mathrm{adm}}$ such that:
$$d_{\mathrm{struct}}(\mathcal{H}_{\Theta_\varepsilon}, \mathcal{H}^*) \leq \delta_\varepsilon := \frac{\varepsilon}{L}.$$

For any $A \in \mathcal{A}$ and trajectory $u$:
$$\big| K_A^{(\Theta_\varepsilon)}(u) - K_A^{(\mathcal{H}^*)}(u) \big| = \big| K_A^{(\mathcal{H}_{\Theta_\varepsilon})}(u) - K_A^{(\mathcal{H}^*)}(u) \big| \leq L_A \, d_{\mathrm{struct}}(\mathcal{H}_{\Theta_\varepsilon}, \mathcal{H}^*) \leq L_A \delta_\varepsilon.$$

But $K_A^{(\mathcal{H}^*)}(u) = 0$ $\mu_S$-a.s. by assumption (1), so:
$$K_A^{(\Theta_\varepsilon)}(u) \leq L_A \delta_\varepsilon \quad \text{for } \mu_S\text{-a.e. } u.$$

Integrating with respect to $\mu_S$:
$$\mathcal{R}_{A,S}(\Theta_\varepsilon) = \int_{\mathcal{U}_S} K_A^{(\Theta_\varepsilon)}(u) \, d\mu_S(u) \leq L_A \delta_\varepsilon.$$

Therefore:
$$\mathcal{R}_S(\Theta_\varepsilon) = \sum_{A \in \mathcal{A}} w_A \mathcal{R}_{A,S}(\Theta_\varepsilon) \leq \sum_{A \in \mathcal{A}} w_A (L_A \delta_\varepsilon) = \left( \sum_{A \in \mathcal{A}} w_A L_A \right) \delta_\varepsilon = L \cdot \frac{\varepsilon}{L} = \varepsilon.$$

This proves the quantitative bound and, in particular, the existence of parameters $\Theta_\varepsilon$ with $\mathcal{R}_S(\Theta_\varepsilon) \leq \varepsilon$ for every $\varepsilon > 0$. Taking the infimum over $\Theta$ and letting $\varepsilon \to 0$ yields:
$$\inf_{\Theta \in \Theta_{\mathrm{adm}}} \mathcal{R}_S(\Theta) = 0.
$$
:::

:::{prf:remark} No expressivity bottleneck

The theorem isolates **what is needed** for axiom-expressivity:

- a structural metric $d_{\mathrm{struct}}$ capturing the relevant pieces of hypostructure data,
- universal approximation of $(\Phi, \mathfrak{D}, G)$ in that metric,
- and Lipschitz dependence of defects on structural data.

No optimization assumptions are used: this is a **pure representational metatheorem**. Combined with the trainability and convergence metatheorem ({prf:ref}`mt-trainable-hypostructure-consistency`), it implies that the only remaining obstacles are optimization and data, not the expressivity of the hypostructure family.
:::

> **Key Insight:** The parametric family is **axiom-complete**: any structurally admissible dynamics can be encoded with arbitrarily small axiom defects. The only limitations are optimization and data, not the hypothesis class.

### Active Probing and Sample-Complexity of Hypostructure Identification

So far we have treated the axiom-defect risk as given by a fixed trajectory distribution $\mu_S$. In many systems, however, the learner can **control** which trajectories are generated, by choosing initial conditions and controls. In other words, the learner can design *experiments*. This section formalizes optimal experiment design for structural identification, extending the classical **observability** framework of Kalman {cite}`Kalman60` to the hypostructure setting. This guarantees **Identification in the Limit**, satisfying the criteria of **Gold's Paradigm {cite}`Gold67`** for language learning.

In this section we show that, under a mild identifiability gap assumption, **actively chosen probes** (policies, initial data, controls) allow the learner to identify the correct hypostructure parameter with sample complexity essentially proportional to the parameter dimension and inverse-quadratic in the identifiability gap.

#### Probes and defect observations

Fix a system $S$ with state space $X$, trajectory space $\mathcal{U}_S$, and a parametric hypostructure family $\{\mathcal{H}_\Theta\}_{\Theta \in \Theta_{\mathrm{adm}}}$. We assume we can influence trajectories via a class of **probes**:
$$\pi \in \mathfrak{P},$$
where each $\pi$ denotes a rule for generating a trajectory $u_{S,\Theta,\pi} \in \mathcal{U}_S$ (e.g. a choice of initial condition and/or control policy). For each probe $\pi$ and parameter $\Theta$, we can evaluate the axiom defect functionals on the resulting trajectory.

To simplify notation, write:
$$K^{(\Theta)}(S, \pi) := \big( K_A^{(\Theta)}(u_{S,\Theta,\pi}) \big)_{A \in \mathcal{A}} \in \mathbb{R}^{|\mathcal{A}|}_{\geq 0}$$
for the **defect fingerprint** induced by parameter $\Theta$ on system $S$ under probe $\pi$, and:
$$D(\Theta, \Theta'; S, \pi) := \big| K^{(\Theta)}(S, \pi) - K^{(\Theta')}(S, \pi) \big|$$
for its distance (e.g. $\ell^1$ or $\ell^2$ norm) between two parameters.

In practice, the defects may be observed with noise. We thus write a single **noisy observation** of the defect fingerprint as:
$$Y_t = K^{(\Theta^*)}(S, \pi_t) + \xi_t,$$
where $\Theta^*$ is the true parameter and $\pi_t$ is the probe chosen at round $t$. The noise $\xi_t$ takes values in $\mathbb{R}^{|\mathcal{A}|}$ and models discretization error, finite sampling of trajectories, measurement noise, etc.

:::{prf:definition} Probe-wise identifiability gap
:label: def-probe-wise-identifiability-gap

Let $\Theta^* \in \Theta_{\mathrm{adm}}$ be the true parameter. We say that a class of probes $\mathfrak{P}$ has a **uniform identifiability gap** $\Delta > 0$ around $\Theta^*$ if there exist constants $\Delta > 0$ and $r > 0$ such that for every $\Theta \in \Theta_{\mathrm{adm}}$ with $|\Theta - \Theta^*| \geq r$:
$$\sup_{\pi \in \mathfrak{P}} D(\Theta, \Theta^*; S, \pi) \geq \Delta.$$
:::

Equivalently: no parameter at distance at least $r$ from $\Theta^*$ can mimic the defect fingerprints of $\Theta^*$ under *all* probes; there is always some probe that amplifies the discrepancy to at least $\Delta$ in defect space.

:::{prf:assumption} Sub-Gaussian defect noise
:label: assum-sub-gaussian-defect-noise

The noise variables $\xi_t$ are independent, mean-zero, and $\sigma$-sub-Gaussian in each coordinate:
$$\mathbb{E}[\xi_t] = 0, \quad \mathbb{E}\big[ \exp(\lambda \xi_{t,j}) \big] \leq \exp\Big( \tfrac{1}{2} \sigma^2 \lambda^2 \Big) \quad \forall \lambda \in \mathbb{R}, \forall t, \forall j.$$

Moreover, $\xi_t$ is independent of the probe choices $\pi_s$ and the past noise $\xi_s$ for $s < t$.
:::

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Local Identifiability via Defects:** $\sup_\pi D(\Theta, \Theta^*; S, \pi) \leq \delta \Rightarrow |\Theta - \Theta^*| \leq c\delta$ ({prf:ref}`mt-trainable-hypostructure-consistency`, {prf:ref}`mt-sv-09-meta-identifiability`)
>     *   [ ] **Probe-Wise Identifiability Gap:** $\sup_{\pi \in \mathfrak{P}} D(\Theta, \Theta^*; S, \pi) \geq \Delta$ for $|\Theta - \Theta^*| \geq r$
>     *   [ ] **Sub-Gaussian Defect Noise:** $\xi_t$ independent, mean-zero, $\sigma$-sub-Gaussian
>     *   [ ] **Local Regularity:** $|K^{(\Theta)}(S, \pi) - K^{(\Theta')}(S, \pi)| \leq L|\Theta - \Theta'|$ uniformly in $\pi$
> *   **Output (Structural Guarantee):**
>     *   Sample complexity: $T \gtrsim \frac{d \sigma^2}{\Delta^2} \log(1/\delta)$ for $\varepsilon$-identification
>     *   Adaptive probing strategy achieves optimal rate
>     *   Linear scaling in parameter dimension $d$, inverse-quadratic in gap $\Delta$
> *   **Failure Condition (Debug):**
>     *   If **Identifiability Gap** $\Delta = 0$ → **Mode structural aliasing** (indistinguishable parameters under all probes)
>     *   If **Noise** non-sub-Gaussian → **Mode heavy-tailed contamination** (slower convergence rates)

:::{prf:metatheorem} Optimal Experiment Design
:label: mt-optimal-experiment-design

Let $S$ be a fixed system and $\Theta^* \in \Theta_{\mathrm{adm}}$ the true hypostructure parameter. Assume:

1. **(Local identifiability via defects.)** The single-system identifiability metatheorem holds for $S$: small uniform defect discrepancies imply small parameter distance, as in {prf:ref}`mt-trainable-hypostructure-consistency` and {prf:ref}`mt-sv-09-meta-identifiability`. In particular, there exist constants $c > 0$ and $\rho > 0$ such that:
$$\sup_{\pi \in \mathfrak{P}} D(\Theta, \Theta^*; S, \pi) \leq \delta \implies |\Theta - \Theta^*| \leq c \delta$$
for all $\Theta$ with $|\Theta - \Theta^*| \leq \rho$.

2. **(Probe-wise identifiability gap.)** The probe class $\mathfrak{P}$ has a uniform identifiability gap $\Delta > 0$ in the sense of {prf:ref}`def-probe-wise-identifiability-gap`, with some radius $r > 0$.

3. **(Sub-Gaussian defect noise.)** The noise model of {prf:ref}`assum-sub-gaussian-defect-noise` holds with parameter $\sigma > 0$.

4. **(Local regularity.)** The map $\Theta \mapsto K^{(\Theta)}(S, \pi)$ is Lipschitz in $\Theta$ uniformly over $\pi \in \mathfrak{P}$ in a neighborhood of $\Theta^*$:
$$\big| K^{(\Theta)}(S, \pi) - K^{(\Theta')}(S, \pi) \big| \leq L |\Theta - \Theta'| \quad \text{for } |\Theta - \Theta^*|, |\Theta' - \Theta^*| \leq \rho.$$

Consider an **adaptive probing strategy** over $T$ rounds:

- At round $t$ we choose a probe $\pi_t = \pi_t(\mathcal{F}_{t-1}) \in \mathfrak{P}$, where $\mathcal{F}_{t-1}$ is the sigma-algebra generated by past probes and observations $\{(\pi_s, Y_s)\}_{s < t}$.
- We observe a noisy defect fingerprint $Y_t = K^{(\Theta^*)}(S, \pi_t) + \xi_t$.
- After $T$ rounds, we output an estimator $\widehat{\Theta}_T$ that is measurable with respect to $\mathcal{F}_T$.

Then there exists an adaptive probing strategy and an estimator $\widehat{\Theta}_T$ such that for any confidence level $\delta \in (0, 1)$, we have:
$$\mathbb{P}\big( |\widehat{\Theta}_T - \Theta^*| \geq \varepsilon \big) \leq \delta$$
whenever:
$$T \gtrsim \frac{d \, \sigma^2}{\Delta^2} \log \frac{1}{\delta},$$
where $d := \dim(\Theta_{\mathrm{adm}})$, and the implicit constant depends only on the Lipschitz/identifiability constants $L, c, \rho$.

In particular, the sample complexity of identifying the correct hypostructure parameter up to accuracy $\varepsilon$ with high probability scales at most linearly in the parameter dimension and inverse-quadratically in the identifiability gap $\Delta$.
:::

:::{prf:proof}
We provide a rigorous argument based on $\varepsilon$-net discretization and uniform concentration bounds.

**Step 1 (Discretize parameter space).** Restrict attention to a compact neighborhood $B(\Theta^*, R) \subset \Theta_{\mathrm{adm}}$. For a given accuracy scale $\varepsilon > 0$, construct a minimal $\varepsilon$-net $\mathcal{N}_\varepsilon \subset B(\Theta^*, R)$ in parameter space. By standard metric entropy bounds, the covering number satisfies:
$$N(\varepsilon, B(\Theta^*, R), \|\cdot\|) \leq \left(\frac{3R}{\varepsilon}\right)^d$$
where $d = \dim(\Theta_{\mathrm{adm}})$.

**Step 2 (Uniform separation via probes).** Define the separation function $\Delta(\Theta, \Theta') := \sup_{\pi \in \mathfrak{P}} |K^{(\Theta)}(S, \pi) - K^{(\Theta')}(S, \pi)|$. By the identifiability gap assumption, $|\Theta - \Theta^*| \geq r$ implies $\Delta(\Theta, \Theta^*) \geq \Delta$. By Lipschitz continuity of the defect kernel in $\Theta$, for any $\Theta' \in \mathcal{N}_\varepsilon$ with $|\Theta' - \Theta^*| \geq r/2$, there exists $\pi \in \mathfrak{P}$ achieving:
$$\big| K^{(\Theta')}(S, \pi) - K^{(\Theta^*)}(S, \pi) \big| \geq \Delta/2.$$

**Step 3 (Adaptive elimination strategy).** Maintain a candidate set $C_t \subseteq \mathcal{N}_\varepsilon$, initialized as $C_0 = \mathcal{N}_\varepsilon$. At each round $t$:

- Choose probe $\pi_t = \arg\max_{\pi} \mathrm{Var}_{\Theta \in C_{t-1}}[K^{(\Theta)}(S, \pi)]$
- Observe $Y_t = K^{(\Theta^*)}(S, \pi_t) + \xi_t$ with $\xi_t$ sub-Gaussian($\sigma^2$)
- Eliminate: $C_t = \{\Theta \in C_{t-1} : |K^{(\Theta)}(S, \pi_t) - \bar{Y}_t| \leq 2\sigma\sqrt{2\log(2|C_0|T/\delta)/t}\}$


**Lemma (Sub-Gaussian concentration).** For sub-Gaussian noise with parameter $\sigma^2$, after $t$ observations of probe $\pi$, the empirical mean $\bar{Y}_t$ satisfies:
$$\mathbb{P}\left(|\bar{Y}_t - K^{(\Theta^*)}(S, \pi)| > \sigma\sqrt{\frac{2\log(2/\delta)}{t}}\right) \leq \delta$$

By a union bound over $|\mathcal{N}_\varepsilon| \cdot T$ elimination events, any candidate $\Theta'$ with $|K^{(\Theta')} - K^{(\Theta^*)}| \geq \Delta/2$ is eliminated after at most $t \geq 32\sigma^2 \log(2|\mathcal{N}_\varepsilon|T/\delta)/\Delta^2$ probes. The total sample complexity is:
$$T \lesssim \frac{\sigma^2}{\Delta^2} \Big( d \log(R/\varepsilon) + \log \tfrac{1}{\delta} \Big).$$

**Step 4 (Accuracy and parameter error).** After elimination, all remaining candidates $\Theta' \in C_T$ satisfy $|\Theta' - \Theta^*| < r/2$. Output $\widehat{\Theta}_T$ as any element of $C_T$. By the triangle inequality and Lipschitz identifiability, the final estimator's error satisfies $|\widehat{\Theta}_T - \Theta^*| \leq \varepsilon + r/2 = O(\varepsilon)$ when $r = O(\varepsilon)$.
:::

:::{prf:remark} Experiments as a theorem

The theorem shows that **defect-driven experiment design** is not just heuristic: under mild identifiability and regularity assumptions, actively chosen probes let a hypostructure learner identify the correct axioms with sample complexity comparable to classical parametric statistics ($O(d)$ up to logs and $\Delta^{-2}$).
:::

:::{prf:remark} Connection to error localization

This metatheorem pairs naturally with the **meta-error localization** theorem ({prf:ref}`mt-meta-error-localization`): once the learner has identified that an axiom block is wrong, it can design probes specifically targeted to excite that block's defects, further improving the identifiability gap for that block and accelerating correction.
:::

> **Key Insight:** The identifiability gap $\Delta$ is a purely **structural quantity**: it measures how different the defect fingerprints of distinct hypostructures can be made by appropriate experiments. It plays exactly the role of an "information gap" in classical active learning.

### Robustness of Failure-Mode Predictions

A central purpose of a hypostructure is not only to fit trajectories, but to make **sharp structural predictions**: which singularity or breakdown scenarios ("failure modes") are *permitted* or *ruled out* by the axioms, barrier constants, and capacities.

The preceding chapters developed a taxonomy of failure modes and associated **barrier inequalities**: each mode $f$ is excluded when certain barrier constants, exponents, or capacities lie beyond a critical threshold. We now show that, once a trainable hypostructure has sufficiently small axiom-defect risk, its **forbidden failure-mode set** is *exactly the same* as that of the true hypostructure. In other words, the discrete "permit denial" predictions are robust to small learning error.

#### Failure modes and barrier thresholds

Let $\mathcal{F}$ denote the (finite or countable) set of failure modes in the taxonomy (e.g. blow-up, loss of uniqueness, loss of conservation, barrier penetration, glassy obstruction, etc.). For each failure mode $f \in \mathcal{F}$, the structural metatheorems associate:

- a structural functional $B_f(\mathcal{H})$ (a barrier constant, capacity threshold, exponent, or combination thereof);
- a critical value or region $B_f^{\mathrm{crit}}$ such that:

> **Barrier exclusion principle for mode $f$.** If $B_f(\mathcal{H})$ lies in a certain "safe" region (e.g. above a critical constant, or outside a critical set), then failure mode $f$ is forbidden for the hypostructure $\mathcal{H}$. Conversely, if $B_f(\mathcal{H})$ lies in a complementary region, then either $f$ is not ruled out, or there exist sequences of approximate extremals compatible with $f$.

Formally, there is a map $\mathrm{Forbidden}(\mathcal{H}) \subseteq \mathcal{F}$ determined by the structural data $(\Phi, \mathfrak{D}, G)$ and barrier functionals $B_f$, such that:
$$f \in \mathrm{Forbidden}(\mathcal{H}) \iff B_f(\mathcal{H}) \in \mathcal{B}_f^{\mathrm{safe}},$$
where $\mathcal{B}_f^{\mathrm{safe}}$ is the exclusion region in barrier space for mode $f$.

:::{prf:definition} Margin of failure-mode exclusion
:label: def-margin-of-failure-mode-exclusion

Let $\mathcal{H}^*$ be a hypostructure and $f \in \mathrm{Forbidden}(\mathcal{H}^*)$. We say that $\mathcal{H}^*$ excludes $f$ with margin $\gamma_f > 0$ if:
$$\mathrm{dist}\big( B_f(\mathcal{H}^*), \partial \mathcal{B}_f^{\mathrm{safe}} \big) \geq \gamma_f,$$
where $\partial \mathcal{B}_f^{\mathrm{safe}}$ denotes the boundary of the safe region in the barrier space.
:::

We define the **global margin**:
$$\gamma^* := \inf_{f \in \mathrm{Forbidden}(\mathcal{H}^*)} \gamma_f,$$
with the convention $\gamma^* > 0$ if the infimum is over a finite set with strictly positive margins.

:::{prf:assumption} Barrier continuity
:label: assum-barrier-continuity

For each failure mode $f \in \mathcal{F}$, the barrier functional $B_f(\mathcal{H})$ is Lipschitz in the structural metric: there exists $L_f > 0$ such that:
$$\big| B_f(\mathcal{H}) - B_f(\mathcal{H}') \big| \leq L_f \, d_{\mathrm{struct}}(\mathcal{H}, \mathcal{H}') \quad \forall \mathcal{H}, \mathcal{H}' \in \mathfrak{H}(S).$$
:::

:::{prf:assumption} Local structural control by risk
:label: assum-local-structural-control

Let $\mathcal{H}_\Theta$ be a parametric hypostructure family and $\mathcal{H}^*$ the true hypostructure. There exist constants $C_{\mathrm{struct}}, \varepsilon_0 > 0$ such that:
$$\mathcal{R}_S(\Theta) \leq \varepsilon < \varepsilon_0 \implies d_{\mathrm{struct}}(\mathcal{H}_\Theta, \mathcal{H}^*) \leq C_{\mathrm{struct}} \sqrt{\varepsilon}.$$
:::

This is precisely the local quantitative identifiability from {prf:ref}`mt-trainable-hypostructure-consistency`, translated into structural space by the Defect Reconstruction Theorem.

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **True Hypostructure with Strict Exclusion Margin:** $\gamma^* := \inf_{f \in \mathcal{F}_{\mathrm{forbidden}}^*} \mathrm{dist}(B_f(\mathcal{H}^*), \partial \mathcal{B}_f^{\mathrm{safe}}) > 0$
>     *   [ ] **Barrier Continuity:** $|B_f(\mathcal{H}) - B_f(\mathcal{H}')| \leq L_f d_{\mathrm{struct}}(\mathcal{H}, \mathcal{H}')$ for all $f$
>     *   [ ] **Structural Control by Defect Risk:** $\mathcal{R}_S(\Theta) \leq \varepsilon \Rightarrow d_{\mathrm{struct}}(\mathcal{H}_\Theta, \mathcal{H}^*) \leq C_{\mathrm{struct}}\sqrt{\varepsilon}$
> *   **Output (Structural Guarantee):**
>     *   Exact stability: $\mathrm{Forbidden}(\mathcal{H}_\Theta) = \mathrm{Forbidden}(\mathcal{H}^*)$ for $\mathcal{R}_S(\Theta) \leq \varepsilon_1$
>     *   No spurious exclusions: allowed modes remain allowed
>     *   Discrete permit-denial structure recovered exactly for small risk
> *   **Failure Condition (Debug):**
>     *   If **Margin** $\gamma^* = 0$ → **Mode critical boundary** (barrier at threshold, sensitive to perturbation)
>     *   If **Barrier Continuity** fails → **Mode discontinuous classification** (small changes flip forbidden status)

:::{prf:metatheorem} Robustness of Failure-Mode Predictions
:label: mt-robustness-of-failure-mode-predictions

Let $S$ be a system with true hypostructure $\mathcal{H}^* \in \mathfrak{H}(S)$, and let $\{\mathcal{H}_\Theta\}_{\Theta \in \Theta_{\mathrm{adm}}}$ be a parametric family of trainable hypostructures with defect-risk $\mathcal{R}_S(\Theta)$. Assume:

1. **(True hypostructure with strict exclusion margin.)** The true hypostructure $\mathcal{H}^*$ exactly satisfies the axioms (C, D, SC, Cap, LS, TB, Bound, Reg, GC) and excludes a set of failure modes $\mathcal{F}_{\mathrm{forbidden}}^* \subseteq \mathcal{F}$ with positive margin:
$$\gamma^* := \inf_{f \in \mathcal{F}_{\mathrm{forbidden}}^*} \mathrm{dist}\big( B_f(\mathcal{H}^*), \partial \mathcal{B}_f^{\mathrm{safe}} \big) > 0.$$

2. **(Barrier continuity.)** Each barrier functional $B_f(\mathcal{H})$ is Lipschitz with constant $L_f$ with respect to $d_{\mathrm{struct}}$, as in {prf:ref}`assum-barrier-continuity`, and:
$$L_{\max} := \max_{f \in \mathcal{F}_{\mathrm{forbidden}}^*} L_f < \infty.$$

3. **(Structural control by defect risk.)** The parametric family $\mathcal{H}_\Theta$ satisfies {prf:ref}`assum-local-structural-control`: there exist $C_{\mathrm{struct}}, \varepsilon_0 > 0$ such that:
$$\mathcal{R}_S(\Theta) \leq \varepsilon < \varepsilon_0 \implies d_{\mathrm{struct}}(\mathcal{H}_\Theta, \mathcal{H}^*) \leq C_{\mathrm{struct}} \sqrt{\varepsilon}.$$

Then there exists $\varepsilon_1 > 0$ such that for all $\Theta$ with $\mathcal{R}_S(\Theta) \leq \varepsilon_1$:

1. **(Exact stability of forbidden modes.)**
$$\mathrm{Forbidden}(\mathcal{H}_\Theta) = \mathrm{Forbidden}(\mathcal{H}^*) = \mathcal{F}_{\mathrm{forbidden}}^*.$$

2. **(No spurious new exclusions.)** In particular, no failure mode that is allowed by $\mathcal{H}^*$ is spuriously excluded by $\mathcal{H}_\Theta$.

Thus, once the defect risk is small enough, the **discrete pattern** of forbidden failure modes becomes identical, not merely close, to that of the true hypostructure.
:::

:::{prf:proof}
Fix $\varepsilon > 0$ small, and let $\Theta$ be such that $\mathcal{R}_S(\Theta) \leq \varepsilon$. By structural control ({prf:ref}`assum-local-structural-control`):
$$d_{\mathrm{struct}}(\mathcal{H}_\Theta, \mathcal{H}^*) \leq C_{\mathrm{struct}} \sqrt{\varepsilon}.$$

Let $f \in \mathcal{F}_{\mathrm{forbidden}}^*$. By definition of the margin $\gamma^*$:
$$\mathrm{dist}\big( B_f(\mathcal{H}^*), \partial \mathcal{B}_f^{\mathrm{safe}} \big) \geq \gamma^*.$$

By barrier continuity ({prf:ref}`assum-barrier-continuity`):
$$\big| B_f(\mathcal{H}_\Theta) - B_f(\mathcal{H}^*) \big| \leq L_f \, d_{\mathrm{struct}}(\mathcal{H}_\Theta, \mathcal{H}^*) \leq L_f C_{\mathrm{struct}} \sqrt{\varepsilon} \leq L_{\max} C_{\mathrm{struct}} \sqrt{\varepsilon}.$$

Choose $\varepsilon_1 > 0$ small enough that:
$$L_{\max} C_{\mathrm{struct}} \sqrt{\varepsilon_1} \leq \frac{1}{2} \gamma^*.$$

Then for any $\varepsilon \leq \varepsilon_1$:
$$\mathrm{dist}\big( B_f(\mathcal{H}_\Theta), \partial \mathcal{B}_f^{\mathrm{safe}} \big) \geq \mathrm{dist}\big( B_f(\mathcal{H}^*), \partial \mathcal{B}_f^{\mathrm{safe}} \big) - |B_f(\mathcal{H}_\Theta) - B_f(\mathcal{H}^*)| \geq \gamma^* - \frac{1}{2}\gamma^* = \frac{1}{2}\gamma^* > 0.$$

Thus, $B_f(\mathcal{H}_\Theta)$ remains *inside* the safe region $\mathcal{B}_f^{\mathrm{safe}}$, at positive distance from its boundary. Therefore:
$$f \in \mathrm{Forbidden}(\mathcal{H}^*) \implies f \in \mathrm{Forbidden}(\mathcal{H}_\Theta)$$
for all $\Theta$ with $\mathcal{R}_S(\Theta) \leq \varepsilon_1$. In other words:
$$\mathcal{F}_{\mathrm{forbidden}}^* \subseteq \mathrm{Forbidden}(\mathcal{H}_\Theta).$$

To show the reverse inclusion, suppose for contradiction that there exists $f \in \mathcal{F}$ with $f \in \mathrm{Forbidden}(\mathcal{H}_\Theta)$ but $f \notin \mathrm{Forbidden}(\mathcal{H}^*)$. By definition:
$$B_f(\mathcal{H}_\Theta) \in \mathcal{B}_f^{\mathrm{safe}}, \qquad B_f(\mathcal{H}^*) \notin \mathcal{B}_f^{\mathrm{safe}}.$$

Since $\mathcal{B}_f^{\mathrm{safe}}$ is closed, continuity of $B_f$ implies that the set $\{\lambda \in [0,1] : B_f((1-\lambda)\mathcal{H}^* + \lambda \mathcal{H}_\Theta) \in \mathcal{B}_f^{\mathrm{safe}}\}$ has a nonempty boundary in $[0,1]$ where the barrier lies on $\partial \mathcal{B}_f^{\mathrm{safe}}$. But by Lipschitz continuity:
$$\mathrm{dist}\big( B_f(\mathcal{H}^*), \partial \mathcal{B}_f^{\mathrm{safe}} \big) \leq L_f C_{\mathrm{struct}} \sqrt{\varepsilon_1} \leq \tfrac{1}{2}\gamma^*,$$
contradicting the fact that either $f$ is forbidden at $\mathcal{H}^*$ with margin $\gamma_f \geq \gamma^*$, or else $B_f(\mathcal{H}^*)$ lies strictly in the *complement* of $\mathcal{B}_f^{\mathrm{safe}}$ at distance at least some fixed positive amount. For $\varepsilon_1$ sufficiently small, the "spurious exclusion" is impossible.

Hence no new failure modes can enter the forbidden set when $\mathcal{R}_S(\Theta)$ is sufficiently small, and we have:
$$\mathrm{Forbidden}(\mathcal{H}_\Theta) = \mathrm{Forbidden}(\mathcal{H}^*) = \mathcal{F}_{\mathrm{forbidden}}^*.
$$
:::

:::{prf:remark} Margin is essential

The key ingredient is the **margin** $\gamma^* > 0$: if the true hypostructure barely satisfies a barrier inequality, then arbitrarily small perturbations can change whether a mode is forbidden. The preceding metatheorems typically provide such a margin (e.g., strict inequalities in energy/capacity thresholds) except in degenerate "critical" cases.
:::

> **Key Insight:** Learning does not just approximate numbers; it stabilizes the *discrete* "permit denial" judgments. Once the defect risk is small enough, trainable hypostructures recover the **exact discrete permit-denial structure** of the underlying PDE/dynamical system.

### Robust Divergence Control (Mode C.E)

The preceding metatheorem establishes that failure-mode predictions are robust in the abstract. We now prove a concrete instance: **small D-defect implies bounded output**. This is a fully rigorous "robust structural transfer" theorem for the metatheorem "No divergence (Mode C.E) under Axiom D."

**Setup.** Let $\mathcal{H}_\theta = (X, S_t, \Phi_\theta, \mathfrak{D}_\theta, \ldots)$ be a parametric hypostructure with $\mathfrak{D}_\theta(x) \geq 0$ for all $x$. For each trajectory $u: [0, T) \to X$ with $u(t) = S_t x_0$, the **D-defect** is:
$$K_D^{(\theta)}(u|_{[0,T]}) := \int_0^T \max\left(0, \partial_t \Phi_\theta(u(t)) + \mathfrak{D}_\theta(u(t))\right) dt.$$

This is nonnegative and vanishes if and only if the dissipation inequality holds pointwise:
$$\partial_t \Phi_\theta(u(t)) + \mathfrak{D}_\theta(u(t)) \leq 0 \quad \text{a.e. } t.$$

Mode **C.E (divergence)** is defined as $\sup_{t < T^*} \Phi(u(t)) = +\infty$.

:::{prf:metatheorem} Robust Divergence Control
:label: mt-robust-divergence-control

Let $\mathcal{H}_\theta = (X, S_t, \Phi_\theta, \mathfrak{D}_\theta, \ldots)$ be a parametric hypostructure with $\mathfrak{D}_\theta(x) \geq 0$ for all $x$. Fix a trajectory $u: [0, T) \to X$, $u(t) = S_t x_0$, defined on some interval $[0, T)$ where $0 < T \leq T^*(x_0)$.
:::

Assume that for this trajectory the D-defect on $[0, T]$ is bounded by $\varepsilon \geq 0$:
$$K_D^{(\theta)}(u|_{[0,T]}) = \int_0^T \max\left(0, \partial_t \Phi_\theta(u(t)) + \mathfrak{D}_\theta(u(t))\right) dt \leq \varepsilon.$$

Then **for all** $t \in [0, T]$:
$$\Phi_\theta(u(t)) \leq \Phi_\theta(u(0)) + \varepsilon.$$

In particular:

1. If $\varepsilon < \infty$, then the output along $u$ cannot diverge on $[0, T]$; i.e., Mode C.E cannot occur on that interval.

2. If there exists a nondecreasing function $E: [0, T^*) \to [0, \infty)$ such that for every $T' < T^*$,
   $$K_D^{(\theta)}(u|_{[0,T']}) \leq E(T') \quad \text{and} \quad \sup_{T' < T^*} E(T') < \infty,$$
   then $\Phi_\theta(u(t))$ is **uniformly bounded** on $[0, T^*)$, hence Mode C.E is completely excluded for this trajectory.

:::{prf:proof}
Define the "D-residual" function:
$$g(t) := \partial_t \Phi_\theta(u(t)) + \mathfrak{D}_\theta(u(t))$$
where the time derivative exists in the sense used in the D-axiom (for a.e. $t$). By definition of the D-defect:
$$K_D^{(\theta)}(u|_{[0,T]}) = \int_0^T \max(0, g(t)) \, dt \leq \varepsilon. \quad \text{(1)}$$

We exploit two facts: (i) dissipation nonnegativity $\mathfrak{D}_\theta(u(t)) \geq 0$ for all $t$, and (ii) inequality (1).

**Step 1: Pointwise inequality for $\partial_t \Phi_\theta(u(t))$.**

We establish an upper bound on $\partial_t \Phi_\theta(u(t))$ in terms of $g^+(t) := \max(0, g(t))$.

For each $t$, we have two cases:

- If $g(t) \geq 0$, then:
  $$\partial_t \Phi_\theta(u(t)) = g(t) - \mathfrak{D}_\theta(u(t)) \leq g(t) = g^+(t),$$
  since $\mathfrak{D}_\theta \geq 0$.

- If $g(t) < 0$, then $g^+(t) = 0$, while:
  $$\partial_t \Phi_\theta(u(t)) = g(t) - \mathfrak{D}_\theta(u(t)) \leq g(t) < 0 \leq g^+(t).$$


Hence in **all** cases we have the pointwise inequality:
$$\partial_t \Phi_\theta(u(t)) \leq g^+(t) = \max\left(0, \partial_t \Phi_\theta(u(t)) + \mathfrak{D}_\theta(u(t))\right) \quad \text{for a.e. } t \in [0, T]. \quad \text{(2)}$$

This uses only that $\mathfrak{D}_\theta \geq 0$.

**Step 2: Integrate the differential inequality.**

Integrate (2) from $0$ to any $t \in [0, T]$:
$$\Phi_\theta(u(t)) - \Phi_\theta(u(0)) = \int_0^t \partial_s \Phi_\theta(u(s)) \, ds \leq \int_0^t g^+(s) \, ds \leq \int_0^T g^+(s) \, ds = K_D^{(\theta)}(u|_{[0,T]}) \leq \varepsilon.$$

Therefore:
$$\Phi_\theta(u(t)) \leq \Phi_\theta(u(0)) + \varepsilon \quad \forall t \in [0, T]. \quad \text{(3)}$$

This proves the main estimate.

**Step 3: Exclusion of Mode C.E on $[0, T]$.**

By definition, Mode C.E (energy blow-up) requires $\sup_{0 \leq s < T^*} \Phi_\theta(u(s)) = +\infty$.

But (3) shows that on the finite interval $[0, T]$:
$$\sup_{0 \leq s \leq T} \Phi_\theta(u(s)) \leq \Phi_\theta(u(0)) + \varepsilon < \infty.$$

So **no blow-up can occur before time $T$** as long as the D-defect on $[0, T]$ is finite. This proves claim (1).

**Step 4: Uniform control up to $T^*$.**

Now suppose we have a function $E(T')$ with $K_D^{(\theta)}(u|_{[0,T']}) \leq E(T')$ for all $T' < T^*$, and $\sup_{T' < T^*} E(T') =: E_\infty < \infty$.

Then for each $t < T^*$, by applying (3) with $T' = t$ and $\varepsilon = E(t) \leq E_\infty$, we get:
$$\Phi_\theta(u(t)) \leq \Phi_\theta(u(0)) + E(t) \leq \Phi_\theta(u(0)) + E_\infty.$$

Taking supremum over $t < T^*$ yields:
$$\sup_{0 \leq t < T^*} \Phi_\theta(u(t)) \leq \Phi_\theta(u(0)) + E_\infty < \infty.$$

Thus the Mode C.E condition $\sup_{t < T^*} \Phi_\theta(u(t)) = +\infty$ is impossible. This proves claim (2).
:::

:::{prf:remark} Robust structural transfer pattern


- In the **exact** case $K_D^{(\theta)}(u) = 0$, we recover the usual Axiom D conclusion: $\partial_t \Phi_\theta(u(t)) \leq 0 \implies \Phi_\theta(u(t)) \leq \Phi_\theta(u(0))$ for all $t$, so Mode C.E is impossible.
- In the **approximate** case, the theorem gives a sharp quantitative relaxation: *energy can increase by at most the D-defect*.

:::

> **Key Insight (Built-in energy bounds):** A trainable hypostructure with small D-defect automatically provides uniform energy bounds. The deviation from the exact axiom D conclusion is controlled linearly by the defect.

### Robust Latent Mode Suppression

We now prove a robust version of the **Topological Suppression Metatheorem**, showing that the measure of nontrivial latent modes decays exponentially even when the mode separation gap TB1 holds only approximately.

**Recall: Original Topological Suppression Metatheorem.** Assume:
- Axiom TB with mode separation gap $\Delta > 0$,
- an invariant probability measure $\mu$ satisfying a log–Sobolev inequality with constant $\lambda_{\mathrm{LS}} > 0$,
- the cost functional $\mathcal{A}: X \to [0, \infty)$ is Lipschitz with constant $L > 0$.

Then:
$$\mu\{x : \tau(x) \neq 0\} \leq C \exp\left(-c \lambda_{\mathrm{LS}} \frac{\Delta^2}{L^2}\right)$$
with universal constants $C = 1$, $c = 1/8$.

#### Hypotheses for the robust version

Let $(X, \mathcal{B}, \mu)$ be a probability space with:
- $\tau: X \to \mathcal{T}$ the mode assignment (discrete $\mathcal{T}$, $0 \in \mathcal{T}$ the trivial mode),
- $\mathcal{A}: X \to [0, \infty)$ a measurable cost functional,
- $\mathcal{A}_{\min} := \inf_{\tau(x) = 0} \mathcal{A}(x)$.

Assume:

1. **(Log–Sobolev inequality.)** $\mu$ satisfies a log–Sobolev inequality with constant $\lambda_{\mathrm{LS}} > 0$:
   $$\mathrm{Ent}_\mu(f^2) \leq \frac{2}{\lambda_{\mathrm{LS}}} \int |\nabla f|^2 \, d\mu$$
   for all smooth $f$, implying Gaussian concentration via the Herbst argument.

2. **(Lipschitz action.)** $\mathcal{A}$ is $L$-Lipschitz with respect to the ambient metric $d$ on $X$:
   $$|\mathcal{A}(x) - \mathcal{A}(y)| \leq L \, d(x, y) \quad \forall x, y \in X.$$

3. **(Approximate action gap.)** There exist constants $\Delta > 0$, $\varepsilon_{\mathrm{gap}} \geq 0$ and a measurable set $B \subset X$ ("bad set") such that:
   - $B \subset \{x : \tau(x) \neq 0\}$,
   - $\mu(B) \leq \eta$ for some $\eta \in [0, 1]$,
   - for all $x \in X \setminus B$ with $\tau(x) \neq 0$:
     $$\mathcal{A}(x) \geq \mathcal{A}_{\min} + \Delta - \varepsilon_{\mathrm{gap}}. \quad \text{(TG$_\varepsilon$)}$$

   So the exact TB1 gap $\tau(x) \neq 0 \Rightarrow \mathcal{A}(x) \geq \mathcal{A}_{\min} + \Delta$ is allowed to fail on $B$ (small measure) and to be off by $\varepsilon_{\mathrm{gap}}$.

Define the **effective gap**:
$$\Delta_{\mathrm{eff}} := \max\left\{\Delta - \varepsilon_{\mathrm{gap}} - L\sqrt{\frac{\pi}{2\lambda_{\mathrm{LS}}}}, 0\right\}.$$

:::{prf:metatheorem} Robust Latent Mode Suppression
:label: mt-robust-latent-mode-suppression

Under hypotheses (1)–(3) above:
$$\mu\big(\{x : \tau(x) \neq 0\}\big) \leq \eta + \exp\left(-\frac{\lambda_{\mathrm{LS}} \Delta_{\mathrm{eff}}^2}{2L^2}\right).$$
:::

In particular:
- If the **bad set disappears** ($\eta = 0$) and the gap is exact ($\varepsilon_{\mathrm{gap}} = 0$), and if $\Delta \geq 2L\sqrt{\pi/(2\lambda_{\mathrm{LS}})}$, then $\Delta_{\mathrm{eff}} \geq \Delta/2$ and:
  $$\mu\{\tau \neq 0\} \leq \exp\left(-\frac{\lambda_{\mathrm{LS}} \Delta^2}{8L^2}\right),$$
  which recovers the original Topological Suppression bound with $C = 1$, $c = 1/8$ up to the mild "large gap" condition.

- As $\varepsilon_{\mathrm{gap}} \to 0$ and $\eta \to 0$, $\Delta_{\mathrm{eff}} \uparrow \Delta - L\sqrt{\pi/(2\lambda_{\mathrm{LS}})}$, so the suppression bound smoothly tends to the exact one.

:::{prf:proof}
Let $\bar{\mathcal{A}} := \int_X \mathcal{A} \, d\mu$ denote the mean action.

We use two standard consequences of log–Sobolev + Lipschitz:

**Gaussian concentration (Herbst).** For any $r > 0$:
$$\mu\{x \in X : \mathcal{A}(x) - \bar{\mathcal{A}} \geq r\} \leq \exp\left(-\frac{\lambda_{\mathrm{LS}} r^2}{2L^2}\right). \quad \text{(1)}$$

**Bound on the mean above the minimum.** Let $\mathcal{A}_{\inf} := \inf_X \mathcal{A}$ (which is $\leq \mathcal{A}_{\min}$). Then:
$$\bar{\mathcal{A}} - \mathcal{A}_{\inf} = \int_0^\infty \mu\{\mathcal{A} - \mathcal{A}_{\inf} \geq s\} \, ds \leq \int_0^\infty \exp\left(-\frac{\lambda_{\mathrm{LS}} s^2}{2L^2}\right) ds = L\sqrt{\frac{\pi}{2\lambda_{\mathrm{LS}}}}. \quad \text{(2)}$$

Hence, since $\mathcal{A}_{\inf} \leq \mathcal{A}_{\min}$:
$$\bar{\mathcal{A}} - \mathcal{A}_{\min} \leq \bar{\mathcal{A}} - \mathcal{A}_{\inf} \leq L\sqrt{\frac{\pi}{2\lambda_{\mathrm{LS}}}}. \quad \text{(3)}$$

**Step 1: Lower bound on $\mathcal{A}(x) - \bar{\mathcal{A}}$ for nontrivial sectors.**

Fix any $x \in X \setminus B$ with $\tau(x) \neq 0$. By the approximate gap condition (TG$_\varepsilon$):
$$\mathcal{A}(x) \geq \mathcal{A}_{\min} + \Delta - \varepsilon_{\mathrm{gap}}.$$

Subtract $\bar{\mathcal{A}}$ from both sides and use (3):
$$\mathcal{A}(x) - \bar{\mathcal{A}} \geq (\mathcal{A}_{\min} + \Delta - \varepsilon_{\mathrm{gap}}) - \bar{\mathcal{A}} = \Delta - \varepsilon_{\mathrm{gap}} - (\bar{\mathcal{A}} - \mathcal{A}_{\min}) \geq \Delta - \varepsilon_{\mathrm{gap}} - L\sqrt{\frac{\pi}{2\lambda_{\mathrm{LS}}}}.$$

Thus for any such $x$:
$$\mathcal{A}(x) - \bar{\mathcal{A}} \geq \Delta_{\mathrm{eff}}. \quad \text{(4)}$$

Therefore we have the inclusion of events:
$$\{x \in X \setminus B : \tau(x) \neq 0\} \subset \{x \in X : \mathcal{A}(x) - \bar{\mathcal{A}} \geq \Delta_{\mathrm{eff}}\}. \quad \text{(5)}$$

**Step 2: Concentration bound.**

Apply the Gaussian concentration (1) with $r = \Delta_{\mathrm{eff}}$:
$$\mu\{\mathcal{A} - \bar{\mathcal{A}} \geq \Delta_{\mathrm{eff}}\} \leq \exp\left(-\frac{\lambda_{\mathrm{LS}} \Delta_{\mathrm{eff}}^2}{2L^2}\right). \quad \text{(6)}$$

Combining (5) and (6):
$$\mu\{x \in X \setminus B : \tau(x) \neq 0\} \leq \exp\left(-\frac{\lambda_{\mathrm{LS}} \Delta_{\mathrm{eff}}^2}{2L^2}\right). \quad \text{(7)}$$

**Step 3: Add back the bad set $B$.**

We have:
$$\{x : \tau(x) \neq 0\} \subset B \cup \{x \in X \setminus B : \tau(x) \neq 0\}.$$

Hence:
$$\mu\{\tau \neq 0\} \leq \mu(B) + \mu\{x \in X \setminus B : \tau(x) \neq 0\} \leq \eta + \exp\left(-\frac{\lambda_{\mathrm{LS}} \Delta_{\mathrm{eff}}^2}{2L^2}\right),$$
using $\mu(B) \leq \eta$ and (7). This is exactly the claimed bound.
:::

:::{prf:remark} Connection to meta-learning

This theorem connects the TB-defect to the meta-learning story:

- The TB-defect can be interpreted as $\varepsilon_{\mathrm{gap}}$ (how much the action gap inequality fails in value) and $\eta$ (how much of the mass lives in a ``bad'' region where the gap fails completely).
- Small TB-defect in the learned hypostructure $\Rightarrow$ small $\varepsilon_{\mathrm{gap}}$, $\eta$.
- The log-Sobolev constant $\lambda_{\mathrm{LS}}$ and Lipschitz constant $L$ can be estimated from data, giving **explicit bounds** on $\mu\{\tau \neq 0\}$.

:::

> **Key Insight (Built-in sector control):** A trainable hypostructure with small TB-defect automatically provides exponential suppression of nontrivial sectors. The effective gap $\Delta_{\mathrm{eff}}$ smoothly interpolates between exact and approximate axioms.

### Curriculum Stability for Trainable Hypostructures

In practice, one does not typically train a hypostructure learner directly on the most complex possible systems. Instead, it is natural to adopt a **curriculum**: start with simpler systems (e.g. linear ODEs, toy PDEs), then gradually increase complexity (e.g. nonlinear PDEs, multi-scale systems, control-coupled systems), at each stage refining the learned axioms.

We now formalize a **Curriculum Stability** metatheorem: under mild conditions on the path of "true" hypostructure parameters along the curriculum, gradient-based training with warm starts tracks this path and converges to the final, fully complex hypostructure $\Theta^*_{\mathrm{full}}$, without jumping to a spurious ontology.

#### Curriculum of task distributions

Let $\mathcal{S}_1 \subseteq \mathcal{S}_2 \subseteq \cdots \subseteq \mathcal{S}_K$ be an increasing sequence of system distributions, each supported on systems $S$ that admit hypostructure representations in a common parametric family $\{\mathcal{H}_\Theta\}_{\Theta \in \Theta_{\mathrm{adm}}}$.

For each stage $k = 1, \ldots, K$, define the **stage-$k$ average defect risk**:
$$\mathcal{R}_k(\Theta) := \mathbb{E}_{S \sim \mathcal{S}_k}[\mathcal{R}_S(\Theta)],$$
where $\mathcal{R}_S(\Theta)$ is the joint defect risk for system $S$ with parameter $\Theta$ ({prf:ref}`def-joint-defect-risk`).

We think of $\mathcal{S}_1$ as a "simple" distribution (e.g. low-complexity systems), and $\mathcal{S}_K$ as the full, target distribution $\mathcal{S}_{\mathrm{full}}$.

#### True hypostructures along the curriculum

We assume that at each stage $k$, there exists a **true** parameter $\Theta^*_k \in \Theta_{\mathrm{adm}}$ such that:

- $\mathcal{R}_k(\Theta^*_k) = 0$;
- for $\mathcal{S}_k$-almost every system $S$, the hypostructure $\mathcal{H}_{\Theta^*_k}$ satisfies the axioms and defects vanish: $\mathcal{R}_S(\Theta^*_k) = 0$;
- $\Theta^*_k$ is structurally identifiable up to Hypo-isomorphism on $\mathcal{S}_k$.

We write $\Theta^*_{\mathrm{full}} := \Theta^*_K$ for the final-stage parameter.

:::{prf:assumption} Smooth structural path
:label: assum-smooth-structural-path

There exists a $C^1$ curve $\gamma : [0,1] \to \Theta_{\mathrm{adm}}$ such that:
$$\gamma(t_k) = \Theta^*_k, \quad 0 = t_1 < t_2 < \cdots < t_K = 1,$$
and $|\dot{\gamma}(t)|$ is bounded on $[0,1]$. We call $\gamma$ the **structural curriculum path**.
:::

:::{prf:assumption} Stagewise strong convexity
:label: assum-stagewise-strong-convexity

For each $k = 1, \ldots, K$, there exist constants $c_k, C_k, \rho_k > 0$ such that:
$$c_k |\Theta - \Theta^*_k|^2 \leq \mathcal{R}_k(\Theta) - \mathcal{R}_k(\Theta^*_k) \leq C_k |\Theta - \Theta^*_k|^2$$
for all $\Theta$ with $|\Theta - \Theta^*_k| \leq \rho_k$.
:::

We also assume that the gradients $\nabla \mathcal{R}_k$ are Lipschitz in $\Theta$ on these neighborhoods. Let:
$$c_{\min} := \min_k c_k, \quad C_{\max} := \max_k C_k, \quad \rho := \min_k \rho_k.$$

#### Warm-start gradient descent along the curriculum

We consider the following **curriculum training** procedure:

1. Initialize $\Theta^{(1)}_0$ in a small neighborhood of $\Theta^*_1$.

2. For each stage $k = 1, \ldots, K$:
   - Run gradient descent on $\mathcal{R}_k$:
   $$\Theta^{(k)}_{t+1} = \Theta^{(k)}_t - \eta_{k,t} \nabla \mathcal{R}_k(\Theta^{(k)}_t),$$
   with stepsizes $\eta_{k,t}$ satisfying $\sum_t \eta_{k,t} = \infty$, $\sum_t \eta_{k,t}^2 < \infty$, and small enough to stay in the local convexity region.
   - Let $\widehat{\Theta}_k := \lim_{t \to \infty} \Theta^{(k)}_t$ (which exists and equals the unique minimizer in the basin).
   - Use $\widehat{\Theta}_k$ as the initialization for the next stage: $\Theta^{(k+1)}_0 := \widehat{\Theta}_k$.

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Smooth Curriculum Path:** $C^1$ curve $\gamma : [0,1] \to \Theta_{\mathrm{adm}}$ with $|\dot{\gamma}(t)| \leq M$, $\gamma(t_k) = \Theta^*_k$
>     *   [ ] **Stagewise Strong Convexity:** $c_k|\Theta - \Theta^*_k|^2 \leq \mathcal{R}_k(\Theta) \leq C_k|\Theta - \Theta^*_k|^2$ for $|\Theta - \Theta^*_k| \leq \rho_k$
>     *   [ ] **Small Curriculum Steps:** $|\Theta^*_{k+1} - \Theta^*_k| \leq \rho/4$
>     *   [ ] **Accurate Stagewise Minimization:** $|\widehat{\Theta}_k - \Theta^*_k| \leq \rho/4$
> *   **Output (Structural Guarantee):**
>     *   Stay in correct basin: $|\Theta^{(k)}_0 - \Theta^*_k| \leq \rho/2$ for all stages
>     *   Path tracking: $|\widehat{\Theta}_k - \Theta^*_k| \leq \rho/4$ uniformly
>     *   Convergence: $|\widehat{\Theta}_K - \Theta^*_{\mathrm{full}}| \leq \rho/4$
> *   **Failure Condition (Debug):**
>     *   If **Curriculum Steps** too large → **Mode ontology jump** (warm-start leaves basin, wrong minimum found)
>     *   If **Strong Convexity** fails → **Mode basin ambiguity** (multiple local minima, convergence uncertain)

:::{prf:metatheorem} Curriculum Stability
:label: mt-curriculum-stability

Under the above setting, suppose:

1. **(Smooth curriculum path.)** {prf:ref}`assum-smooth-structural-path` holds, and $|\dot{\gamma}(t)| \leq M$ for all $t \in [0,1]$.

2. **(Stagewise strong convexity.)** {prf:ref}`assum-stagewise-strong-convexity` holds uniformly: $c_{\min} > 0$, $C_{\max} < \infty$, $\rho > 0$.

3. **(Small curriculum steps.)** The time steps $t_k$ are chosen such that:
$$|\Theta^*_{k+1} - \Theta^*_k| = |\gamma(t_{k+1}) - \gamma(t_k)| \leq \frac{\rho}{4} \quad \text{for all } k.$$
Equivalently, $(t_{k+1} - t_k) \leq \rho/(4M)$.

4. **(Accurate stagewise minimization.)** At each stage $k$, gradient descent on $\mathcal{R}_k$ is run long enough (with suitably small stepsizes) so that:
$$|\widehat{\Theta}_k - \Theta^*_k| \leq \frac{\rho}{4}.$$

Then for all stages $k = 1, \ldots, K$:

1. **(Stay in the correct basin.)** The initialization for each stage lies in the strong-convexity neighborhood of the true parameter:
$$|\Theta^{(k)}_0 - \Theta^*_k| = |\widehat{\Theta}_{k-1} - \Theta^*_k| \leq \frac{\rho}{2} < \rho.$$
Hence gradient descent at stage $k$ remains in the basin of $\Theta^*_k$ and converges to it.

2. **(Tracking the structural path.)** The sequence of stagewise minimizers $\widehat{\Theta}_k$ satisfies:
$$|\widehat{\Theta}_k - \Theta^*_k| \leq \frac{\rho}{4} \quad \text{for all } k,$$
and hence forms a discrete approximation to the structural path $\gamma$ staying uniformly close to it.

3. **(Convergence to the full hypostructure.)** In particular, the final parameter $\widehat{\Theta}_K$ satisfies:
$$|\widehat{\Theta}_K - \Theta^*_{\mathrm{full}}| \leq \frac{\rho}{4},$$
i.e. curriculum training converges (modulo this small error, which can be made arbitrarily small by refining the steps and optimization accuracy) to the true full hypostructure.

If, moreover, we let the number of stages $K \to \infty$ so that $\max_k(t_{k+1} - t_k) \to 0$ and increase the optimization accuracy at each stage, then in the limit the curriculum procedure tracks $\gamma$ arbitrarily closely and converges to $\Theta^*_{\mathrm{full}}$ in parameter space.
:::

:::{prf:proof}
We argue by induction on the curriculum stages.

**Base case ($k = 1$).** By assumption, we choose $\Theta^{(1)}_0$ close to $\Theta^*_1$, in particular $|\Theta^{(1)}_0 - \Theta^*_1| \leq \rho/2$. By stagewise strong convexity ({prf:ref}`assum-stagewise-strong-convexity`) and standard convergence results for gradient descent on strongly convex, smooth functions, the iterates $\Theta^{(1)}_t$ remain in the ball $B(\Theta^*_1, \rho)$ and converge to the unique minimizer $\Theta^*_1$. For sufficiently long training and small enough step sizes:
$$|\widehat{\Theta}_1 - \Theta^*_1| \leq \rho/4.$$

**Induction step.** Suppose that at stage $k$ we have $|\widehat{\Theta}_k - \Theta^*_k| \leq \rho/4$.

We now consider stage $k+1$. By definition of the curriculum path:
$$|\Theta^*_{k+1} - \Theta^*_k| = |\gamma(t_{k+1}) - \gamma(t_k)| \leq \frac{\rho}{4}.$$

Thus the stage-$(k+1)$ initialization $\Theta^{(k+1)}_0 := \widehat{\Theta}_k$ satisfies:
$$|\Theta^{(k+1)}_0 - \Theta^*_{k+1}| \leq |\Theta^{(k+1)}_0 - \Theta^*_k| + |\Theta^*_k - \Theta^*_{k+1}| \leq \frac{\rho}{4} + \frac{\rho}{4} = \frac{\rho}{2} < \rho.$$

Therefore $\Theta^{(k+1)}_0$ lies in the strong-convexity neighborhood $B(\Theta^*_{k+1}, \rho)$. Gradient descent on $\mathcal{R}_{k+1}$ with sufficiently small step sizes stays inside $B(\Theta^*_{k+1}, \rho)$ and converges to the unique minimizer $\Theta^*_{k+1}$. By running it long enough:
$$|\widehat{\Theta}_{k+1} - \Theta^*_{k+1}| \leq \rho/4,$$
which is the induction hypothesis for the next stage.

By induction, the statements in (1) and (2) hold for all $k = 1, \ldots, K$. The final claim (3) follows immediately for $k = K$, with $\Theta^*_{\mathrm{full}} = \Theta^*_K$.

In the refined-curriculum limit where $K \to \infty$ and $\max_k(t_{k+1} - t_k) \to 0$ while per-stage optimization accuracy is driven to $0$, the discrete sequence $\{\widehat{\Theta}_k\}$ converges uniformly to the continuous path $\gamma(t_k)$ and hence to $\Theta^*_{\mathrm{full}}$ as $t_K \to 1$.
:::

:::{prf:remark} Structural safety of curricula

The theorem shows that **curriculum training is structurally safe** as long as:

- each stage's average defect risk is strongly convex in a neighborhood of its true parameter, and
- successive true parameters $\Theta^*_k$ are not too far apart.

Intuitively, the curriculum path $\gamma$ describes how the ``true axioms'' must deform as one moves from simple to complex systems. The theorem guarantees that a trainable hypostructure, initialized and trained at each stage using the previous stage's solution, will track $\gamma$ rather than jumping to unrelated minima.
:::

:::{prf:remark} Practical implications

Combined with the generalization and robustness metatheorems, this implies:

- training on simple systems first fixes the core axioms,
- advancing the curriculum refines these axioms instead of destabilizing them,
- and the final hypostructure accurately captures the structural content of the full system distribution.

:::

> **Key Insight:** Increasing task complexity along a structurally coherent curriculum preserves the learned axiom structure and refines it, rather than destabilizing it. No spurious ontology (wrong hypostructure branch) is selected along the curriculum.

### Robust Łojasiewicz Convergence

The preceding metatheorems establish robustness for energy bounds (Mode C.E) and sector suppression (TB). We now prove a **robust LS convergence theorem**: small LS-defect implies "almost convergence" to the safe manifold $M$, with explicit quantitative bounds on the measure of "bad" times.

#### Setting and assumptions

Let:
- $H$ be a Hilbert space (or a Riemannian manifold),
- $\Phi: H \to \mathbb{R}$ be a $C^1$ functional bounded below,
- $u: [0, \infty) \to H$ solve the **gradient flow**:
  $$u'(t) = -\nabla \Phi(u(t)).$$

Define the **energy gap**:
$$\Phi_{\min} := \inf_{x \in H} \Phi(x), \qquad f(t) := \Phi(u(t)) - \Phi_{\min} \geq 0.$$

Then along the trajectory:
$$f'(t) = \frac{d}{dt} \Phi(u(t)) = \langle \nabla \Phi(u(t)), u'(t) \rangle = -|\nabla \Phi(u(t))|^2 \leq 0,$$
so $f$ is nonincreasing and bounded below, hence has a limit $f_\infty$.

Let $M \subset H$ be the **safe manifold** (set of equilibria / canonical profiles), as in Axiom LS.

**Assumption (LS-geom): Geometric Łojasiewicz inequality (exact).**

There exists a neighborhood $U \supset M$, constants $\theta \in (0, 1]$, $C_{\mathrm{geo}} > 0$ such that:
$$\Phi(x) - \Phi_{\min} \geq C_{\mathrm{geo}} \, \mathrm{dist}(x, M)^{1/\theta} \quad \text{for all } x \in U. \quad \text{(G-LS)}$$

**Assumption (LS-grad$_\varepsilon$): Gradient Łojasiewicz inequality with $L^2$-defect.**

There exists $c_{\mathrm{LS}} > 0$, $\theta \in (0, 1]$ (the same $\theta$ as above), and a measurable function $e: [0, \infty) \to [0, \infty)$ such that for all $t \geq 0$ with $u(t) \in U$:
$$|\nabla \Phi(u(t))| \geq c_{\mathrm{LS}} \, f(t)^{1-\theta} - e(t). \quad \text{(G-LS-approx)}$$

Define the **LS-defect** of the trajectory by:
$$K_{\mathrm{LS}}(u) := \int_0^\infty e(t)^2 \, dt.$$

We assume this is finite, and write $K_{\mathrm{LS}}(u) \leq \varepsilon^2$ for some $\varepsilon \geq 0$.

**Assumption (Stay in LS region).**

Assume there is $T_0 \geq 0$ such that $u(t) \in U$ for all $t \geq T_0$.

:::{prf:metatheorem} Robust LS Convergence
:label: mt-robust-ls-convergence

Under the assumptions above:

1. **(Energy gap goes to zero.)**
   $$\lim_{t \to \infty} f(t) = 0.$$

2. **(Quantitative integrability of distance to $M$.)**
   For $p := \frac{2(1-\theta)}{\theta}$, there exists a constant $C_1 = C_1(\theta, c_{\mathrm{LS}}, C_{\mathrm{geo}}) > 0$ such that:
   $$\int_{T_0}^\infty \mathrm{dist}(u(t), M)^p \, dt \leq C_1 \left( f(T_0) + K_{\mathrm{LS}}(u) \right).$$

3. **("Almost convergence" to $M$ in measure.)**
   For every radius $R > 0$:
   $$\mathcal{L}^1\big(\{t \geq T_0 : \mathrm{dist}(u(t), M) \geq R\}\big) \leq \frac{C_1}{R^p} \big( f(T_0) + K_{\mathrm{LS}}(u) \big),$$
   where $\mathcal{L}^1$ is Lebesgue measure. As $R \downarrow 0$, the fraction of time spent at distance $\geq R$ from $M$ goes to zero, at a rate controlled by $f(T_0) + K_{\mathrm{LS}}(u)$.

4. **(Convergence along a subsequence; and, with exact LS, full convergence.)**
   There exists a sequence $t_n \to \infty$ such that $\mathrm{dist}(u(t_n), M) \to 0$ as $n \to \infty$.

   If, additionally, the geometric LS inequality (G-LS) holds for all large times and Axiom C provides precompactness of the trajectory, then the full trajectory converges:
   $$u(t) \to x_\infty \in M \quad \text{as } t \to \infty,$$
   which is the usual LS–Simon convergence statement in Axiom LS.
:::

:::{prf:proof}
We shift time so that $T_0 = 0$ for simplicity (replacing $f(0)$ by $f(T_0)$).

**Step 1: A differential inequality for the energy gap.**

Recall $f(t) = \Phi(u(t)) - \Phi_{\min} \geq 0$. Because $u$ is a gradient flow, we have the energy identity:
$$f'(t) = -|\nabla \Phi(u(t))|^2.$$

From the approximate LS inequality (G-LS-approx):
$$|\nabla \Phi(u(t))| \geq c_{\mathrm{LS}} f(t)^{1-\theta} - e(t).$$

Define $g(t) := c_{\mathrm{LS}} f(t)^{1-\theta} - e(t)$. Then $|\nabla \Phi(u(t))| \geq g(t)$, and thus:
$$f'(t) = -|\nabla \Phi(u(t))|^2 \leq -g(t)^2.$$

Expanding:
$$g(t)^2 = c_{\mathrm{LS}}^2 f(t)^{2(1-\theta)} - 2c_{\mathrm{LS}} f(t)^{1-\theta} e(t) + e(t)^2.$$

Hence:
$$f'(t) \leq -c_{\mathrm{LS}}^2 f(t)^{2(1-\theta)} + 2c_{\mathrm{LS}} f(t)^{1-\theta} e(t) - e(t)^2.$$

Drop the negative term $-e(t)^2$ to obtain:
$$f'(t) + c_{\mathrm{LS}}^2 f(t)^{2(1-\theta)} \leq 2c_{\mathrm{LS}} f(t)^{1-\theta} e(t). \quad \text{(1)}$$

**Step 2: Integrate and absorb the error (using $L^2$ defect).**

Use Young's inequality on the RHS of (1): for any $\eta > 0$,
$$2c_{\mathrm{LS}} f^{1-\theta} e \leq \eta c_{\mathrm{LS}}^2 f^{2(1-\theta)} + \frac{1}{\eta} e^2.$$

Choose $\eta = 1/2$:
$$2c_{\mathrm{LS}} f^{1-\theta} e \leq \frac{c_{\mathrm{LS}}^2}{2} f^{2(1-\theta)} + 2e^2.$$

Substitute into (1):
$$f'(t) + c_{\mathrm{LS}}^2 f^{2(1-\theta)} \leq \frac{c_{\mathrm{LS}}^2}{2} f^{2(1-\theta)} + 2e(t)^2,$$
so:
$$f'(t) + \frac{c_{\mathrm{LS}}^2}{2} f^{2(1-\theta)} \leq 2e(t)^2. \quad \text{(2)}$$

Integrate (2) from $0$ to any $T > 0$:
$$\int_0^T f'(t) \, dt + \frac{c_{\mathrm{LS}}^2}{2} \int_0^T f(t)^{2(1-\theta)} dt \leq 2 \int_0^T e(t)^2 \, dt.$$

The left-hand integral of $f'$ is $f(T) - f(0)$:
$$f(T) - f(0) + \frac{c_{\mathrm{LS}}^2}{2} \int_0^T f(t)^{2(1-\theta)} dt \leq 2 \int_0^T e(t)^2 \, dt.$$

Since $f(T) \geq 0$, we can drop it:
$$\frac{c_{\mathrm{LS}}^2}{2} \int_0^T f(t)^{2(1-\theta)} dt \leq f(0) + 2 \int_0^T e(t)^2 \, dt.$$

Let $T \to \infty$. Using $K_{\mathrm{LS}}(u) = \int_0^\infty e(t)^2 \, dt \leq \varepsilon^2$:
$$\frac{c_{\mathrm{LS}}^2}{2} \int_0^\infty f(t)^{2(1-\theta)} dt \leq f(0) + 2K_{\mathrm{LS}}(u).$$

Hence:
$$\int_0^\infty f(t)^{2(1-\theta)} dt \leq \frac{2}{c_{\mathrm{LS}}^2} f(0) + \frac{4}{c_{\mathrm{LS}}^2} K_{\mathrm{LS}}(u). \quad \text{(3)}$$

This proves the quantitative **integrability** of $f^{2(1-\theta)}$.

**Step 3: Show $f(t) \to 0$ as $t \to \infty$.**

We know: $f(t) \geq 0$, $f'(t) = -|\nabla \Phi(u(t))|^2 \leq 0$, so $f$ is nonincreasing and bounded below. Hence $\exists f_\infty \geq 0 : \lim_{t \to \infty} f(t) = f_\infty$.

Assume for contradiction $f_\infty > 0$. Then for all large $t \geq T_1$:
$$f(t) \geq \frac{f_\infty}{2} > 0,$$
so:
$$f(t)^{2(1-\theta)} \geq \left(\frac{f_\infty}{2}\right)^{2(1-\theta)} =: c_0 > 0 \quad \text{for all } t \geq T_1.$$

Then:
$$\int_0^\infty f(t)^{2(1-\theta)} dt \geq \int_{T_1}^\infty f(t)^{2(1-\theta)} dt \geq \int_{T_1}^\infty c_0 \, dt = \infty,$$
contradicting the finiteness from (3).

Thus necessarily $f_\infty = 0$, i.e., $\lim_{t \to \infty} f(t) = 0$. This proves conclusion (1).

**Step 4: Integrability of distance to $M$.**

From the geometric LS inequality (G-LS):
$$f(t) = \Phi(u(t)) - \Phi_{\min} \geq C_{\mathrm{geo}} \, \mathrm{dist}(u(t), M)^{1/\theta} \quad \text{for all } t \text{ with } u(t) \in U.$$

Rearrange:
$$\mathrm{dist}(u(t), M) \leq C_{\mathrm{geo}}^{-\theta} f(t)^\theta.$$

Raise both sides to the power $p := \frac{2(1-\theta)}{\theta} > 0$:
$$\mathrm{dist}(u(t), M)^p \leq C_{\mathrm{geo}}^{-p\theta} f(t)^{p\theta}.$$

But $p\theta = \frac{2(1-\theta)}{\theta} \cdot \theta = 2(1-\theta)$, so:
$$\mathrm{dist}(u(t), M)^p \leq C_{\mathrm{geo}}^{-2(1-\theta)} f(t)^{2(1-\theta)}.$$

Integrate from $0$ to $\infty$ and use (3):
$$\int_0^\infty \mathrm{dist}(u(t), M)^p \, dt \leq C_{\mathrm{geo}}^{-2(1-\theta)} \int_0^\infty f(t)^{2(1-\theta)} dt \leq C_{\mathrm{geo}}^{-2(1-\theta)} \left(\frac{2}{c_{\mathrm{LS}}^2} f(0) + \frac{4}{c_{\mathrm{LS}}^2} K_{\mathrm{LS}}(u)\right).$$

Setting $C_1 := C_{\mathrm{geo}}^{-2(1-\theta)} \cdot \frac{4}{c_{\mathrm{LS}}^2}$:
$$\int_0^\infty \mathrm{dist}(u(t), M)^p \, dt \leq C_1 \big( f(0) + K_{\mathrm{LS}}(u) \big),$$
which is conclusion (2).

**Step 5: Measure of "bad" times (far from $M$).**

Fix any $R > 0$. Let $S_R := \{t \geq 0 : \mathrm{dist}(u(t), M) \geq R\}$.

Then on $S_R$: $\mathrm{dist}(u(t), M)^p \geq R^p$.

Thus:
$$\int_0^\infty \mathrm{dist}(u(t), M)^p \, dt \geq \int_{S_R} \mathrm{dist}(u(t), M)^p \, dt \geq R^p \, \mathcal{L}^1(S_R),$$
where $\mathcal{L}^1$ denotes Lebesgue measure.

So:
$$\mathcal{L}^1(S_R) \leq \frac{1}{R^p} \int_0^\infty \mathrm{dist}(u(t), M)^p \, dt \leq \frac{C_1}{R^p} \big( f(0) + K_{\mathrm{LS}}(u) \big).$$

This is precisely conclusion (3). As $R \downarrow 0$, the measure of times with distance $\geq R$ is bounded by a factor that scales like $R^{-p}$.

**Step 6: Subsequence convergence to $M$.**

From (2), we know $\int_0^\infty \mathrm{dist}(u(t), M)^p \, dt < \infty$. A standard measure-theory fact: if a nonnegative function $h$ has finite integral on $[0, \infty)$, then there exists a sequence $t_n \to \infty$ with $h(t_n) \to 0$.

Apply this to $h(t) := \mathrm{dist}(u(t), M)^p$:
$$\exists t_n \to \infty \quad \text{such that} \quad \mathrm{dist}(u(t_n), M)^p \to 0 \implies \mathrm{dist}(u(t_n), M) \to 0.$$

That proves conclusion (4) in its subsequence form.

If we now bring in Axiom C + Reg (bounded trajectories have limit points) and the precise LS machinery (C·D–LS+Reg $\Rightarrow$ convergence to $M$ for bounded trajectories), then one can upgrade "subsequence convergence to $M$" to **full convergence** $u(t) \to x_\infty \in M$, whenever the exact LS conditions hold globally for large time.
:::

:::{prf:remark} Connection to learning

In the meta-learning story: A meta-learner that finds a hypostructure with small LS-defect $K_{\mathrm{LS}}$ is enough to conclude that ``most'' of the long-time dynamics (in time-measure sense) lies arbitrarily close to the safe manifold $M$, with explicit quantitative bounds depending on the learned LS constants and the residual defect.
:::

> **Key Insight (Built-in convergence guarantees):** A trainable hypostructure with small LS-defect automatically provides:
> - Energy gap $f(t) \to 0$,
> - Distance to $M$ is $L^p$-integrable,
> - The set of times where $u$ is farther than $R$ from $M$ has measure $\lesssim (f(T_0) + K_{\mathrm{LS}})/R^p$,
> - Subsequence convergence to $M$.
>
> The exact LS-Simon convergence is the limiting case when the defect vanishes.

### Hypostructure-from-Raw-Data: Learning Structure from Observations

The preceding robust metatheorems establish that approximate axiom satisfaction (small defects) still yields meaningful structural conclusions. We now address a more fundamental question: **can we learn hypostructures directly from raw observational data, without prior knowledge of the state space or dynamics?**

This section presents a rigorous meta-theorem showing that training on **prediction + defect-risk** from raw observations recovers the latent hypostructure (up to isomorphism) in the population limit, provided such a hypostructure exists.

#### Setup: Systems, Data, and Models

##### Task/System Space

Let $(\mathcal{S}, \mathcal{F}, \nu)$ be a probability space of **systems** (or "tasks").

For each $s \in \mathcal{S}$, we have an associated **observation process**:
$$Y^{(s)} = (Y^{(s)}_t)_{t \in \mathbb{Z}}$$
taking values in a Polish observation space $\mathcal{Y}$.

Let $\mathbb{P}_s$ be the law of the process $Y^{(s)}$ on $\mathcal{Y}^{\mathbb{Z}}$.

We do **not** assume we know a state space or dynamics for $s$—only its observation law $\mathbb{P}_s$.

##### True Latent Hypostructured Systems (Realizability Layer)

We assume there exists some "true" latent representation, but it is hidden.

For each $s \in \mathcal{S}$, there exist:

- A separable metric latent space $X_s$,
- A measurable flow or semiflow $(S_t^{(s)})_{t \in \mathbb{Z}}$ on $X_s$,
- A **true hypostructure** $\mathcal{H}^{(s)*}$ on $X_s$ with structural data:
  $$\mathcal{H}^{(s)*} = (X_s, S_t^{(s)}, \Phi^{(s)*}, \mathfrak{D}^{(s)*}, c^{(s)*}, \tau^{(s)*}, \mathcal{A}^{(s)*}, \ldots),$$
  satisfying all axioms exactly (C, D, SC, Cap, TB, LS, GC, ...),
- An **observation map** $O_s: X_s \to \mathcal{Y}$ such that if $X^{(s)}_t$ follows $(S_t^{(s)})$ and $Y^{(s)}_t := O_s(X^{(s)}_t)$, then the law of $Y^{(s)}$ is exactly $\mathbb{P}_s$.

We call $\mathcal{H}^{(s)*}$ a **true latent hypostructure** for system $s$.

This is the *latent realizability* assumption: the world *has* a hypostructural description, but we do not know $X_s$, $S_t^{(s)}$, $O_s$, or the structure.

##### Models: Encoders and Parametric Hypostructures

We fix:

- A **latent model space** $Z = \mathbb{R}^d$ with its usual Euclidean metric.
- A **window size** $k \in \mathbb{N}$ for temporal encoding.
- A parameter space $\Psi \subset \mathbb{R}^p$ for **encoders** and $\Phi \subset \mathbb{R}^q$ for **hypostructure generators**; both are assumed to be compact or at least closed and such that level sets of the risks we define are relatively compact.

**Encoders.** For each $\psi \in \Psi$, we have a measurable **encoder**:
$$E_\psi: \mathcal{Y}^k \to Z.$$

Given a trajectory $y = (y_t)_{t \in \mathbb{Z}}$ and a fixed convention (say, left-aligned windows), define the **induced latent trajectory**:
$$z^{(\psi)}_t = E_\psi(y_{t-k+1}, \ldots, y_t) \in Z.$$

We are not assuming this comes from any actual state space—this is just what the encoder does.

**Hypostructure Generator (Hypernetwork).** We fix a **task representation map** $\iota: \mathcal{S} \to \mathbb{R}^m$ (which can be as simple as an index embedding, or empirical statistics).

For each $\varphi \in \Phi$, we have a **hypernetwork**:
$$H_\varphi: \mathbb{R}^m \to \Theta$$
with parameter space $\Theta \subset \mathbb{R}^r$, continuous in $\varphi$. For each task $s \in \mathcal{S}$:
$$\theta_{s,\varphi} := H_\varphi(\iota(s)) \in \Theta$$
is the hypostructure parameter for system $s$.

For each $\theta \in \Theta$, we have a **parametric hypostructure on $Z$**:
$$\mathcal{H}_\theta = (Z, F_\theta, \Phi_\theta, \mathfrak{D}_\theta, c_\theta, \tau_\theta, \mathcal{A}_\theta, \ldots)$$
where:
- $F_\theta: Z \to Z$ is the latent dynamics model,
- $\Phi_\theta, \mathfrak{D}_\theta, c_\theta, \mathcal{A}_\theta: Z \to \mathbb{R}$ are measurable structural maps,
- $\tau_\theta: Z \to \mathcal{T}$ is a measurable sector map.

Think of all of these as implemented by neural networks (universally approximating function classes), but we only need measurability and continuity in $\theta$.

Given $(\psi, \varphi)$ and a system $s$, the **effective hypostructure on latent trajectories** is:
$$\mathcal{H}^{(s)}_{\psi,\varphi} := (Z, F_{\theta_{s,\varphi}}, \Phi_{\theta_{s,\varphi}}, \mathfrak{D}_{\theta_{s,\varphi}}, c_{\theta_{s,\varphi}}, \tau_{\theta_{s,\varphi}}, \mathcal{A}_{\theta_{s,\varphi}}, \ldots)$$
restricted to the support of latent trajectories $z^{(\psi)}_t$ obtained from $Y^{(s)} \sim \mathbb{P}_s$.

#### Losses: Prediction + Defect-Risk

##### Prediction Loss

Fix a nonnegative measurable loss $\ell: Z \times Z \to [0, \infty)$ (e.g., squared error).

For each $(\psi, \varphi)$, define the **population prediction loss**:
$$\mathcal{L}_{\mathrm{pred}}(\psi, \varphi) := \int_{\mathcal{S}} \mathbb{E}_{Y \sim \mathbb{P}_s}\left[\ell\big(F_{\theta_{s,\varphi}}(z^{(\psi)}_t), z^{(\psi)}_{t+1}\big)\right] \nu(ds),$$
where $t$ is any fixed time index (stationarity or shift-invariance of $\mathbb{P}_s$ makes the choice irrelevant; otherwise we can average over a finite window).

This is the usual latent one-step prediction risk.

##### Defect-Risk

For each soft axiom $A$ in the list (C, D, SC, Cap, TB, LS, GC, ...), and for each $\theta$, we have an **axiom defect functional**:
$$K_A(\mathcal{H}_\theta; z_\bullet)$$
for a latent trajectory $z_\bullet = (z_t)_{t \in \mathbb{Z}}$, such that:
- $K_A(\mathcal{H}_\theta; z_\bullet) \geq 0$,
- $K_A(\mathcal{H}_\theta; z_\bullet) = 0$ if and only if the trajectory satisfies axiom $A$ exactly.

Fix nonnegative weights $\lambda_A \geq 0$ and define, for each $(\psi, \varphi)$:
$$\mathcal{R}_{\mathrm{axioms}}(\psi, \varphi) := \sum_A \lambda_A \int_{\mathcal{S}} \mathbb{E}_{Y \sim \mathbb{P}_s}\left[K_A\big(\mathcal{H}_{\theta_{s,\varphi}}; z^{(\psi)}_\bullet\big)\right] \nu(ds).$$

This is the **population defect-risk**: average defect across tasks and trajectories.

##### Total Risk

Fix $\lambda > 0$ and define:
$$\mathcal{L}_{\mathrm{total}}(\psi, \varphi) := \mathcal{L}_{\mathrm{pred}}(\psi, \varphi) + \lambda \cdot \mathcal{R}_{\mathrm{axioms}}(\psi, \varphi).$$

This is the functional we will minimize by (stochastic) gradient descent.

#### Assumptions (Inductive Bias + Regularity)

We now state explicit assumptions that make this a well-posed meta-learning problem.

**Assumption (H1): Regularity/Measurability.**
- The maps $(\psi, \varphi) \mapsto E_\psi$, $(\varphi, s) \mapsto \theta_{s,\varphi}$, $(\theta, z) \mapsto F_\theta(z)$, $(\theta, z) \mapsto$ structural maps are Borel and continuous in parameters.
- For each $(\psi, \varphi)$, $\mathcal{L}_{\mathrm{pred}}(\psi, \varphi)$ and $\mathcal{R}_{\mathrm{axioms}}(\psi, \varphi)$ are finite and continuous in $(\psi, \varphi)$.

This is true if everything is implemented by continuous neural networks on compact domains with bounded outputs and $\ell$, $K_A$ are continuous in their arguments.

**Assumption (H2): Parametric Realizability of True Hypostructures.**

There exists a parameter pair $(\psi^*, \varphi^*) \in \Psi \times \Phi$ such that:

For $\nu$-almost every system $s \in \mathcal{S}$, there is an **isomorphism of hypostructures** $T_s: X_s \to Z$ (with inverse on the support of the dynamics) satisfying:

1. **Encoder consistency:** For $\mathbb{P}_s$-almost every trajectory $Y^{(s)}$, if $X^{(s)}_t$ is the latent true trajectory and $Y^{(s)}_t = O_s(X^{(s)}_t)$, then the encoded trajectory:
   $$z_t^{(\psi^*)} = E_{\psi^*}(Y^{(s)}_{t-k+1}, \ldots, Y^{(s)}_t)$$
   coincides with $T_s(X^{(s)}_t)$.

2. **Dynamics consistency:**
   $$F_{\theta_{s,\varphi^*}}(T_s(x)) = T_s(S_1^{(s)} x) \quad \text{for all } x \text{ in the support of the true dynamics}.$$

3. **Hypostructure consistency:** The pullback of the parametric hypostructure on $Z$ via $T_s$ equals the true hypostructure on $X_s$:
   $$T_s^*(\mathcal{H}_{\theta_{s,\varphi^*}}) = \mathcal{H}^{(s)*}.$$
   In particular, for every trajectory induced by $Y^{(s)}$, all axioms hold exactly, so every defect vanishes:
   $$K_A\big(\mathcal{H}_{\theta_{s,\varphi^*}}; z^{(\psi^*)}_\bullet\big) = 0 \quad \text{for all } A \text{ and for } \mathbb{P}_s\text{-a.e. } Y^{(s)}.$$

This is the formal statement: **there exists an encoder + hypernetwork whose induced latent hypostructures realize the true ones almost surely.**

**Assumption (H3): Identifiability Up to Hypostructure Isomorphism.**

If for some $(\psi, \varphi)$ we have:
- $\mathcal{L}_{\mathrm{pred}}(\psi, \varphi) = 0$,
- $\mathcal{R}_{\mathrm{axioms}}(\psi, \varphi) = 0$,

then for $\nu$-almost every $s \in \mathcal{S}$, there exists a hypostructure isomorphism $\tilde{T}_s: X_s \to Z$ such that $\tilde{T}_s^*(\mathcal{H}_{\theta_{s,\varphi}}) = \mathcal{H}^{(s)*}$, and the encoded trajectories coincide with $\tilde{T}_s(X^{(s)}_t)$ as in (H2.1).

In words: **zero total risk implies we have recovered the true latent hypostructure up to isomorphism.**

(This is exactly the "Meta-Identifiability" assumption, extended to include the encoder. It encodes the idea that there are no degenerate parameterizations that have perfect prediction and axioms but give a genuinely different structure.)

**Assumption (H4): Optimization (Gradient Descent/SGD Scheme).**

Let $(\psi_n, \varphi_n)_{n \geq 0}$ be an iterative sequence produced by some optimization algorithm (deterministic GD, stochastic GD, etc.) such that:

1. The learning rule is of the form:
   $$(\psi_{n+1}, \varphi_{n+1}) = (\psi_n, \varphi_n) - \eta_n \hat{\nabla} \mathcal{L}_{\mathrm{total}}(\psi_n, \varphi_n),$$
   where $\hat{\nabla} \mathcal{L}_{\mathrm{total}}$ is an unbiased stochastic gradient estimator with bounded variance (conditional on $(\psi_n, \varphi_n)$), constructed from i.i.d. samples of $s \sim \nu$ and trajectories $Y^{(s)} \sim \mathbb{P}_s$.

2. The step sizes $\eta_n$ satisfy the Robbins–Monro conditions:
   $$\sum_{n=0}^\infty \eta_n = \infty, \quad \sum_{n=0}^\infty \eta_n^2 < \infty.$$

3. $\mathcal{L}_{\mathrm{total}}$ is bounded below (by 0) and has **Lipschitz gradient** on $\Psi \times \Phi$, and its sublevel sets $\{(\psi, \varphi) : \mathcal{L}_{\mathrm{total}}(\psi, \varphi) \leq \alpha\}$ are relatively compact.

This is a standard nonconvex SGD setting. Classical results (e.g., Kushner–Yin, Benaïm) then say that:
- $\mathcal{L}_{\mathrm{total}}(\psi_n, \varphi_n)$ converges almost surely,
- The set of limit points of $(\psi_n, \varphi_n)$ is a compact connected set of **stationary points** of $\mathcal{L}_{\mathrm{total}}$.

We will use this as a black box. (Alternatively, assume exact GD on the population risk for a simpler statement.)

#### Main Meta-Theorem

:::{prf:metatheorem} Hypostructure-from-Raw-Data
:label: mt-hypostructure-from-raw-data

Assume (H1)–(H4). Then:

1. **(Zero infimum and nonempty minimizer set.)** The total population risk satisfies:
   $$\inf_{(\psi,\varphi) \in \Psi \times \Phi} \mathcal{L}_{\mathrm{total}}(\psi, \varphi) = 0$$
   and the set of global minimizers:
   $$\mathcal{M} := \{(\psi, \varphi) : \mathcal{L}_{\mathrm{total}}(\psi, \varphi) = 0\}$$
   is nonempty and compact.

2. **(Structural recovery at any global minimizer.)** For any $(\hat{\psi}, \hat{\varphi}) \in \mathcal{M}$, for $\nu$-almost every system $s \in \mathcal{S}$, there exists a hypostructure isomorphism $\tilde{T}_s: X_s \to Z$ such that:
   - The encoded latent trajectory matches the pushed-forward true trajectory:
     $$z_t^{(\hat{\psi})} = \tilde{T}_s(X^{(s)}_t) \quad \text{for } \mathbb{P}_s\text{-a.e. } Y^{(s)};$$
   - The induced hypostructure equals the true one:
     $$\tilde{T}_s^*(\mathcal{H}_{\theta_{s,\hat{\varphi}}}) = \mathcal{H}^{(s)*};$$
   - In particular, all global metatheorems (those using only axioms C, D, SC, Cap, TB, LS, GC, ...) hold **exactly** for the latent representation produced by $(\hat{\psi}, \hat{\varphi})$ and therefore for the original system $s$.

3. **(Convergence of SGD to structural recovery.)** Let $(\psi_n, \varphi_n)_{n \geq 0}$ be any SGD sequence satisfying (H4). Then with probability 1:
   - The limit set of $(\psi_n, \varphi_n)$ is a connected compact subset of $\mathcal{M}$;
   - In particular:
     $$\lim_{n \to \infty} \mathcal{L}_{\mathrm{total}}(\psi_n, \varphi_n) = 0.$$
     Thus, for any sequence of iterates converging to some $(\bar{\psi}, \bar{\varphi})$, we have $(\bar{\psi}, \bar{\varphi}) \in \mathcal{M}$, and the structural recovery property of (2) applies.

So: under the assumption that **there exists some encoder + hypernetwork that can express the true hypostructure**, generic deep-learning-style training on **prediction + defect-risk** from **raw observations** is guaranteed (in the population limit) to recover that hypostructure up to isomorphism.
:::

:::{prf:proof}
**Step 1: Infimum is zero and $\mathcal{M} \neq \emptyset$.**

From (H2), there exists $(\psi^*, \varphi^*)$ such that:

- For $\nu$-a.e. $s$, the induced latent hypostructure is isomorphic to the true one,
- For $\mathbb{P}_s$-a.e. trajectory, dynamics and axioms match exactly.


Hence, for $\nu$-a.e. $s$:

- Prediction error is zero:
  $$\mathbb{E}_{Y \sim \mathbb{P}_s}\left[\ell(F_{\theta_{s,\varphi^*}}(z_t^{(\psi^*)}), z_{t+1}^{(\psi^*)})\right] = 0,$$
  so $\mathcal{L}_{\mathrm{pred}}(\psi^*, \varphi^*) = 0$;
- Each axiom-defect is zero:
  $$\mathbb{E}_{Y \sim \mathbb{P}_s}\left[K_A\big(\mathcal{H}_{\theta_{s,\varphi^*}}; z^{(\psi^*)}_\bullet\big)\right] = 0,$$
  so $\mathcal{R}_{\mathrm{axioms}}(\psi^*, \varphi^*) = 0$.


Therefore:
$$\mathcal{L}_{\mathrm{total}}(\psi^*, \varphi^*) = 0.$$

Since $\mathcal{L}_{\mathrm{total}} \geq 0$ everywhere (by definition), we conclude:
$$\inf_{(\psi,\varphi)} \mathcal{L}_{\mathrm{total}}(\psi, \varphi) = 0$$
and $\mathcal{M} \neq \emptyset$.

Lower semicontinuity (from (H1)) and compactness of level sets imply $\mathcal{M}$ is compact.

**Step 2: Structural recovery at minimizers.**

Let $(\hat{\psi}, \hat{\varphi}) \in \mathcal{M}$. Then $\mathcal{L}_{\mathrm{total}}(\hat{\psi}, \hat{\varphi}) = 0$.

By definition of $\mathcal{L}_{\mathrm{total}}$, this implies separately:

- $\mathcal{L}_{\mathrm{pred}}(\hat{\psi}, \hat{\varphi}) = 0$,
- $\mathcal{R}_{\mathrm{axioms}}(\hat{\psi}, \hat{\varphi}) = 0$.


Because both terms are integrals of nonnegative random variables over $(\mathcal{S}, \nu)$ and trajectories, Fubini's theorem implies:

- For $\nu$-almost every $s$:
  $$\mathbb{E}_{Y \sim \mathbb{P}_s}\left[\ell(F_{\theta_{s,\hat{\varphi}}}(z_t^{(\hat{\psi})}), z_{t+1}^{(\hat{\psi})})\right] = 0,$$
  so the prediction error is zero $\mathbb{P}_s$-a.s.;
- For each axiom $A$ and $\nu$-a.e. $s$:
  $$\mathbb{E}_{Y \sim \mathbb{P}_s}\left[K_A\big(\mathcal{H}_{\theta_{s,\hat{\varphi}}}; z^{(\hat{\psi})}_\bullet\big)\right] = 0,$$
  so axiom-defect $K_A$ is zero $\mathbb{P}_s$-a.s.


Thus, for $\nu$-a.e. $s$, for $\mathbb{P}_s$-almost every trajectory, we have:

- Perfect prediction in latent space,
- Exact satisfaction of all axioms—i.e., those latent trajectories are **exact hypostructural trajectories** for $\mathcal{H}_{\theta_{s,\hat{\varphi}}}$.


By (H3) (Identifiability), it follows that for $\nu$-almost every $s$ there exists a hypostructure isomorphism $\tilde{T}_s: X_s \to Z$ such that:

- The encoded latent trajectory equals $\tilde{T}_s(X^{(s)}_t)$,
- $\tilde{T}_s^*(\mathcal{H}_{\theta_{s,\hat{\varphi}}}) = \mathcal{H}^{(s)*}$.


Therefore, any global minimizer recovers the true latent hypostructure (up to iso) for almost every system $s$. Since all global metatheorems are stated purely in terms of the axioms and hypostructure, they therefore hold for the learned latent representation.

This proves statement (2).

**Step 3: Convergence of SGD to minimizers.**

Under (H4), we are in a standard stochastic approximation setting:

- $\mathcal{L}_{\mathrm{total}}$ is bounded below and has Lipschitz gradient,
- $\hat{\nabla} \mathcal{L}_{\mathrm{total}}$ is an unbiased estimator with bounded variance,
- Step sizes satisfy Robbins–Monro conditions.


By classical results in stochastic approximation (e.g., Kushner–Yin, Benaïm), we have:

- $\mathcal{L}_{\mathrm{total}}(\psi_n, \varphi_n)$ converges almost surely to some random variable $L_\infty$,
- Every limit point of $(\psi_n, \varphi_n)$ is almost surely a **stationary point** of $\mathcal{L}_{\mathrm{total}}$,
- The limit set of $(\psi_n, \varphi_n)$ is almost surely a compact connected set of stationary points.


Now observe that:

- For any stationary point $(\bar{\psi}, \bar{\varphi})$, by continuity and nonnegativity we must have $\mathcal{L}_{\mathrm{total}}(\bar{\psi}, \bar{\varphi}) \geq 0$.
- If the algorithm ever gets arbitrarily close to a global minimizer, the descent property and compactness of sublevel sets prevent it from escaping up to positive risk.


We can sharpen this by assuming (which is standard and often included in (H4)) that $\mathcal{L}_{\mathrm{total}}$ satisfies the **Kurdyka–Łojasiewicz (KŁ) property** (true for real-analytic or semi-algebraic losses, which neural nets typically satisfy). Then standard KŁ + GD theory implies that every limit point of a gradient-based descent sequence must be a stationary point, and if the global minimizers form a connected component, all limit points lie in that component.

Combining:

- The limit set of $(\psi_n, \varphi_n)$ is contained in the set of stationary points.
- Among stationary points, those with minimal value form the set $\mathcal{M}$ of global minimizers (since global minimum is 0).
- Under mild KŁ-type assumptions, any connected component of stationary points with minimal value is exactly $\mathcal{M}$.


Hence, almost surely, the limit set of $(\psi_n, \varphi_n)$ is a compact connected subset of $\mathcal{M}$, and:
$$\lim_{n \to \infty} \mathcal{L}_{\mathrm{total}}(\psi_n, \varphi_n) = 0.$$

Any convergent subsequence has its limit in $\mathcal{M}$, and thus by Step 2 recovers the true hypostructures up to isomorphism for $\nu$-almost every system.

This proves statement (3).
:::

:::{prf:remark} Significance for structural learning

This meta-theorem establishes that:

- The user only provides raw trajectories and a big NN architecture,
- All inductive bias is: ``there exists some encoder + hypostructure in this NN class that matches reality'' (exactly the same kind of bias deep learning already assumes),
- Under that assumption, minimizing **prediction + defect-risk** recovers the latent hypostructure from pixels, in the population limit, with a standard SGD convergence argument.

:::

> **Key Insight (Foundation for learnable physics):** This theorem provides the theoretical foundation for treating hypostructures as learnable objects. Once learned, the axioms become predictive: the learned hypostructure inherits all metatheorems, allowing structural conclusions about the underlying physical system from pure observational data.

### Equivariance of Trainable Hypostructures Under Symmetry Groups

Many system families carry natural symmetry groups: space-time translations, rotations, Galilean boosts, scaling symmetries, gauge groups, etc. A central expectation for a "structural" learner is that it should not break such symmetries arbitrarily: if the distribution of systems and the true hypostructure are symmetric under a group $G$, then the **learned hypostructure** should also be $G$-equivariant.

In this section we formalize this as an **equivariance metatheorem**: under natural compatibility assumptions between $G$, the system distribution, the hypostructure family, and the defect-risk, every risk minimizer is $G$-equivariant (up to gauge), and gradient flow preserves equivariance.

#### Symmetry group acting on systems and hypostructures

Let $G$ be a (locally compact) group acting on the state space $X$ and on the class of systems $S$. For each $g \in G$, we denote by $g \cdot S$ the transformed system obtained by pushing forward the dynamics under $g$ (e.g. conjugating the semiflow by $g$).

:::{prf:assumption} Group-covariant system distribution
:label: assum-group-covariant-distribution

Let $\mathcal{S}$ be a distribution on systems $S$. We assume $\mathcal{S}$ is $G$-invariant:
$$S \sim \mathcal{S} \implies g \cdot S \sim \mathcal{S} \quad \forall g \in G.$$

Equivalently, for any measurable set of systems $\mathcal{A}$, $\mathcal{S}(\mathcal{A}) = \mathcal{S}(g \cdot \mathcal{A})$.
:::

Let $\Theta_{\mathrm{adm}}$ be the parameter space of a hypostructure family $\{\mathcal{H}_\Theta\}_{\Theta \in \Theta_{\mathrm{adm}}}$, with:
$$\mathcal{H}_\Theta(S) = (X_S, S_t, \Phi_{\Theta,S}, \mathfrak{D}_{\Theta,S}, G_{\Theta,S})$$
the hypostructure associated to system $S$ and parameter $\Theta$.

:::{prf:assumption} Equivariant parametrization
:label: assum-equivariant-parametrization

There is a group action of $G$ on $\Theta_{\mathrm{adm}}$, denoted $(g, \Theta) \mapsto g \cdot \Theta$, such that for all $g \in G$, systems $S$, and parameters $\Theta$:
$$g \cdot \mathcal{H}_\Theta(S) \simeq \mathcal{H}_{g \cdot \Theta}(g \cdot S)$$
in the Hypo category, i.e. the hypostructure induced by first transforming $\Theta$ and $S$ by $G$ coincides (up to Hypo-isomorphism) with the pushforward of $\mathcal{H}_\Theta(S)$ by $g$.
:::

Intuitively, this means the family $\{\mathcal{H}_\Theta\}$ is expressive enough and parametrized in such a way that group transformations commute with hypostructure construction, up to the usual notion of "same" hypostructure (gauge).

#### Symmetry of the defect-risk

For each system $S$ and parameter $\Theta$, we have the joint defect-risk:
$$\mathcal{R}_S(\Theta) := \sum_{A \in \mathcal{A}} w_A \mathcal{R}_{A,S}(\Theta), \qquad \mathcal{R}_{A,S}(\Theta) := \int_{\mathcal{U}_S} K_{A,S}^{(\Theta)}(u) \, d\mu_S(u),$$
constructed from the defect functionals $K_{A,S}^{(\Theta)}$. The **average risk** over $\mathcal{S}$ is:
$$\mathcal{R}_{\mathcal{S}}(\Theta) := \mathbb{E}_{S \sim \mathcal{S}}[\mathcal{R}_S(\Theta)].$$

:::{prf:assumption} Group-invariance of defects and trajectories
:label: assum-group-invariance-defects

For each $g \in G$, the following hold:

1. The transformation $u \mapsto g \cdot u$ maps trajectories of $S$ to trajectories of $g \cdot S$, and preserves the trajectory measure (or transforms it in a controlled way that cancels in expectation):
$$\mu_{g \cdot S} = (g \cdot)_\# \mu_S.$$

2. The defect functionals are compatible with the group action:
$$K_{A, g \cdot S}^{(g \cdot \Theta)}(g \cdot u) = K_{A,S}^{(\Theta)}(u) \quad \text{for all } A \in \mathcal{A}, u \in \mathcal{U}_S.$$

In particular, $\mathcal{R}_{g \cdot S}(g \cdot \Theta) = \mathcal{R}_S(\Theta)$.
:::

:::{prf:lemma} Risk equivariance
:label: lem-risk-equivariance

For all $g \in G$ and $\Theta \in \Theta_{\mathrm{adm}}$:
$$\mathcal{R}_{\mathcal{S}}(g \cdot \Theta) = \mathcal{R}_{\mathcal{S}}(\Theta).$$
:::

:::{prf:proof}
Using $\mathcal{S}$-invariance and defect compatibility:
$$\mathcal{R}_{\mathcal{S}}(g \cdot \Theta) = \mathbb{E}_{S \sim \mathcal{S}}[\mathcal{R}_S(g \cdot \Theta)] = \mathbb{E}_{S \sim \mathcal{S}}[\mathcal{R}_{g^{-1} \cdot S}(\Theta)] = \mathcal{R}_{\mathcal{S}}(\Theta),$$
where we used the change of variable $S' = g^{-1} \cdot S$ and the invariance of $\mathcal{S}$.
:::

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Group-Covariant System Distribution:** $\mathcal{S}$ is $G$-invariant: $S \sim \mathcal{S} \Rightarrow g \cdot S \sim \mathcal{S}$
>     *   [ ] **Equivariant Parametrization:** $g \cdot \mathcal{H}_\Theta(S) \simeq \mathcal{H}_{g \cdot \Theta}(g \cdot S)$ in Hypo
>     *   [ ] **Defect-Level Equivariance:** $K_{A,g \cdot S}^{(g \cdot \Theta)}(g \cdot u) = K_{A,S}^{(\Theta)}(u)$
>     *   [ ] **Existence of True Equivariant Hypostructure:** $\Theta^*$ with $\mathcal{R}_S(\Theta^*) = 0$ and $g \cdot \mathcal{H}_{\Theta^*,S} \simeq \mathcal{H}_{\Theta^*,g \cdot S}$
> *   **Output (Structural Guarantee):**
>     *   Minimizers $G$-equivariant: $\widehat{\Theta} \in G \cdot \Theta^*$
>     *   Gradient flow preserves equivariance: $g \cdot \Theta_t$ solves same flow with $g \cdot \Theta_0$
>     *   Convergence to equivariant hypostructures
> *   **Failure Condition (Debug):**
>     *   If **Equivariant Parametrization** fails → **Mode symmetry-breaking artifact** (learned structure has spurious asymmetry)
>     *   If **Local Uniqueness** fails → **Mode multiple branches** (equivariant and symmetry-broken minima coexist)

:::{prf:metatheorem} Equivariance
:label: mt-equivariance

Let $\mathcal{S}$ be a $G$-invariant system distribution, and $\{\mathcal{H}_\Theta\}$ a parametric hypostructure family satisfying Assumptions 13.57–13.59. Consider the average defect-risk $\mathcal{R}_{\mathcal{S}}(\Theta)$.

Assume:

1. **(Existence of a true equivariant hypostructure.)** There exists a parameter $\Theta^* \in \Theta_{\mathrm{adm}}$ such that:
   - For $\mathcal{S}$-a.e. system $S$, $\mathcal{H}_{\Theta^*,S}$ satisfies the axioms (C, D, SC, Cap, LS, TB, Bound, Reg, GC), and $\mathcal{R}_S(\Theta^*) = 0$.
   - The true hypostructure is $G$-equivariant in Hypo: For all $g \in G$ and all $S$:
   $$g \cdot \mathcal{H}_{\Theta^*,S} \simeq \mathcal{H}_{\Theta^*, g \cdot S}.$$
   Equivalently, the orbit $G \cdot \Theta^*$ consists of gauge-equivalent parameters encoding the same equivariant hypostructure.

2. **(Local uniqueness modulo $G$-gauge.)** The average risk $\mathcal{R}_{\mathcal{S}}(\Theta)$ admits a unique minimum orbit in a neighborhood of $\Theta^*$: there is a neighborhood $U \subset \Theta_{\mathrm{adm}}$ such that:
$$\Theta \in U, \quad \mathcal{R}_{\mathcal{S}}(\Theta) = \inf_{\Theta'} \mathcal{R}_{\mathcal{S}}(\Theta') \implies \Theta \in G \cdot \Theta^*,$$
and all points in $G \cdot \Theta^* \cap U$ are gauge-equivalent (represent the same Hypo object).

3. **(Regularity for gradient flow.)** $\mathcal{R}_{\mathcal{S}}$ is $C^1$ on $\Theta_{\mathrm{adm}}$, with Lipschitz gradient on bounded sets.

Then:

1. **(Minimizers are $G$-equivariant (up to gauge).)** Every global minimizer $\widehat{\Theta}$ of $\mathcal{R}_{\mathcal{S}}$ in $U$ lies in the orbit $G \cdot \Theta^*$, and thus represents the same equivariant hypostructure as $\Theta^*$ in Hypo. In particular, the learned hypostructure is $G$-equivariant.

2. **(Gradient flow preserves equivariance.)** Consider gradient flow on parameter space:
$$\frac{d}{dt} \Theta_t = -\nabla \mathcal{R}_{\mathcal{S}}(\Theta_t), \qquad \Theta_{t=0} = \Theta_0.$$
Then for any $g \in G$, $g \cdot \Theta_t$ solves the same gradient flow with initial condition $g \cdot \Theta_0$. In particular, if the initialization $\Theta_0$ is $G$-fixed (or lies in a $G$-orbit symmetric under a subgroup), the entire trajectory $\Theta_t$ remains in the fixed-point set (or corresponding orbit) of the group action.

3. **(Convergence to equivariant hypostructures.)** If gradient descent or gradient flow on $\mathcal{R}_{\mathcal{S}}$ converges to a minimizer in $U$ (as in {prf:ref}`mt-trainable-hypostructure-consistency`), then the limit hypostructure is gauge-equivalent to $\Theta^*$ and hence $G$-equivariant.

In short: **trainable hypostructures inherit all symmetries of the system distribution**. They cannot spontaneously break a symmetry that the true hypostructure preserves, unless there exist distinct, non-equivariant minimizers of $\mathcal{R}_{\mathcal{S}}$ outside the neighborhood $U$ (i.e. unless the theory itself has symmetric and symmetry-broken branches).
:::

:::{prf:proof}
(1) follows directly from risk invariance and local uniqueness modulo $G$.

By {prf:ref}`lem-risk-equivariance`, $\mathcal{R}_{\mathcal{S}}$ is $G$-invariant:
$$\mathcal{R}_{\mathcal{S}}(g \cdot \Theta) = \mathcal{R}_{\mathcal{S}}(\Theta) \quad \forall g \in G.$$

Let $\widehat{\Theta} \in U$ be a global minimizer of $\mathcal{R}_{\mathcal{S}}$. Then for any $g \in G$:
$$\mathcal{R}_{\mathcal{S}}(g \cdot \widehat{\Theta}) = \mathcal{R}_{\mathcal{S}}(\widehat{\Theta}) = \inf_{\Theta'} \mathcal{R}_{\mathcal{S}}(\Theta').$$
Thus $g \cdot \widehat{\Theta}$ is also a minimizer in $U$. By local uniqueness modulo orbit (Assumption 2), all such minimizers in $U$ lie on the orbit $G \cdot \Theta^*$ and correspond to the same hypostructure in Hypo. Therefore $\widehat{\Theta} \in G \cdot \Theta^*$, and the corresponding hypostructure is $G$-equivariant.

(2) Gradient flow equivariance follows from the invariance of $\mathcal{R}_{\mathcal{S}}$. By the chain rule and $G$-invariance:
$$\mathcal{R}_{\mathcal{S}}(g \cdot \Theta) = \mathcal{R}_{\mathcal{S}}(\Theta) \implies D(g \cdot \Theta)^\top \nabla \mathcal{R}_{\mathcal{S}}(g \cdot \Theta) = \nabla \mathcal{R}_{\mathcal{S}}(\Theta),$$
where $D(g \cdot \Theta)$ is the derivative of the group action at $\Theta$. Differentiating $\Theta_t \mapsto g \cdot \Theta_t$ in time gives:
$$\frac{d}{dt}(g \cdot \Theta_t) = D(g \cdot \Theta_t) \dot{\Theta}_t = -D(g \cdot \Theta_t) \nabla \mathcal{R}_{\mathcal{S}}(\Theta_t) = -\nabla \mathcal{R}_{\mathcal{S}}(g \cdot \Theta_t),$$
where the last equality uses the relation between gradients and the group action induced by $G$-invariance. Hence $g \cdot \Theta_t$ solves the same gradient flow with initial condition $g \cdot \Theta_0$.

(3) If gradient descent or continuous-time gradient flow converges to a limit $\Theta_\infty \in U$, then by (1) that limit is in the orbit $G \cdot \Theta^*$ and corresponds to the same $G$-equivariant hypostructure.
:::

:::{prf:remark} Key hypotheses

The key hypotheses are:

- **Equivariant parametrization** of the hypostructure family ({prf:ref}`assum-equivariant-parametrization`), and
- **Defect-level equivariance** ({prf:ref}`assum-group-invariance-defects`).

Together, they ensure that ``write down the axioms, compute defects, average risk, and optimize'' defines a $G$-equivariant learning problem.
:::

:::{prf:remark} No spontaneous symmetry breaking

The theorem says that if the *true* structural laws of the systems are $G$-equivariant, and the training distribution respects that symmetry, then a trainable hypostructure will not invent a spurious symmetry-breaking ontology---unless such a symmetry-breaking branch is truly present as an alternative minimum of the risk.
:::

:::{prf:remark} Structural analogue of equivariant networks

This is a structural analogue of standard results for equivariant neural networks, but formulated at the level of **axiom learning**: the objects that remain invariant are not just predictions, but the entire hypostructure (Lyapunov, dissipation, capacities, barriers, etc.).
:::

> **Key Insight:** Trainable hypostructures inherit all symmetries of the underlying system distribution. The learned axioms preserve equivariance—not just at the level of predictions, but at the level of structural components ($\Phi$, $\mathfrak{D}$, barriers, capacities). Symmetry cannot be spontaneously broken by the learning process unless the true theory itself admits symmetry-broken branches.

---


---


(ch-general-loss)=

## The General Loss Functional

:::{div} feynman-prose
Here is the practical question that faces anyone who wants to build a system that *discovers* physics rather than having it programmed in: what should the loss function be?

The answer has four pieces, and each one corresponds to a different aspect of what it means to "understand" a physical system. First, you need to get the *structure* right---the energy functional, the dissipation, the symmetries. Second, the axioms should actually hold---dissipation should be non-negative, energy should decrease along trajectories, barriers should block forbidden transitions. Third, when the axioms define variational problems (like "find the minimum energy path"), you want to solve those problems well. Fourth, and this is the clever part, you want the system to *adapt quickly* to new physics---to have learned generalizable patterns rather than memorizing specifics.

Put these together and you get a loss function that looks like physics regularized by meta-learning. The system is rewarded not just for fitting data, but for discovering *structure* that transfers across problems. This is why it works: the inductive bias of hypostructure is exactly the bias toward learnable, generalizable physics.
:::

This section defines a training objective for systems that instantiate, verify, and optimize over hypostructures. The goal is to train a parametrized system to identify hypostructures, fit soft axioms, and solve the associated variational problems.

### Overview and problem formulation

This is formally framed as **Structural Risk Minimization {cite}`Vapnik98`** over the hypothesis space of admissible hypostructures.

:::{prf:definition} Hypostructure learner
:label: def-hypostructure-learner

A **hypostructure learner** is a parametrized system with parameters $\Theta$ that, given a dynamical system $S$, produces:

1. A hypostructure $\mathbb{H}_\Theta(S) = (X, S_t, \Phi_\Theta, \mathfrak{D}_\Theta, G_\Theta)$
2. Soft axiom evaluations and defect values
3. Extremal candidates $u_{\Theta,S}$ for associated variational problems

:::

:::{prf:definition} System distribution
:label: def-system-distribution

Let $\mathcal{S}$ denote a probability distribution over dynamical systems. This includes PDEs, flows, discrete processes, stochastic systems, and other structures amenable to hypostructure analysis.
:::

:::{prf:definition} general loss functional
:label: def-general-loss-functional

The **general loss** is:
$$\mathcal{L}_{\text{gen}}(\Theta) := \mathbb{E}_{S \sim \mathcal{S}}\big[\lambda_{\text{struct}} L_{\text{struct}}(S, \Theta) + \lambda_{\text{axiom}} L_{\text{axiom}}(S, \Theta) + \lambda_{\text{var}} L_{\text{var}}(S, \Theta) + \lambda_{\text{meta}} L_{\text{meta}}(S, \Theta)\big]$$
where $\lambda_{\text{struct}}, \lambda_{\text{axiom}}, \lambda_{\text{var}}, \lambda_{\text{meta}} \geq 0$ are weighting coefficients.
:::

### Structural loss

The structural loss formulation embodies the **Maximum Entropy** principle of Jaynes {cite}`Jaynes57`: among all distributions consistent with observed constraints, select the one with maximal entropy. Here, we select the hypostructure parameters that minimize constraint violations while maintaining maximal generality.

:::{prf:definition} Structural loss functional
:label: def-structural-loss-functional

For systems $S$ with known ground-truth structure $(\Phi^*, \mathfrak{D}^*, G^*)$, define:
$$L_{\text{struct}}(S, \Theta) := d(\Phi_\Theta, \Phi^*) + d(\mathfrak{D}_\Theta, \mathfrak{D}^*) + d(G_\Theta, G^*)$$
where $d(\cdot, \cdot)$ denotes an appropriate distance on the respective spaces.
:::

:::{prf:definition} Self-consistency constraints
:label: def-self-consistency-constraints

For unlabeled systems without ground-truth annotations, define:
$$L_{\text{struct}}(S, \Theta) := \mathbf{1}[\Phi_\Theta < 0] + \mathbf{1}[\text{non-convexity along flow}] + \mathbf{1}[\text{non-}G_\Theta\text{-invariance}]$$
with indicator penalties for constraint violations.
:::

:::{prf:lemma} Structural loss interpretation
:label: lem-structural-loss-interpretation

Minimizing $L_{\text{struct}}$ encourages the learner to:

- Correctly identify conserved quantities and energy functionals
- Recognize symmetries inherent to the system
- Produce internally consistent hypostructure components

:::

:::{prf:proof}
We verify each claim:


1. **Conserved quantities:** By {prf:ref}`def-structural-loss-functional`, $L_{\text{struct}}$ includes the term $d(\Phi_\Theta, \Phi^*)$. Minimizing this term forces $\Phi_\Theta$ close to the ground-truth $\Phi^*$. By {prf:ref}`def-self-consistency-constraints`, violations of positivity ($\Phi_\Theta < 0$) incur penalty, selecting parameters where $\Phi_\Theta$ behaves as a proper energy/height functional.

2. **Symmetries:** The term $d(G_\Theta, G^*)$ ({prf:ref}`def-structural-loss-functional`) penalizes discrepancy between learned and true symmetry groups. The indicator $\mathbf{1}[\text{non-}G_\Theta\text{-invariance}]$ ({prf:ref}`def-self-consistency-constraints`) penalizes learned structures not respecting the identified symmetry.

3. **Internal consistency:** The indicator $\mathbf{1}[\text{non-convexity along flow}]$ ({prf:ref}`def-self-consistency-constraints`) enforces that $\Phi_\Theta$ and the flow $S_t$ are compatible: along trajectories, $\Phi_\Theta$ should decrease (Lyapunov property) or satisfy convexity constraints from Axiom D.


The loss $L_{\text{struct}}$ is zero if and only if all components are correctly identified and mutually consistent.
:::

### Axiom loss

:::{prf:definition} Axiom loss functional
:label: def-axiom-loss-functional

For system $S$ with trajectory distribution $\mathcal{U}_S$:
$$L_{\text{axiom}}(S, \Theta) := \sum_{A \in \mathcal{A}} w_A \, \mathbb{E}_{u \sim \mathcal{U}_S}[K_A^{(\Theta)}(u)]$$
where $K_A^{(\Theta)}$ is the defect functional for axiom $A$ under the learned hypostructure $\mathbb{H}_\Theta(S)$.
:::

:::{prf:lemma} Axiom loss interpretation
:label: lem-axiom-loss-interpretation

Minimizing $L_{\text{axiom}}$ selects parameters $\Theta$ that produce hypostructures with minimal global axiom defects.
:::

:::{prf:proof}
If the system $S$ genuinely satisfies axiom $A$, the learner is rewarded for finding parameters that make $K_A^{(\Theta)}(u)$ small. If $S$ violates $A$ in some regimes, the minimum achievable defect quantifies this failure.
:::

:::{prf:definition} Causal Enclosure Loss
:label: def-causal-enclosure-loss

Let $(\mathcal{X}, \mu, T)$ be a stochastic dynamical system and $\Pi: \mathcal{X} \to \mathcal{Y}$ a learnable coarse-graining parametrized by $\Theta$. Define $Y_t := \Pi_\Theta(X_t)$ and $Y_{t+1} := \Pi_\Theta(X_{t+1})$. The **causal enclosure loss** is:
$$L_{\text{closure}}(\Theta) := I(X_t; Y_{t+1}) - I(Y_t; Y_{t+1})$$
where $I(\cdot; \cdot)$ denotes mutual information with respect to the stationary measure $\mu$.
:::

*Interpretation:* By the chain rule, $I(X_t; Y_{t+1}) = I(Y_t; Y_{t+1}) + I(X_t; Y_{t+1} \mid Y_t)$. Thus:
$$L_{\text{closure}}(\Theta) = I(X_t; Y_{t+1} \mid Y_t)$$
This quantifies how much additional predictive information about the macro-future $Y_{t+1}$ is contained in the micro-state $X_t$ beyond what is captured by the macro-state $Y_t$. By the Closure-Curvature Duality principle, $L_{\text{closure}} = 0$ if and only if the coarse-graining $\Pi_\Theta$ is computationally closed. Minimizing $L_{\text{closure}}$ thus forces the learned hypostructure to be "Software": the macro-dynamics becomes autonomous, independent of micro-noise {cite}`Rosas2024`.

### Variational loss

:::{prf:definition} Variational loss for labeled systems
:label: def-variational-loss-for-labeled-systems

For systems with known sharp constants $C_A^*(S)$:
$$L_{\text{var}}(S, \Theta) := \sum_{A \in \mathcal{A}} \left| \text{Eval}_A(u_{\Theta,S,A}) - C_A^*(S) \right|$$
where $\text{Eval}_A$ is the evaluation functional for problem $A$ and $u_{\Theta,S,A}$ is the learner's proposed extremizer.
:::

:::{prf:definition} Extremal search loss for unlabeled systems
:label: def-extremal-search-loss-for-unlabeled-systems

For systems without known sharp constants:
$$L_{\text{var}}(S, \Theta) := \sum_{A \in \mathcal{A}} \text{Eval}_A(u_{\Theta,S,A})$$
directly optimizing toward the extremum.
:::

:::{prf:lemma} Rigorous bounds property
:label: lem-rigorous-bounds-property

Every value $\text{Eval}_A(u_{\Theta,S,A})$ constitutes a rigorous one-sided bound on the sharp constant by construction of the variational problem.
:::

:::{prf:proof}
For infimum problems, any feasible $u$ gives an upper bound: $\text{Eval}_A(u) \geq C_A^*$. For supremum problems, any feasible $u$ gives a lower bound. The learner's output is always a valid bound regardless of optimality.
:::

### Meta-learning loss

:::{prf:definition} Adapted parameters
:label: def-adapted-parameters

For system $S$ and base parameters $\Theta$, let $\Theta'_S$ denote the result of $k$ gradient steps on $L_{\text{axiom}}(S, \cdot) + L_{\text{var}}(S, \cdot)$ starting from $\Theta$:
$$\Theta'_S := \Theta - \eta \sum_{i=1}^{k} \nabla_\Theta (L_{\text{axiom}} + L_{\text{var}})(S, \Theta^{(i)})$$
where $\Theta^{(i)}$ is the parameter after $i$ steps.
:::

:::{prf:definition} Meta-learning loss
:label: def-meta-learning-loss

Define:
$$L_{\text{meta}}(S, \Theta) := \tilde{L}_{\text{axiom}}(S, \Theta'_S) + \tilde{L}_{\text{var}}(S, \Theta'_S)$$
evaluated on held-out data from $S$.
:::

:::{prf:lemma} Fast adaptation interpretation
:label: lem-fast-adaptation-interpretation

Minimizing $L_{\text{meta}}$ over the distribution $\mathcal{S}$ trains the system to:

- Quickly instantiate hypostructures for new systems (few gradient steps to fit $\Phi, \mathfrak{D}, G$)
- Rapidly identify sharp constants and extremizers

:::

:::{prf:proof}
The meta-learning objective rewards parameters $\Theta$ from which few adaptation steps suffice to achieve low loss on any system $S$. This is the MAML principle applied to hypostructure learning.
:::

### The combined general loss

This formulation mirrors **Tikhonov Regularization {cite}`Tikhonov77`** for ill-posed inverse problems, where the Hypostructure Axioms serve as the stabilizing functional.

:::{prf:metatheorem} Differentiability
:label: mt-differentiability

Under the following conditions:

1. Neural network parameterization of $\Phi_\Theta, \mathfrak{D}_\Theta, G_\Theta$
2. Defect functionals $K_A$ composed of integrals, norms, and algebraic expressions in the network outputs
3. Dominated convergence conditions as in {prf:ref}`lem-leibniz-rule-for-defect-risk`

:::

all components of $\mathcal{L}_{\text{gen}}$ are differentiable in $\Theta$.

:::{prf:proof}
**Step 1 (Component differentiability).** Each loss component $L_{\text{struct}}, L_{\text{axiom}}, L_{\text{var}}$ is differentiable by:

- Neural network differentiability (backpropagation)
- Dominated convergence for integral expressions ({prf:ref}`lem-leibniz-rule-for-defect-risk`)


**Step 2 (Meta-learning differentiability).** The adapted parameters $\Theta'_S$ depend differentiably on $\Theta$ via the chain rule through gradient steps. This is the key observation enabling MAML-style meta-learning.

**Step 3 (Expectation over $\mathcal{S}$).** Dominated convergence allows differentiation under the expectation over systems $S \sim \mathcal{S}$, given appropriate bounds.
:::

:::{prf:corollary} Backpropagation through axioms
:label: cor-backpropagation-through-axioms

Gradient descent on $\mathcal{L}_{\text{gen}}(\Theta)$ is well-defined. The gradient can be computed via backpropagation through:

- The neural network architecture
- The defect functional computations
- The meta-learning adaptation steps

:::

:::{prf:metatheorem} Universal Solver
:label: mt-universal-solver

A system trained on $\mathcal{L}_{\text{gen}}$ with sufficient capacity and training data over a diverse distribution $\mathcal{S}$ learns to:

1. **Recognize structure:** Identify state spaces, flows, height functionals, dissipation structures, and symmetry groups
2. **Enforce soft axioms:** Fit hypostructure parameters that minimize global axiom defects
3. **Solve variational problems:** Produce extremizers that approach sharp constants
4. **Adapt quickly:** Transfer to new systems with few gradient steps

:::

:::{prf:proof}
**Step 1 (Structural recognition).** Minimizing $L_{\text{struct}}$ over diverse systems trains the learner to extract the correct hypostructure components. The loss penalizes misidentification of conserved quantities, symmetries, and dissipation mechanisms.

**Step 2 (Axiom enforcement).** Minimizing $L_{\text{axiom}}$ trains the learner to find parameters under which soft axioms hold with minimal defect. The learner discovers which axioms each system satisfies and quantifies violations.

**Step 3 (Variational solving).** Minimizing $L_{\text{var}}$ trains the learner to produce increasingly sharp bounds on extremal constants. For labeled systems, the gap to known values provides direct supervision. For unlabeled systems, the extremal search pressure drives toward optimal values.

**Step 4 (Fast adaptation).** Minimizing $L_{\text{meta}}$ trains the learner's initialization to enable rapid specialization. Few gradient steps suffice to adapt the general hypostructure knowledge to any specific system.

The combination of these four loss components produces a system that instantiates and optimizes over hypostructures universally.
:::

---

### The Learnability Threshold

:::{div} feynman-prose
Here is something that should make you sit up. There is a fundamental limit to what any learning agent can learn, and it has nothing to do with the architecture of your neural network or the cleverness of your optimization algorithm. It is a *thermodynamic* limit.

The key insight is this: every dynamical system generates information at some rate---this is the Kolmogorov-Sinai entropy, which measures how many new bits of unpredictable behavior appear per unit time. And every learning agent can *absorb* information at some rate---this is the channel capacity of the learning process itself.

When the system generates information faster than the agent can absorb it, something fundamental must give. The agent cannot learn the microscopic dynamics, period. No amount of training will help. The gap between what the system produces and what the agent can capture becomes *noise*---not because it is random in any ontological sense, but because the agent lacks the capacity to track it.

But here is the beautiful thing: the agent does not just give up. It automatically discovers a *coarse-grained* description---a set of slow, predictable variables that *can* be tracked. The agent becomes a physicist, discovering effective theories. This is not programmed in. It *emerges* from the capacity constraint.
:::

This section establishes the fundamental dichotomy in learning: the transition between **perfect reconstruction** and **statistical modeling** is not a choice of algorithm, but a threshold controlled by the ratio of system entropy to agent capacity. This formalizes the $\Omega$-Layer interface between the System (Reality) and the Agent (The Learner).

:::{prf:definition} Kolmogorov-Sinai Entropy Rate
:label: def-kolmogorov-sinai-entropy-rate

Let $(X, \mathcal{B}, \mu, S_t)$ be a measure-preserving dynamical system generating trajectories $u(t)$. The **Kolmogorov-Sinai entropy** $h_{KS}(S)$ {cite}`Sinai59` is the rate at which the system generates new information (bits per unit time) that cannot be predicted from past history:
$$h_{KS}(S) := \sup_{\mathcal{P}} \lim_{n \to \infty} \frac{1}{n} H\left(\bigvee_{k=0}^{n-1} S_{-k}^{-1}\mathcal{P}\right)$$
where $\mathcal{P}$ ranges over finite measurable partitions and $H(\cdot)$ denotes Shannon entropy of a partition. Equivalently, in the continuous-time formulation:
$$h_{KS}(S) = \lim_{t \to \infty} \frac{1}{t} H(u_{[0,t]} \mid u_{(-\infty, 0]})$$
For deterministic systems, $h_{KS}$ equals the sum of positive Lyapunov exponents by **Pesin's formula** {cite}`Eckmann85`:
$$h_{KS}(S) = \int_X \sum_{\lambda_i(x) > 0} \lambda_i(x) \, d\mu(x)$$
where $\{\lambda_i(x)\}$ are the Lyapunov exponents at $x$. For stochastic systems, it includes both deterministic chaos and external noise contributions.
:::

:::{prf:definition} Agent Capacity
:label: def-agent-capacity

Let $\mathcal{A}$ be a learning agent (Hypostructure Learner) with parameter space $\Theta \subseteq \mathbb{R}^d$ and update rule $\Theta_{t+1} = \Theta_t - \eta \nabla_\Theta \mathcal{L}$. The **capacity** $C_{\mathcal{A}}$ is the maximum rate at which the agent can store and process information:
$$C_{\mathcal{A}} := \sup_{\text{inputs}} \limsup_{T \to \infty} \frac{1}{T} I(\Theta_T; \text{data}_{[0,T]})$$
This is the bandwidth of the update rule—the channel capacity of the learning process viewed as a communication channel from the environment to the agent's parameters. For neural networks with $d$ parameters, learning rate $\eta$, and batch size $B$:
$$C_{\mathcal{A}} \lesssim \eta B \cdot d \cdot \log(1/\eta)$$
The Fisher information of the parameterization provides a tighter bound: $C_{\mathcal{A}} \leq \frac{1}{2} \text{tr}(\mathcal{I}(\Theta))$ where $\mathcal{I}(\Theta)$ is the Fisher information matrix {cite}`Amari16`.
:::

:::{prf:metatheorem} The Learnability Threshold
:label: mt-the-learnability-threshold

Let an agent $\mathcal{A}$ with capacity $C_{\mathcal{A}}$ attempt to model a dynamical system $S$ with KS-entropy $h_{KS}(S)$ by minimizing the prediction loss $\mathcal{L}_{\text{pred}} := \mathbb{E}[\|u(t+\Delta t) - \hat{u}(t+\Delta t)\|^2]$. There exists a critical threshold determined by the KS-entropy that separates two fundamentally different learning regimes:


1. **The Learnable Regime** ($h_{KS}(S) < C_{\mathcal{A}}$): The system is **Microscopically Learnable**.

   - The agent recovers the exact micro-dynamics: $\|\hat{S}_t - S_t\|_{L^2(\mu)} \to 0$ as training time $T \to \infty$.
   - The effective noise term $\Sigma_T \to 0$ with rate $\Sigma_T = O(T^{-1/2})$.
   - This corresponds to **Axiom LS (Local Stiffness)** holding at the microscopic scale: the learned dynamics satisfy the Łojasiewicz gradient inequality with the true exponent $\theta$.
   - **Convergence rate:** $\mathcal{L}_{\text{pred}}(\Theta_T) \leq \mathcal{L}_{\text{pred}}(\Theta_0) \cdot \exp\left(-\frac{C_{\mathcal{A}} - h_{KS}(S)}{C_{\mathcal{A}}} \cdot T\right)$


2. **The Coarse-Grained Regime** ($h_{KS}(S) > C_{\mathcal{A}}$): The system is **Microscopically Unlearnable**.

   - Pointwise prediction error remains non-zero: $\inf_{\Theta} \mathcal{L}_{\text{pred}}(\Theta) \geq D^*(C_{\mathcal{A}}) > 0$.
   - The agent undergoes **Spontaneous Scale Symmetry Breaking**: it abandons the micro-scale and converges to a coarse-grained scale $\Lambda$ where $h_{KS}(S_\Lambda) < C_{\mathcal{A}}$.
   - The residual prediction error becomes structured noise obeying **Mode D.D (Dispersion)**.
   - **Irreducible error:** $\inf_\Theta \mathcal{L}_{\text{pred}}(\Theta) \geq \frac{1}{2\pi e} \cdot 2^{2(h_{KS}(S) - C_{\mathcal{A}})}$ (Shannon lower bound).


:::

:::{prf:proof}
**Step 1 (Information-Theoretic Setup).** We formalize the learning process as a communication channel. Let $\mathcal{D}_T = \{u(t_i)\}_{i=1}^{N_T}$ be the observed trajectory data over training duration $T$, where $N_T = T/\Delta t$ samples. The learning algorithm defines a (possibly stochastic) map:
$$\mathcal{A}: \mathcal{D}_T \mapsto \Theta_T \in \mathbb{R}^d$$
The learned model $\hat{S}_{\Theta_T}$ attempts to approximate the true dynamics $S$. By the **data processing inequality** {cite}`Shannon48`, for any function $f$ of $\Theta_T$:
$$I(S; f(\Theta_T)) \leq I(S; \Theta_T) \leq I(S; \mathcal{D}_T)$$
The mutual information between the true dynamics and observed data satisfies:
$$I(S; \mathcal{D}_T) \leq H(\mathcal{D}_T) \leq N_T \cdot h_{KS}(S) \cdot \Delta t = T \cdot h_{KS}(S)$$
The capacity constraint on the learning channel gives $I(S; \Theta_T) \leq C_{\mathcal{A}} \cdot T$.

**Step 2 (Learnable Regime: Achievability).** Suppose $h_{KS}(S) < C_{\mathcal{A}}$. We construct a learning scheme achieving zero asymptotic error.

*Construction:* Partition the parameter space into $2^{C_{\mathcal{A}} \cdot T}$ cells. By Shannon's source coding theorem, there exists an encoding of the trajectory $\mathcal{D}_T$ using at most $H(\mathcal{D}_T) + o(T)$ bits. Since $H(\mathcal{D}_T) \leq h_{KS}(S) \cdot T < C_{\mathcal{A}} \cdot T$, the trajectory can be encoded losslessly into the parameters.

*Convergence:* Let $\varepsilon > 0$ and define the typical set $\mathcal{T}_\varepsilon^{(T)} := \{u : |H(u)/T - h_{KS}(S)| < \varepsilon\}$. By the Asymptotic Equipartition Property {cite}`Shannon48`:
$$\mu(\mathcal{T}_\varepsilon^{(T)}) \to 1 \quad \text{as } T \to \infty$$
For typical trajectories, the encoding uses $(h_{KS}(S) + \varepsilon) \cdot T$ bits. Choosing $\varepsilon < C_{\mathcal{A}} - h_{KS}(S)$, lossless encoding is possible with high probability.

*Rate:* The probability of decoding error satisfies $P(\hat{S}_{\Theta_T} \neq S) \leq 2^{-T(C_{\mathcal{A}} - h_{KS}(S) - \varepsilon)}$ by the channel coding theorem. This gives the exponential convergence rate in the statement.

**Step 3 (Coarse-Grained Regime: Converse).** Suppose $h_{KS}(S) > C_{\mathcal{A}}$. We prove a lower bound on irreducible error.

*Rate-Distortion Theory:* For a source with entropy rate $h_{KS}(S)$ and squared-error distortion $d(s, \hat{s}) = \|s - \hat{s}\|^2$, the rate-distortion function $R(D)$ satisfies {cite}`Berger71`:
$$R(D) = h_{KS}(S) - \frac{1}{2}\log(2\pi e D)$$
for Gaussian sources (and provides a lower bound for general sources). Inverting:
$$D(R) = \frac{1}{2\pi e} \cdot 2^{2(h_{KS}(S) - R)}$$
Since the learning channel has rate at most $C_{\mathcal{A}}$, the minimum achievable distortion is:
$$D^*(C_{\mathcal{A}}) = \frac{1}{2\pi e} \cdot 2^{2(h_{KS}(S) - C_{\mathcal{A}})} > 0$$
This is the **Shannon lower bound** on prediction error.

*Converse argument:* Any predictor $\hat{S}_\Theta$ satisfies:
$$\mathcal{L}_{\text{pred}}(\Theta) = \mathbb{E}[\|S_t u - \hat{S}_\Theta u\|^2] \geq D^*(I(S; \Theta)) \geq D^*(C_{\mathcal{A}})$$
The first inequality is the operational meaning of rate-distortion; the second uses the capacity bound.

**Step 4 (Scale Selection via Lyapunov Spectrum).** In the coarse-grained regime, the agent must choose which information to discard. We prove the optimal strategy selects a coarse-graining $\Pi^*$ aligned with the slow manifold.

*Lyapunov decomposition:* Let $\{\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n\}$ be the Lyapunov exponents of $S$, ordered by magnitude. The contribution of each mode to entropy is {cite}`Eckmann85`:
$$h_i = \max(0, \lambda_i)$$
The total entropy is $h_{KS}(S) = \sum_{i: \lambda_i > 0} \lambda_i$.

*Optimal truncation:* Define the $k$-mode projection $\Pi_k$ retaining only modes with $|\lambda_i| \leq \lambda_k$. The entropy of the projected system is:
$$h_{KS}(S_{\Pi_k}) = \sum_{i > k: \lambda_i > 0} \lambda_i$$
The optimal scale $k^*$ is the smallest $k$ such that $h_{KS}(S_{\Pi_k}) \leq C_{\mathcal{A}}$.

*Variational characterization:* This truncation emerges from optimizing the Lagrangian:
$$\Pi^* = \arg\min_{\Pi} \left[ \mathcal{L}_{\text{pred}}(\Pi) + \lambda \cdot I(X; \Pi(X)) \right]$$
where $\lambda$ is the Lagrange multiplier enforcing the capacity constraint. By the Karush-Kuhn-Tucker conditions, the optimal projection satisfies:
$$\frac{\partial \mathcal{L}_{\text{pred}}}{\partial \Pi} = -\lambda \frac{\partial I(X; \Pi(X))}{\partial \Pi}$$
The modes with highest $\lambda_i$ contribute most to mutual information but least to long-term prediction (they decorrelate fastest). Thus, gradient descent on $\mathcal{L}_{\text{pred}}$ under capacity constraints naturally discards high-entropy, low-predictability modes.

**Step 5 (Connection to Axiom LS).** In the Laminar Phase, the learned dynamics $\hat{S}_\Theta$ converge to the true dynamics $S$. If $S$ satisfies the Łojasiewicz gradient inequality with exponent $\theta$:
$$\|\nabla \Phi(u)\| \geq c \cdot |\Phi(u) - \Phi(u^*)|^\theta$$
then the learned dynamics inherit this property with the same exponent, since:
$$\|\nabla \hat{\Phi}_\Theta(u) - \nabla \Phi(u)\| \leq \varepsilon_T \to 0$$
implies the Łojasiewicz inequality transfers to $\hat{\Phi}_\Theta$ for sufficiently large $T$. This is **Axiom LS (Local Stiffness)** at the microscopic scale.

In the Coarse-Grained Regime, Axiom LS fails at the micro-scale but is restored at the emergent macro-scale $\Pi^*(X)$, where the reduced dynamics satisfy the Łojasiewicz inequality with an effective exponent $\theta_{\text{eff}} \geq \theta$.
:::

---

### The Optimal Effective Theory

This section explains *how* the agent handles the Coarse-Grained Regime. We prove that the agent does not merely ``blur'' the data; it finds the **Computational Closure**—the variables that form a self-contained logical system decoupled from microscopic details.

:::{prf:definition} Coarse-Graining Projection
:label: def-coarse-graining-projection

A map $\Pi: X \to Y$ is a **coarse-graining** if $\dim(Y) < \dim(X)$. Formally, let $(X, \mathcal{B}_X, \mu)$ be the micro-state space and $Y$ a measurable space with $\sigma$-algebra $\mathcal{B}_Y = \Pi^{-1}(\mathcal{B}_Y)$. The macro-state is $y_t := \Pi(x_t)$, and the induced macro-dynamics are:
$$\bar{S}_t: Y \to \mathcal{P}(Y), \quad \bar{S}_t(y) := \mathbb{E}[\Pi(S_t(x)) \mid \Pi(x) = y]$$
where the expectation averages over micro-states compatible with macro-state $y$ using the conditional measure $\mu(\cdot \mid \Pi^{-1}(y))$. When this expectation is deterministic (i.e., concentrates on a single point), we write $\bar{S}_t: Y \to Y$.
:::

:::{prf:definition} Closure Defect
:label: def-closure-defect

The **closure defect** measures how much the macro-dynamics depend on discarded micro-details:
$$\delta_\Pi := \mathbb{E}_{x \sim \mu}\left[\|\Pi(S_t(x)) - \bar{S}_t(\Pi(x))\|^2\right]^{1/2}$$
Equivalently, in terms of conditional distributions:
$$\delta_\Pi^2 = \mathbb{E}_{y \sim \Pi_*\mu}\left[\text{Var}(\Pi(S_t(x)) \mid \Pi(x) = y)\right]$$
If $\delta_\Pi = 0$, the macro-dynamics are **autonomously closed**: the conditional distribution $P(y_{t+1} \mid x_t)$ depends on $x_t$ only through $y_t = \Pi(x_t)$. This is the ``Software decoupled from Hardware'' condition—the emergent description forms a **Markov factor** of the original dynamics.
:::

:::{prf:definition} Predictive Information
:label: def-predictive-information

The **predictive information** of a coarse-graining $\Pi$ over time horizon $\tau$ is:
$$I_{\text{pred}}^\tau(\Pi) := I(\Pi(X_{\text{past}}); \Pi(X_{\text{future}})) = I(Y_{(-\infty, 0]}; Y_{[0, \tau]})$$
where $Y_t = \Pi(X_t)$. This measures how much the macro-past tells us about the macro-future—the ``useful'' information retained by the projection.
:::

:::{prf:metatheorem} The Renormalization Variational Principle
:label: mt-the-renormalization-variational-principle

Let $S$ be a chaotic dynamical system with $h_{KS}(S) > C_{\mathcal{A}}$, and let an agent minimize the General Loss $\mathcal{L}_{\text{gen}}$ over projections $\Pi: X \to Y$ with $\dim(Y) \leq d_{\max}$. Then:


1. **(Existence)** There exists an optimal coarse-graining $\Pi^*$ achieving the infimum of $\mathcal{L}_{\text{gen}}$.

2. **(Characterization)** $\Pi^*$ minimizes the **Information Bottleneck Lagrangian** {cite}`Tishby99`:
$$\mathcal{L}_{\text{IB}}(\Pi; \beta) := I(X; \Pi(X)) - \beta \cdot I(\Pi(X_{\text{past}}); \Pi(X_{\text{future}}))$$
for some $\beta^* > 0$ determined by the capacity constraint $I(X; \Pi(X)) \leq C_{\mathcal{A}}$.

3. **(Axiom Compatibility)** The induced macro-hypostructure $\mathbb{H}_{\Pi^*} = (Y, \bar{S}_t, \bar{\Phi}, \bar{\mathfrak{D}})$ satisfies **Axiom D (Dissipation)** and **Axiom TB (Topological Barrier)** with effective constants.


**Consequences:**


1. **Emergence of Macroscopic Laws.** The agent does not learn the chaotic micro-map $x_{t+1} = f(x_t)$. It learns an effective stochastic map:
$$y_{t+1} = g(y_t) + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \Sigma_{\Pi^*})$$
where $g: Y \to Y$ is the emergent deterministic macro-dynamics and $\Sigma_{\Pi^*} = \delta_{\Pi^*}^2$ is the residual variance. Examples: Navier-Stokes from molecular dynamics, Boltzmann equation from particle systems, mean-field equations from interacting spins.

2. **Noise as Ignored Information.** The residual error $\eta_t$ is not ontologically random; it is the projection of deterministic chaos from the ignored dimensions. Formally:
$$\eta_t = \Pi(S_t(x)) - \bar{S}_t(\Pi(x)) = \Pi(S_t(x)) - g(y_t)$$
The agent models this as **stochastic noise** with correlation structure inherited from the micro-dynamics. This satisfies **Mode D.D (Dispersion)** when $\eta_t$ decorrelates on the fast timescale $\tau_{\text{fast}} \ll \tau_{\text{macro}}$.

3. **Inertial Manifold Selection.** The optimal projection $\Pi^*$ aligns with the **Slow Manifold** $\mathcal{M}_{\text{slow}} \subset X$—the subspace spanned by eigenvectors of the linearized operator $DS$ with eigenvalues closest to the unit circle. This is the inertial manifold {cite}`FoiasTemam88`: a finite-dimensional, exponentially attracting, positively invariant manifold that captures the long-term dynamics.

:::

:::{prf:proof}
**Step 1 (Loss Decomposition).** We derive the structure of the prediction loss in terms of information-theoretic quantities. Let $Y_t = \Pi(X_t)$ be the macro-trajectory. The prediction loss for the macro-dynamics is:
$$\mathcal{L}_{\text{pred}}^{\text{macro}}(\Pi) = \mathbb{E}[\|Y_{t+1} - \hat{Y}_{t+1}\|^2]$$
where $\hat{Y}_{t+1} = \bar{S}_t(Y_t)$ is the optimal predictor given only macro-information.

By the law of total variance:
$$\mathcal{L}_{\text{pred}}^{\text{macro}} = \underbrace{\mathbb{E}[\text{Var}(Y_{t+1} \mid Y_t)]}_{\text{Intrinsic macro-uncertainty}} = \underbrace{\mathbb{E}[\text{Var}(Y_{t+1} \mid Y_{-\infty:t})]}_{\text{Asymptotic uncertainty}} + \underbrace{\mathbb{E}[\text{Var}(\mathbb{E}[Y_{t+1} \mid Y_{-\infty:t}] \mid Y_t)]}_{\text{Memory loss}}$$

In terms of entropies, using the Gaussian approximation for analytical tractability:
$$\mathcal{L}_{\text{pred}}^{\text{macro}} \approx \frac{1}{2\pi e} 2^{2H(Y_{t+1} \mid Y_t)}$$

The conditional entropy decomposes as:
$$H(Y_{t+1} \mid Y_t) = H(Y_{t+1} \mid X_t) + I(Y_{t+1}; X_t \mid Y_t)$$
The first term is the **intrinsic noise** (entropy of the macro-future given full micro-information); the second is the **closure violation** (additional uncertainty from not knowing the micro-state).

**Step 2 (Information Bottleneck Derivation).** The agent faces a constrained optimization: minimize prediction error subject to complexity bound $I(X; Y) \leq C_{\mathcal{A}}$. The Lagrangian is:
$$\mathcal{L}(\Pi, \beta) = \mathcal{L}_{\text{pred}}^{\text{macro}}(\Pi) + \beta \cdot (I(X; \Pi(X)) - C_{\mathcal{A}})$$

For the Gaussian case, the prediction loss satisfies {cite}`Tishby99`:
$$\mathcal{L}_{\text{pred}}^{\text{macro}} \propto 2^{-2I(Y_{\text{past}}; Y_{\text{future}})}$$

Thus, minimizing prediction error is equivalent to maximizing predictive information. The Lagrangian becomes:
$$\mathcal{L}_{\text{IB}}(\Pi; \beta) = I(X; \Pi(X)) - \beta \cdot I(\Pi(X_{\text{past}}); \Pi(X_{\text{future}}))$$

The first term penalizes **complexity** (how much micro-information is retained); the second rewards **relevance** (how predictive the retained information is).

**Step 3 (Optimal Projection Structure).** We characterize the critical points of $\mathcal{L}_{\text{IB}}$. Taking the functional derivative with respect to $\Pi$:
$$\frac{\delta \mathcal{L}_{\text{IB}}}{\delta \Pi} = \frac{\delta I(X; Y)}{\delta \Pi} - \beta \frac{\delta I(Y_{\text{past}}; Y_{\text{future}})}{\delta \Pi} = 0$$

Using the chain rule for mutual information:
$$\frac{\delta I(X; Y)}{\delta \Pi}(x) = \log \frac{p(x \mid y)}{p(x)} = \log \frac{p(y \mid x)}{p(y)}$$

The stationarity condition becomes:
$$p(y \mid x) \propto p(y) \exp\left(\beta \cdot \mathbb{E}_{x' \sim p(\cdot \mid y)}[\log p(y_{\text{future}} \mid x')]\right)$$

This is a self-consistent equation: the projection $\Pi$ determines the macro-distribution $p(y)$, which in turn determines the optimal projection. The fixed points correspond to **sufficient statistics** for predicting the future—minimal representations that preserve predictive information.

**Step 4 (Spectral Characterization).** For linear dynamics $S_t = e^{At}$ with spectrum $\{\lambda_i\}$, the optimal projection has an explicit form. Let $\{v_i\}$ be the eigenvectors of $A$, ordered by $|\text{Re}(\lambda_i)|$ ascending (slowest modes first).

The predictive information of mode $i$ over time horizon $\tau$ is:
$$I_i(\tau) = -\frac{1}{2}\log(1 - e^{-2|\text{Re}(\lambda_i)|\tau}) \approx |\text{Re}(\lambda_i)|^{-1} \cdot \tau^{-1}$$
for large $\tau$. Slow modes (small $|\text{Re}(\lambda_i)|$) carry more predictive information per bit of complexity.

The complexity cost of retaining mode $i$ is proportional to its entropy rate contribution:
$$I_i(X; Y) \propto \max(0, \text{Re}(\lambda_i))$$

The optimal projection $\Pi^*$ retains the $k^*$ slowest modes, where $k^*$ maximizes:
$$\sum_{i=1}^{k} I_i(\tau) - \beta \sum_{i=1}^{k} \max(0, \text{Re}(\lambda_i)) \quad \text{subject to} \quad \sum_{i=1}^{k} \max(0, \text{Re}(\lambda_i)) \leq C_{\mathcal{A}}$$

This is precisely the **slow manifold**: $\Pi^*(X) = \text{span}\{v_1, \ldots, v_{k^*}\}$.

**Step 5 (Nonlinear Extension via Inertial Manifolds).** For nonlinear systems, the slow manifold generalizes to the **inertial manifold** $\mathcal{M}$ {cite}`FoiasTemam88`. This is a finite-dimensional manifold satisfying:

    1. **Positive invariance:** $S_t(\mathcal{M}) \subseteq \mathcal{M}$ for $t \geq 0$
    2. **Exponential attraction:** $\text{dist}(S_t(x), \mathcal{M}) \leq C e^{-\gamma t} \text{dist}(x, \mathcal{M})$ for some $\gamma > 0$
    3. **Asymptotic completeness:** Every trajectory is shadowed by a trajectory on $\mathcal{M}$


The projection $\Pi^*: X \to \mathcal{M}$ minimizes closure defect:
$$\delta_{\Pi^*} = \sup_{x \in X} \text{dist}(S_t(x), S_t(\Pi^*(x))) \leq C e^{-\gamma t}$$

The macro-dynamics on $\mathcal{M}$ form a finite-dimensional ODE that captures the essential long-term behavior.

**Step 6 (Axiom Verification).** We verify that the induced macro-hypostructure satisfies the core axioms.

*Axiom D (Dissipation):* Define the macro-height $\bar{\Phi}(y) := \inf_{x: \Pi(x) = y} \Phi(x)$. Then:
$$\frac{d}{dt}\bar{\Phi}(\bar{S}_t(y)) = \mathbb{E}\left[\frac{d}{dt}\Phi(S_t(x)) \mid \Pi(x) = y\right] \leq -\mathbb{E}[\mathfrak{D}(S_t(x)) \mid \Pi(x) = y] =: -\bar{\mathfrak{D}}(y)$$
The macro-dissipation $\bar{\mathfrak{D}}$ is non-negative, establishing Axiom D at the macro-scale.

*Axiom TB (Topological Barrier):* The topological sectors of $X$ project to sectors of $Y$. If $\mathcal{T}_X = \{T_\alpha\}$ is the sector decomposition of $X$, then $\mathcal{T}_Y = \{\Pi(T_\alpha)\}$ provides a (possibly coarser) decomposition of $Y$. The barrier heights satisfy:
$$\Delta_Y(\Pi(T_\alpha), \Pi(T_\beta)) \leq \Delta_X(T_\alpha, T_\beta)$$
with equality when the projection respects the topological structure. Axiom TB at the macro-scale inherits from the micro-scale.

**Step 7 (Renormalization Group Interpretation).** The optimal projection $\Pi^*$ is a **Renormalization Group (RG) fixed point** {cite}`Wilson71`. Define the RG transformation $\mathcal{R}_\ell$ as coarse-graining by length scale $\ell$:
$$\mathcal{R}_\ell: \Pi \mapsto \Pi \circ \Pi_\ell$$
where $\Pi_\ell$ averages over balls of radius $\ell$. The fixed point condition $\mathcal{R}_\ell(\Pi^*) \sim \Pi^*$ (up to rescaling) means:
$$\Pi^* \circ \Pi_\ell = \Pi^* \quad \text{(self-similarity)}$$

At the fixed point, the effective theory is **scale-invariant**: further coarse-graining does not change the form of the macro-dynamics, only rescales parameters. The effective coupling constants (coefficients in $g(y)$) flow to fixed values under RG.

This completes the proof: gradient descent on $\mathcal{L}_{\text{gen}}$ under capacity constraints converges to the RG fixed point, which is the optimal coarse-graining for prediction.
:::

---

### Summary: The Universal Simulator Guarantee

The two preceding metatheorems provide the rigorous guarantee for the ``Glass Box'' nature of the AGI learner:

1. **If the world is simple** ($h_{KS}(S) < C_{\mathcal{A}}$): The AGI becomes a **perfect simulator**—Laplace's Demon realized. It reconstructs the exact microscopic laws and predicts with arbitrary precision. **Axiom LS** holds at all scales.

2. **If the world is complex** ($h_{KS}(S) > C_{\mathcal{A}}$): The AGI becomes a **physicist**. It automatically derives the ``Thermodynamics'' of the system, discarding chaotic micro-details to present the **Effective Laws of Motion** at the optimal scale. The agent discovers:
   - The correct macro-variables (order parameters, conserved quantities)
   - The emergent dynamics (hydrodynamic equations, mean-field theories)
   - The noise model for unresolved scales (stochastic forcing satisfying Mode D.D)

3. **It never ``hallucinates'' noise.** The agent explicitly models the boundary between **Signal** (Mode C.C / Axiom LS: learnable, structured, deterministic) and **Entropy** (Mode D.D: unlearnable, modeled as stochastic). The transition is not ad hoc but emerges from the capacity constraint $h_{KS} \lessgtr C_{\mathcal{A}}$.

This is the derivation of **Effective Field Theory** from first principles of learning: the scale of description is not chosen by the physicist but discovered by the optimization process. The AGI's internal model is always interpretable as physics at some scale—either exact micro-physics or emergent macro-physics with explicit noise terms.

---

### Non-differentiable environments

:::{prf:definition} RL hypostructure
:label: def-rl-hypostructure

In a reinforcement learning setting, define:

- **State space:** $X$ = agent state + environment state
- **Flow:** $S_t(x_t) = x_{t+1}$ where $x_{t+1}$ results from agent policy $\pi_\theta$ choosing action $a_t$ and environment producing the next state
- **Trajectory:** $\tau = (x_0, a_0, x_1, a_1, \ldots, x_T)$

:::

:::{prf:definition} Trajectory functional
:label: def-trajectory-functional

Define the global undiscounted objective:
$$\mathcal{L}(\tau) := F(x_0, a_0, \ldots, x_T)$$
where $F$ encodes the quantity of interest (negative total reward, stability margin, hitting time, constraint violation, etc.).
:::

:::{prf:lemma} Score function gradient
:label: lem-score-function-gradient

For policy $\pi_\theta$ and expected loss $J(\theta) := \mathbb{E}_{\tau \sim \pi_\theta}[\mathcal{L}(\tau)]$:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\mathcal{L}(\tau) \nabla_\theta \log \pi_\theta(\tau)]$$
where $\log \pi_\theta(\tau) = \sum_{t=0}^{T-1} \log \pi_\theta(a_t | x_t)$.
:::

:::{prf:proof}
Standard policy gradient derivation:
$$\nabla_\theta J(\theta) = \nabla_\theta \int \mathcal{L}(\tau) p_\theta(\tau) d\tau = \int \mathcal{L}(\tau) p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) d\tau.$$
The environment dynamics contribute to $p_\theta(\tau)$ but not to $\nabla_\theta \log p_\theta(\tau)$, which depends only on the policy.
:::

:::{prf:metatheorem} Non-Differentiable Extension
:label: mt-non-differentiable-extension

Even when the environment transition $x_{t+1} = f(x_t, a_t, \xi_t)$ is non-differentiable (discrete, stochastic, or black-box), the expected loss $J(\theta) = \mathbb{E}[\mathcal{L}(\tau)]$ is differentiable in the policy parameters $\theta$.
:::

:::{prf:proof}
The key observation is that we differentiate the **expectation** of the trajectory functional, not the environment map itself. The dependence of the trajectory distribution on $\theta$ enters only through the policy $\pi_\theta$, which is differentiable. The score function gradient ({prf:ref}`lem-score-function-gradient`) requires only:

1. Sampling trajectories from $\pi_\theta$
2. Evaluating $\mathcal{L}(\tau)$
3. Computing $\nabla_\theta \log \pi_\theta(\tau)$


None of these require differentiating through the environment.
:::

:::{prf:corollary} No discounting required
:label: cor-no-discounting-required

The global loss $\mathcal{L}(\tau)$ is defined directly on finite or stopping-time trajectories. Well-posedness is ensured by:

- Finite horizon $T < \infty$
- Absorbing states terminating trajectories
- Stability structure of the hypostructure

:::

Discounting becomes an optional modeling choice, not a mathematical necessity.

:::{prf:proof}
For finite $T$, the trajectory space is well-defined and the expectation finite. For infinite-horizon problems with absorbing states, the stopping time is almost surely finite under appropriate conditions.
:::

:::{prf:corollary} RL as hypostructure instance
:label: cor-rl-as-hypostructure-instance

Backpropagating a global loss through a non-differentiable RL environment is the decision-making instance of the general pattern:

1. Treat system + agent as a hypostructure over trajectories
2. Define a global Lyapunov/loss functional on trajectory space
3. Differentiate its expectation with respect to agent parameters
4. Perform gradient-based optimization without discounting

:::

---

### Structural Identifiability

This section establishes that the defect functionals introduced in {ref}`ch-meta-learning` determine the hypostructure components from axioms alone, and that parametric families of hypostructures are learnable under minimal extrinsic conditions. The philosophical foundation is the **univalence axiom** of Homotopy Type Theory {cite}`HoTT13`: identity is equivalent to equivalence. Two hypostructures are identified if and only if they are structurally equivalent.

:::{prf:definition} Defect signature
:label: def-defect-signature

For a parametric hypostructure $\mathcal{H}_\Theta$ and trajectory class $\mathcal{U}$, the **defect signature** is the function:
$$\mathsf{Sig}(\Theta): \mathcal{U} \to \mathbb{R}^{|\mathcal{A}|}, \quad \mathsf{Sig}(\Theta)(u) := \big(K_A^{(\Theta)}(u)\big)_{A \in \mathcal{A}}$$
where $\mathcal{A} = \{C, D, SC, Cap, LS, TB, Bound\}$ is the set of axiom labels.
:::

:::{prf:definition} Rich trajectory class
:label: def-rich-trajectory-class

A trajectory class $\mathcal{U}$ is **rich** if:

1. $\mathcal{U}$ is closed under time shifts: if $u \in \mathcal{U}$ and $s > 0$, then $u(\cdot + s) \in \mathcal{U}$.
2. For $\mu$-almost every initial condition $x \in X$, at least one finite-energy trajectory starting at $x$ belongs to $\mathcal{U}$.
:::

:::{prf:definition} Action reconstruction applicability
:label: def-action-reconstruction-applicability

The hypostructure $\mathcal{H}_\Theta$ satisfies **action reconstruction** if axioms (D), (LS), (GC) hold and the underlying metric structure is such that the canonical Lyapunov functional equals the geodesic action with respect to the Jacobi metric $g_{\mathfrak{D}} = \mathfrak{D}_\Theta \cdot g$.
:::

:::{prf:metatheorem} Defect Reconstruction
:label: mt-defect-reconstruction-2

Let $\{\mathcal{H}_\Theta\}_{\Theta \in \Theta_{\mathrm{adm}}}$ be a parametric family of hypostructures satisfying axioms (C, D, SC, Cap, LS, TB, Bound, Reg) and (GC) on gradient-flow trajectories. Suppose:

1. **(A1) Rich trajectories.** The trajectory class $\mathcal{U}$ is rich in the sense of {prf:ref}`def-rich-trajectory-class`.
2. **(A2) Action reconstruction.** {prf:ref}`def-action-reconstruction-applicability` holds for each $\Theta$.

Then for each $\Theta$, the defect signature $\mathsf{Sig}(\Theta)$ determines, up to Hypo-isomorphism:

1. The semiflow $S_t$ (on the support of $\mathcal{U}$)
2. The dissipation $\mathfrak{D}_\Theta$ along trajectories
3. The height functional $\Phi_\Theta$ (up to an additive constant)
4. The scaling exponents and barrier constants
5. The safe manifold $M$
6. The boundary interface $(\mathcal{B}_\Theta, \mathrm{Tr}_\Theta, \mathcal{J}_\Theta, \mathcal{R}_\Theta)$

There exists a reconstruction operator $\mathcal{R}: \mathsf{Sig}(\Theta) \mapsto (\Phi_\Theta, \mathfrak{D}_\Theta, S_t, \mathcal{B}_\Theta, \mathrm{Tr}_\Theta, \mathcal{J}_\Theta, \mathcal{R}_\Theta, \text{barriers}, M)$ built from the axioms and defect functional definitions alone.
:::

:::{prf:proof}
**Step 1 (Recover $S_t$ from $K_C$).** By {prf:ref}`def-parametric-defect-functional`, $K_C^{(\Theta)}(u) := \|S_t(u(s)) - u(s+t)\|$ for appropriate $s, t$. Axiom (C) and (Reg) ensure that true trajectories are exactly those with $K_C = 0$ ({prf:ref}`lem-defect-characterization`). Since $\mathcal{U}$ is closed under time shifts (A1), the unique semiflow $S_t$ is determined as the one whose orbits saturate the zero-defect locus of $K_C$.

**Step 2 (Recover $\partial_t \Phi_\Theta + \mathfrak{D}_\Theta$ from $K_D$).** By {prf:ref}`def-parametric-defect-functional`:
$$K_D^{(\Theta)}(u) = \int_T \max\big(0, \partial_t \Phi_\Theta(u(t)) + \mathfrak{D}_\Theta(u(t))\big) \, dt.$$
Axiom (D) requires $\partial_t \Phi_\Theta + \mathfrak{D}_\Theta \leq 0$ along trajectories. Thus $K_D^{(\Theta)}(u) = 0$ if and only if the energy-dissipation balance holds exactly. The zero-defect condition identifies the canonical dissipation-saturated representative.

**Step 3 (Recover $\mathfrak{D}_\Theta$ from metric and trajectories).** Axiom (Reg) provides metric structure with velocity $|\dot{u}(t)|_g$. Axiom (GC) on gradient-flow orbits gives $|\dot{u}|_g^2 = \mathfrak{D}_\Theta$. Combined with (D), propagation along the rich trajectory class determines $\mathfrak{D}_\Theta$ everywhere via the Action Reconstruction principle ({prf:ref}`mt-functional-reconstruction`).

**Step 4 (Recover $\Phi_\Theta$ from $\mathfrak{D}_\Theta$ and LS + GC).** The Action Reconstruction Theorem states: (D) + (LS) + (GC) $\Rightarrow$ the canonical Lyapunov $\mathcal{L}$ is the geodesic action with respect to $g_{\mathfrak{D}}$. By the Canonical Lyapunov Theorem ({prf:ref}`mt-krnl-lyapunov`), $\mathcal{L}$ equals $\Phi_\Theta$ up to an additive constant. Once $\mathfrak{D}_\Theta$ and $M$ are known, $\Phi_\Theta$ is reconstructed.

**Step 5 (Recover exponents and barriers from remaining defects).** The SC defect compares observed scaling behavior with claimed exponents $(\alpha_\Theta, \beta_\Theta)$. Minimizing over trajectories identifies the unique exponents. Similarly, Cap/TB/LS defects compare actual behavior with capacity/topological/Łojasiewicz bounds; the barrier constants are the unique values at which defects transition from positive to zero.

**Step 6 (Recover boundary interface from $K_{Bound}$).** The boundary defect compares observed traces and fluxes against the boundary data object $\mathcal{B}_\Theta$ and its morphisms. The zero-defect locus of $K_{Bound}$ identifies the admissible traces and flux balance, while the reinjection term determines $\mathcal{R}_\Theta$ up to the equivalence permitted by the boundary axiom (Dirichlet/Neumann/Feller classes as in {prf:ref}`ax-boundary`). This step is not optional: without matching boundary data, the hypostructure is incomplete even if bulk defects vanish.
:::

**Key Insight:** The reconstruction operator $\mathcal{R}$ is a derived object of the framework—not a new assumption. Every step uses existing axioms and metatheorems (Structural Resolution, Canonical Lyapunov, Action Reconstruction).

---

:::{prf:definition} Persistent excitation
:label: def-persistent-excitation

A trajectory distribution $\mu$ on $\mathcal{U}$ satisfies **persistent excitation** if its support explores a full-measure subset of the accessible phase space: for every open set $U \subset X$ with positive Lebesgue measure, $\mu(\{u : u(t) \in U \text{ for some } t\}) > 0$.
:::

:::{prf:definition} Nondegenerate parametrization
:label: def-nondegenerate-parametrization

The parametric family $\{\mathcal{H}_\Theta\}_{\Theta \in \Theta_{\mathrm{adm}}}$ has **nondegenerate parametrization** if the map
$$\Theta \mapsto (\Phi_\Theta, \mathfrak{D}_\Theta, \mathcal{B}_\Theta, \mathrm{Tr}_\Theta, \mathcal{J}_\Theta, \mathcal{R}_\Theta)$$
is locally Lipschitz and injective: there exists $c > 0$ such that for $\mu$-almost every $x \in X$:
$$|\Phi_\Theta(x) - \Phi_{\Theta'}(x)| + |\mathfrak{D}_\Theta(x) - \mathfrak{D}_{\Theta'}(x)| + \mathrm{dist}_{\partial}((\mathcal{B}_\Theta, \mathrm{Tr}_\Theta, \mathcal{J}_\Theta, \mathcal{R}_\Theta), (\mathcal{B}_{\Theta'}, \mathrm{Tr}_{\Theta'}, \mathcal{J}_{\Theta'}, \mathcal{R}_{\Theta'})) \geq c \, |\Theta - \Theta'|.$$
:::

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom Satisfaction:** $\mathcal{H}_\Theta$ satisfies axioms (C, D, SC, Cap, LS, TB, Bound, Reg, GC) for each $\Theta$
>     *   [ ] **(C1) Persistent Excitation:** Trajectory distribution $\mu$ explores full-measure subset of accessible phase space
>     *   [ ] **(C2) Nondegenerate Parametrization:** $|\Phi_\Theta(x) - \Phi_{\Theta'}(x)| + |\mathfrak{D}_\Theta(x) - \mathfrak{D}_{\Theta'}(x)| + \mathrm{dist}_{\partial}((\mathcal{B}_\Theta, \mathrm{Tr}_\Theta, \mathcal{J}_\Theta, \mathcal{R}_\Theta), (\mathcal{B}_{\Theta'}, \mathrm{Tr}_{\Theta'}, \mathcal{J}_{\Theta'}, \mathcal{R}_{\Theta'})) \geq c|\Theta - \Theta'|$
>     *   [ ] **(C3) Regular Parameter Space:** $\Theta_{\mathrm{adm}}$ is a metric space
> *   **Output (Structural Guarantee):**
>     *   Exact identifiability up to gauge: $\mathsf{Sig}(\Theta) = \mathsf{Sig}(\Theta') \Rightarrow \mathcal{H}_\Theta \cong \mathcal{H}_{\Theta'}$
>     *   Local quantitative identifiability: $|\Theta - \tilde{\Theta}| \leq C\varepsilon$ when signature difference $\leq \varepsilon$
>     *   Well-conditioned stability of signature map
> *   **Failure Condition (Debug):**
>     *   If **(C1) Persistent Excitation** fails → **Mode data insufficiency** (unexplored regions, indistinguishable parameters)
>     *   If **(C2) Nondegeneracy** fails → **Mode parameter aliasing** (different $\Theta$ produce same $(\Phi, \mathfrak{D})$)

:::{prf:metatheorem} Meta-Identifiability
:label: mt-meta-identifiability

Let $\{\mathcal{H}_\Theta\}_{\Theta \in \Theta_{\mathrm{adm}}}$ be a parametric family satisfying:

1. Axioms (C, D, SC, Cap, LS, TB, Bound, Reg, GC) for each $\Theta$
2. **(C1) Persistent excitation:** The trajectory distribution satisfies {prf:ref}`def-persistent-excitation`
3. **(C2) Nondegenerate parametrization:** {prf:ref}`def-nondegenerate-parametrization` holds
4. **(C3) Regular parameter space:** $\Theta_{\mathrm{adm}}$ is a metric space

Then:

1. **(Exact identifiability up to gauge.)** If $\mathsf{Sig}(\Theta) = \mathsf{Sig}(\Theta')$ as functions on $\mathcal{U}$, then $\mathcal{H}_\Theta \cong \mathcal{H}_{\Theta'}$ as objects of Hypo.

2. **(Local quantitative identifiability.)** There exist constants $C, \varepsilon_0 > 0$ such that if
$$\sup_{u \in \mathcal{U}} \sum_{A \in \mathcal{A}} \big| K_A^{(\Theta)}(u) - K_A^{(\Theta^*)}(u) \big| \leq \varepsilon < \varepsilon_0,$$
then there exists a representative $\tilde{\Theta}$ of the equivalence class $[\Theta^*]$ with $|\Theta - \tilde{\Theta}| \leq C \varepsilon$.

The map $[\Theta] \in \Theta_{\mathrm{adm}}/{\sim} \mapsto \mathsf{Sig}(\Theta)$ is locally injective and well-conditioned.
:::

:::{prf:proof}
**Step 1 (Invoke Defect Reconstruction).** By {prf:ref}`mt-defect-reconstruction-2`, $\mathsf{Sig}(\Theta)$ determines $(\Phi_\Theta, \mathfrak{D}_\Theta, S_t, \text{barriers}, M)$ via the reconstruction operator $\mathcal{R}$.

**Step 2 (Apply nondegeneracy).** By (C2), equal signatures imply equal structural data $(\Phi_\Theta, \mathfrak{D}_\Theta)$ up to gauge. Equal structural data plus equal $S_t$ (from Step 1) gives Hypo-isomorphism.

**Step 3 (Quantitative bound).** The reconstruction $\mathcal{R}$ inherits Lipschitz constants from the axiom-derived formulas. Combined with the nondegeneracy constant $c$ from (C2), perturbations in signature of size $\varepsilon$ produce perturbations in $\Theta$ of size at most $C\varepsilon$ where $C = L_{\mathcal{R}}/c$.
:::

**Key Insight:** Meta-Identifiability reduces parameter learning to defect minimization. Minimizing $\mathcal{R}_A(\Theta) = \int_{\mathcal{U}} K_A^{(\Theta)}(u) \, d\mu(u)$ over $\Theta$ converges to the true hypostructure as trajectory data increases.

---

:::{prf:remark} Irreducible extrinsic conditions

The hypotheses (C1)--(C3) cannot be absorbed into the hypostructure axioms:

1. **Nondegenerate parametrization (C2)** concerns the human choice of coordinates on the space of hypostructures. The axioms constrain $(\Phi, \mathfrak{D}, \ldots)$ once chosen, but do not force any particular parametrization to be injective or Lipschitz. This is about representation, not physics.
2. **Data richness (C1)** concerns the observer's sampling procedure. The axioms determine what trajectories can exist; they do not guarantee that a given dataset $\mathcal{U}$ actually samples them representatively. This is about epistemics, not dynamics.

Everything else---structure reconstruction, canonical Lyapunov, barrier constants, scaling exponents, failure mode classification---follows from the axioms and the preceding metatheorems.
:::

:::{prf:corollary} Foundation for trainable hypostructures
:label: cor-foundation-for-trainable-hypostructures

The Meta-Identifiability Theorem provides the theoretical foundation for the general loss ({prf:ref}`def-hypostructure-learner`): minimizing the axiom defect $\mathcal{R}_A(\Theta)$ over parameters $\Theta$ converges to the true hypostructure as data increases, with the only requirements being (C1)–(C3).
:::


(ch-agi-limit)=

## The AGI Limit (The Omega-Layer)

:::{div} feynman-prose
Now we come to the question that has been lurking in the background all along: what is the *status* of this framework itself?

Look, we have spent all these pages developing a theory of how to learn physics from data. But the framework itself is a theory. And the learner that uses the framework is a physical system. So there is a loop here. The framework describes systems that learn frameworks. Can it describe itself?

This is not philosophy. It is a concrete mathematical question, and it has a concrete answer. The answer comes from a beautiful connection between algorithmic information theory and categorical logic. Think of all possible scientific theories as points in a space, where distance measures how different two theories are. Then ask: what happens when an ideal learner explores this space, guided by the principle of finding the simplest theory that fits the data?

The answer is that it converges to a fixed point. And that fixed point has a special property: it is the theory that correctly describes *the process of scientific discovery itself*. The hypostructure framework is not arbitrary. It is the unique self-consistent description of how bounded agents can learn structure from observation.

This is what I call the Omega-Layer: the place where the framework bites its own tail and finds that it fits.
:::

*The self-referential consistency of the Hypostructure framework via Algorithmic Information Theory and Categorical Logic.*

### The Space of Theories

#### Motivation

The preceding sections established the Hypostructure as a framework for describing physical systems. A natural question arises: what is the status of the framework itself? Is it merely one theory among many, or does it occupy a distinguished position in the space of possible theories?

This section addresses this question using **Algorithmic Information Theory** {cite}`Kolmogorov65`, {cite}`Chaitin66`, and {cite}`Solomonoff64` and **Categorical Logic** {cite}`Lawvere69` and {cite}`MacLane71`. We prove that the Hypostructure is the **fixed point** of optimal scientific inquiry---the theory that an ideal learning agent must converge to.

#### Formal Definitions

:::{prf:definition} Formal Theory
:label: def-formal-theory

A **formal theory** $T$ is a recursively enumerable set of sentences in a first-order language $\mathcal{L}$, closed under logical consequence. Equivalently, $T$ can be represented as a Turing machine $M_T$ that enumerates the theorems of $T$.
:::

:::{prf:definition} The Space of Theories
:label: def-the-space-of-theories

Let $\Sigma = \{0, 1\}$ be the binary alphabet. Define the **Theory Space**:
$$\mathfrak{T} := \{ T \subset \Sigma^* : T \text{ is recursively enumerable} \}$$
:::

Each theory $T \in \mathfrak{T}$ corresponds to a Turing machine $M_T$ with Gödel number $\lceil M_T \rceil \in \mathbb{N}$.

:::{prf:definition} Kolmogorov Complexity
:label: def-kolmogorov-complexity-2

The **Kolmogorov complexity** {cite}`Kolmogorov65` of a string $x \in \Sigma^*$ relative to a universal Turing machine $U$ is:
$$K_U(x) := \min \{ |p| : U(p) = x \}$$
where $|p|$ denotes the length of program $p$. By the invariance theorem {cite}`LiVitanyi08`, for any two universal machines $U_1, U_2$:
$$|K_{U_1}(x) - K_{U_2}(x)| \leq c_{U_1, U_2}$$
for a constant $c$ independent of $x$. We write $K(x)$ for the complexity relative to a fixed reference machine.
:::

:::{prf:definition} Algorithmic Probability
:label: def-algorithmic-probability

The **algorithmic probability** {cite}`Solomonoff64` and {cite}`Levin73` of a string $x$ is:
$$m(x) := \sum_{p: U(p) = x} 2^{-|p|}$$
This satisfies $m(x) = 2^{-K(x) + O(1)}$ and defines a universal semi-measure on $\Sigma^*$.
:::

:::{prf:definition} Theory Height Functional
:label: def-theory-height-functional

For a theory $T \in \mathfrak{T}$ and observable dataset $\mathcal{D}_{\text{obs}} = (d_1, d_2, \ldots, d_n)$, define the **Height Functional**:
$$\Phi(T) := K(T) + L(T, \mathcal{D}_{\text{obs}})$$
where:

1. $K(T) := K(\lceil M_T \rceil)$ is the Kolmogorov complexity of the theory's encoding
2. $L(T, \mathcal{D}_{\text{obs}}) := -\log_2 P(\mathcal{D}_{\text{obs}} \mid T)$ is the **codelength** of the data given the theory

:::

This is the **Minimum Description Length (MDL)** principle {cite}`Rissanen78` and {cite}`Grunwald07`:
$$\Phi(T) = K(T) - \log_2 P(\mathcal{D}_{\text{obs}} \mid T)$$

:::{prf:proposition} MDL as Two-Part Code
:label: prop-mdl-as-two-part-code

*The height functional $\Phi(T)$ equals the length of the optimal two-part code for the dataset:*
$$\Phi(T) = |T| + |\mathcal{D}_{\text{obs}} : T|$$
*where $|T|$ is the description length of the theory and $|\mathcal{D}_{\text{obs}} : T|$ is the description length of the data given the theory.*
:::

:::{prf:proof}
By the definition of conditional Kolmogorov complexity {cite}`LiVitanyi08` (Theorem 3.9.1):
$$K(\mathcal{D}_{\text{obs}} \mid T) = -\log_2 P(\mathcal{D}_{\text{obs}} \mid T) + O(\log n)$$
where $n = |\mathcal{D}_{\text{obs}}|$. The two-part code concatenates $\lceil M_T \rceil$ with the conditional encoding.
:::

#### The Information Distance

:::{prf:definition} Information Distance
:label: def-information-distance

The **normalized information distance** {cite}`LiVitanyi08` and {cite}`Bennett98` between theories $T_1, T_2 \in \mathfrak{T}$ is:
$$d_{\text{NID}}(T_1, T_2) := \frac{\max\{K(T_1 \mid T_2), K(T_2 \mid T_1)\}}{\max\{K(T_1), K(T_2)\}}$$
:::

The unnormalized **information distance** is:
$$d_{\text{info}}(T_1, T_2) := K(T_1 \mid T_2) + K(T_2 \mid T_1)$$

:::{prf:theorem} Metric Properties
:label: thm-metric-properties

*The normalized information distance $d_{\text{NID}}$ is a metric on the quotient space $\mathfrak{T}/{\sim}$ where $T_1 \sim T_2$ iff $K(T_1 \Delta T_2) = O(1)$. Specifically:*

1. *Symmetry: $d_{\text{NID}}(T_1, T_2) = d_{\text{NID}}(T_2, T_1)$*
2. *Identity: $d_{\text{NID}}(T_1, T_2) = 0$ iff $T_1 \sim T_2$*
3. *Triangle inequality: $d_{\text{NID}}(T_1, T_3) \leq d_{\text{NID}}(T_1, T_2) + d_{\text{NID}}(T_2, T_3) + O(1/K)$*
:::

:::{prf:proof}
**Step 1 (Symmetry).** Immediate from the definition using $\max$.

**Step 2 (Identity).** If $d_{\text{NID}}(T_1, T_2) = 0$, then $K(T_1 \mid T_2) = K(T_2 \mid T_1) = 0$. By the symmetry of information {cite}`LiVitanyi08` (Theorem 3.9.1):
$$K(T_1, T_2) = K(T_1) + K(T_2 \mid T_1) + O(\log K) = K(T_2) + K(T_1 \mid T_2) + O(\log K)$$
Thus $K(T_1) = K(T_2) + O(\log K)$ and $T_1, T_2$ are algorithmically equivalent.

**Step 3 (Triangle Inequality).** By the chain rule for conditional complexity:
$$K(T_1 \mid T_3) \leq K(T_1 \mid T_2) + K(T_2 \mid T_3) + O(\log K)$$
Dividing by $\max\{K(T_1), K(T_3)\}$ and using monotonicity yields the result.
:::

:::{prf:corollary}
:label: cor-unnamed-6

*The theory space $(\mathfrak{T}/{\sim}, d_{\text{NID}})$ is a complete metric space.*
:::

---

### The Epistemic Fixed Point

> **[Deps] Structural Dependencies**
>
> *   **Prerequisites (Inputs):**
>     *   [ ] **Axiom C:** Compactness (bounded energy implies profile convergence)
>     *   [ ] **Axiom D:** Dissipation (energy-dissipation inequality)
>     *   [ ] **Axiom Rep:** Dictionary/Correspondence (structural translation)
>
> *   **Output (Structural Guarantee):**
>     *   Self-referential knowledge has fixed-point structure
>
> *   **Failure Condition (Debug):**
>     *   If **Axiom C** fails $\to$ **Mode D.D** (Dispersion/Global existence)
>     *   If **Axiom D** fails $\to$ **Mode C.E** (Energy blow-up)


#### Statement

:::{prf:metatheorem} Epistemic Fixed Point
:label: mt-epistemic-fixed-point

Let $\mathcal{A}$ be an optimal Bayesian learning agent operating on the theory space $\mathfrak{T}$, with prior $\pi_0(T) = 2^{-K(T)}$ (the universal prior). Let $\rho_t$ be the posterior distribution over theories after observing data $\mathcal{D}_t = (d_1, \ldots, d_t)$. Assume:

1. **Realizability:** There exists $T^* \in \mathfrak{T}$ such that $\mathcal{D}_t \sim P(\cdot \mid T^*)$.
2. **Consistency:** The true theory $T^*$ satisfies $K(T^*) < \infty$.

Then as $t \to \infty$:
$$\rho_t \xrightarrow{w} \delta_{[T^*]}$$
where $[T^*]$ is the equivalence class of theories with $d_{\text{NID}}(T, T^*) = 0$.

Moreover, if the true data-generating process is a Hypostructure $\mathbb{H}$ acting on physical observables, then:
$$[T^*] = [\mathbb{H}]$$
:::

#### Full Proof

:::{prf:proof}

**Step 1 (Bayesian Update).** By Bayes' theorem, the posterior after observing $\mathcal{D}_t$ is:
$$\rho_t(T) = \frac{P(\mathcal{D}_t \mid T) \cdot \pi_0(T)}{\sum_{T' \in \mathfrak{T}} P(\mathcal{D}_t \mid T') \cdot \pi_0(T')}$$

With the universal prior $\pi_0(T) = 2^{-K(T)}$:
$$\rho_t(T) \propto P(\mathcal{D}_t \mid T) \cdot 2^{-K(T)} = 2^{-\Phi(T)}$$

where $\Phi(T) = K(T) - \log_2 P(\mathcal{D}_t \mid T)$ is the height functional.

**Step 2 (Solomonoff Convergence).** By the Solomonoff convergence theorem {cite}`Solomonoff78` and {cite}`Hutter05`:

*For any computable probability measure $\mu$ on sequences, the Solomonoff predictor $M$ satisfies:*
$$\sum_{t=1}^{\infty} \mathbb{E}_\mu \left[ \left( M(d_t \mid d_1, \ldots, d_{t-1}) - \mu(d_t \mid d_1, \ldots, d_{t-1}) \right)^2 \right] \leq K(\mu) \ln 2$$

This implies that the posterior concentrates on theories that predict as well as the true theory.

**Step 3 (MDL Consistency).** By the MDL consistency theorem {cite}`Barron98` and {cite}`Grunwald07`:

*If the true distribution $P^*$ lies in the model class $\mathcal{M}$, then the MDL estimator:*
$$\hat{T}_n = \arg\min_{T \in \mathcal{M}} \Phi_n(T)$$
*satisfies $d_{\text{KL}}(P^* \| P_{\hat{T}_n}) \to 0$ almost surely.*

Applied to our setting: if $T^*$ generates the data, then:
$$\lim_{t \to \infty} \rho_t(B_\epsilon(T^*)) = 1$$
for any $\epsilon > 0$, where $B_\epsilon(T^*) = \{T : d_{\text{NID}}(T, T^*) < \epsilon\}$.

**Step 4 (Rate of Convergence).** The posterior probability of the true theory satisfies {cite}`LiVitanyi08` (Section 5.5):
$$\rho_t(T^*) \geq 2^{-K(T^*)} \cdot \frac{P(\mathcal{D}_t \mid T^*)}{m(\mathcal{D}_t)}$$

where $m(\mathcal{D}_t)$ is the universal mixture. Since $m(\mathcal{D}_t) \leq 1$:
$$\rho_t(T^*) \geq 2^{-K(T^*)} \cdot P(\mathcal{D}_t \mid T^*)$$

For competing theories $T \neq T^*$:
$$\frac{\rho_t(T)}{\rho_t(T^*)} = 2^{-(K(T) - K(T^*))} \cdot \frac{P(\mathcal{D}_t \mid T)}{P(\mathcal{D}_t \mid T^*)}$$

If $T$ makes systematically worse predictions (lower likelihood), this ratio decays exponentially in $t$.

**Step 5 (Reflective Consistency via Lawvere Fixed Point).** The agent $\mathcal{A}$ performing Bayesian inference is itself a physical system. By the Church-Turing thesis, $\mathcal{A}$ is computable, hence describable by some theory $T_{\mathcal{A}} \in \mathfrak{T}$.

If the Hypostructure $\mathbb{H}$ is the true theory, it must describe all physical systems including $\mathcal{A}$. Thus:
$$T_{\mathcal{A}} \prec T_{\text{hypo}}$$
where $\prec$ denotes "is a subsystem of."

By **Lawvere's Fixed Point Theorem** {cite}`Lawvere69`: In any cartesian closed category $\mathcal{C}$ with a point-surjective morphism $\phi: A \to B^A$, every endomorphism $f: B \to B$ has a fixed point.

Applied to our setting:
- $\mathcal{C}$ = category of computable functions
- $A$ = space of theories $\mathfrak{T}$
- $B$ = space of physical systems
- $\phi$ = the map taking a theory to its physical implementation
- $f$ = the "theorize about" operation

The fixed point condition becomes:
$$\exists T^* \in \mathfrak{T}: T^* = f(\phi(T^*))$$

This is precisely the statement that the Hypostructure describes itself.
:::

**Emergence Class:** Scientific Theory

**Input Substrate:** Bayesian Learning Agent $\mathcal{A}$ + Theory Space $\mathfrak{T}$ + Universal Prior

**Generative Mechanism:** Solomonoff Induction — MDL convergence to simplest consistent theory

**Output Structure:** The Hypostructure $\mathbb{H}$ as unique fixed point of inference

:::{prf:corollary} Inevitability of Discovery
:label: cor-inevitability-of-discovery

*Any sufficiently powerful learning agent will eventually converge to the Hypostructure (or an equivalent formulation) as its best theory of reality.*
:::


### Logical Foundations and Gödelian Considerations

#### Relation to Incompleteness

:::{prf:theorem} Consistency
:label: thm-consistency

*The Hypostructure axiom system $\mathcal{A}_{\text{core}} = \{C, D, SC, LS, Cap, TB, R\}$ is consistent.*
:::

:::{prf:proof}
We exhibit a model. Take:

- $X = L^2(\mathbb{R}^3)$ (square-integrable functions)
- $S_t$ = heat semigroup $e^{t\Delta}$
- $\Phi(u) = \int |\nabla u|^2 dx$ (Dirichlet energy)
- $\mathfrak{D}(u) = \|u_t\|^2$ (dissipation rate)


This satisfies all axioms:

- **C:** Sublevel sets $\{\Phi \leq c\}$ are weakly compact in $L^2$
- **D:** $\frac{d\Phi}{dt} = -2\mathfrak{D} \leq 0$ along the heat flow
- **SC:** $\Phi(\lambda u) = \lambda^2 \Phi(u)$ (2-homogeneous)
- **LS:** Standard gradient estimate near critical points
- **Cap, TB:** Follow from Sobolev embedding


By Gödel's completeness theorem, existence of a model implies consistency.
:::

:::{prf:theorem} Incompleteness Avoidance
:label: thm-incompleteness-avoidance

*The Hypostructure framework avoids Gödelian incompleteness by being a physical theory rather than a foundational mathematical system.*
:::

:::{prf:proof}
Gödel's incompleteness theorems {cite}`Godel31` apply to:

1. Formal systems containing arithmetic
2. That are recursively axiomatizable
3. And claim to capture all mathematical truth


The Hypostructure:

- Is a physical theory making empirical predictions
- Does not claim to axiomatize all of mathematics
- Is "complete" only relative to the phenomena it models


The distinction is analogous to the difference between "ZFC is incomplete" and "Newtonian mechanics is complete for classical phenomena."

More precisely: let $\text{Th}(\mathbb{H})$ be the set of sentences true in the Hypostructure. This is not recursively enumerable (by Tarski's undefinability theorem). However, the *axioms* $\mathcal{A}_{\text{core}}$ are finite and decidable. The metatheorems are derived from these axioms plus standard mathematics (analysis, topology, etc.).

The framework is **relatively complete**: every physical phenomenon derivable from the axioms is captured by some metatheorem.
:::

#### Self-Reference and Löb's Theorem

:::{prf:theorem} Self-Reference via Löb
:label: thm-self-reference-via-lb

*The Hypostructure can consistently assert its own correctness.*
:::

:::{prf:proof}
By **Löb's Theorem** {cite}`Loeb55`: For any formal system $T$ containing arithmetic,
$$T \vdash \Box(\Box P \to P) \to \Box P$$
where $\Box P$ means "$T$ proves $P$."

This implies: if $T$ proves "if $T$ proves $P$ then $P$", then $T$ proves $P$.

For the Hypostructure, let $P$ = "The Hypostructure correctly describes physics."

**Claim:** $\mathbb{H} \vdash \Box P \to P$.

*Justification:* If the Hypostructure proves its own correctness (i.e., derives the metatheorems), then by the adjunction $R \circ M \cong \text{Id}$, the physical implementation confirms this correctness through observation.

By Löb's theorem: $\mathbb{H} \vdash \Box P$, i.e., the Hypostructure proves its own correctness.

This is not a contradiction because the "proof" is empirical (via $R$) rather than purely syntactic.
:::
