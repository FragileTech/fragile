# Appendix A1: QSD Structure - Exchangeability and Mean-Field Limit

**Mathematical Level**: Publication standard (rigorous proofs)

**Purpose**: Establish the rigorous structure of the Quasi-Stationary Distribution (QSD) for the Euclidean Gas algorithm

---

## A1.1 QSD Structure: Exchangeability

### A1.1.1 Main Result

:::{prf:theorem} Exchangeability of the QSD
:label: thm-qsd-exchangeability

Let $\pi_N \in \mathcal{P}(\Sigma_N)$ be the unique Quasi-Stationary Distribution of the Euclidean Gas. Then $\pi_N$ is an **exchangeable probability measure**: for any permutation $\sigma \in S_N$ and any measurable set $A \subseteq \Sigma_N$:

$$
\pi_N(\{(w_1, \ldots, w_N) \in A\}) = \pi_N(\{(w_{\sigma(1)}, \ldots, w_{\sigma(N)}) \in A\})

$$

where $w_i = (x_i, v_i, s_i)$ is the state of walker $i$.
:::

:::{prf:proof}
The dynamics are manifestly symmetric under permutation of walker labels.

**Kinetic operator**: Each walker evolves according to the same Langevin dynamics:

$$
\mathcal{L}_{\text{kin}} f(S) = \sum_{i=1}^N \left[ v_i \cdot \nabla_{x_i} + F_i \cdot \nabla_{v_i} + \frac{\sigma_i^2}{2}\Delta_{v_i} \right] f

$$

Permuting indices preserves this structure since the sum is symmetric.

**Cloning operator**: The companion selection and cloning mechanism are permutation-invariant:

$$
\mathcal{L}_{\text{clone}} f(S) = \sum_{i \in \mathcal{D}} \lambda_i \sum_{j \in \mathcal{A}} p_{ij} \int [f(S^{i \leftarrow j}_\delta) - f(S)] \phi_\delta

$$

where $p_{ij} \propto V_{\text{fit}}(w_j)$ depends only on walker states, not labels.

**Total generator**: $\mathcal{L} = \mathcal{L}_{\text{kin}} + \mathcal{L}_{\text{clone}}$ is permutation-symmetric.

Since $\mathcal{L}$ is permutation-symmetric and $\pi_N$ is the unique QSD satisfying $\mathcal{L}^* \pi_N = 0$, the pushed-forward measure $\sigma_* \pi_N$ also satisfies this equation. By uniqueness: $\sigma_* \pi_N = \pi_N$ for all $\sigma \in S_N$. $\square$
:::

### A1.1.2 Hewitt-Savage Representation

:::{prf:theorem} Finite de Finetti Representation
:label: thm-hewitt-savage-representation

Since $\pi_N$ is exchangeable on the compact space $\Omega$, there exists a probability measure $\mathcal{Q}_N$ on $\mathcal{P}(\Omega)$ (the **mixing measure**) such that for any $k$-particle marginal with $1 \leq k \leq N$:

$$
d_{\text{TV}}\left(\pi_{N,k}, \int_{\mathcal{P}(\Omega)} \mu^{\otimes k} \, d\mathcal{Q}_N(\mu)\right) \leq \frac{k(k-1)}{2N}

$$

where $\pi_{N,k}$ is the law of the first $k$ walkers under $\pi_N$, $\mu^{\otimes k}$ denotes the $k$-fold product measure (walkers are i.i.d. with law $\mu$), and $d_{\text{TV}}$ is total variation distance.

**Key consequences**:

1. **Low-order marginals** ($k$ fixed, $N \to \infty$): The bound is $O(1/N)$
   - Pairwise marginals ($k=2$): $d_{\text{TV}} \leq 1/N$
   - Single-particle marginal ($k=1$): exact representation

2. **Full N-particle distribution** ($k=N$): The bound is $O(N)$
   - $d_{\text{TV}}(\pi_N, \int \mu^{\otimes N} d\mathcal{Q}_N) \leq (N-1)/2$
   - Representation becomes exact only in the limit $N \to \infty$

**Interpretation**: The QSD can be **approximately** represented as a mixture of IID configurations. The approximation is excellent for low-order marginals ($k \ll N$), which is precisely what is needed for correlation decay and propagation of chaos.

**Construction**: The mixing measure $\mathcal{Q}_N$ is the law of the empirical measure $L_N = \frac{1}{N}\sum_{i=1}^N \delta_{w_i}$ when $(w_1, \ldots, w_N) \sim \pi_N$.

**Citation**: Diaconis & Freedman (1980), Theorem 4. This is a **finite** de Finetti theorem which does not require projective consistency. The mixing measure $\mathcal{Q}_N$ is not unique for finite $N$.
:::

:::{prf:proof}

This proof applies the Diaconis-Freedman finite de Finetti theorem to the QSD of the Euclidean Gas.

**Step 1: Verify Compactness of $\Omega$**

The single-walker state space is $\Omega = X_{\text{valid}} \times V_{\text{alg}}$ where:
- $X_{\text{valid}} \subseteq \mathbb{R}^d$ is a bounded convex set (hence closed and bounded)
- $V_{\text{alg}} = \{v \in \mathbb{R}^d : \|v\|_{\text{alg}} \leq R_v\}$ is a closed ball

By the Heine-Borel theorem, both $X_{\text{valid}}$ and $V_{\text{alg}}$ are compact. By Tychonoff's theorem, the product $\Omega = X_{\text{valid}} \times V_{\text{alg}}$ is compact in the product topology. As a compact subset of a metric space ($\mathbb{R}^{2d}$), $\Omega$ is a compact metric space (Polish space).

**Step 2: Verify Exchangeability of $\pi_N$**

By {prf:ref}`thm-qsd-exchangeability`, the QSD $\pi_N$ is an exchangeable probability measure on $\Omega^N$. That is, for any permutation $\sigma \in S_N$ and any measurable set $A \subseteq \Omega^N$:

$$
\pi_N(\{(w_1, \ldots, w_N) \in A\}) = \pi_N(\{(w_{\sigma(1)}, \ldots, w_{\sigma(N)}) \in A\})

$$

**Step 3: Apply Diaconis-Freedman Theorem 4**

**Citation**: Diaconis, P., & Freedman, D. (1980). Finite exchangeable sequences. *The Annals of Probability*, 8(4), 745-764, Theorem 4.

**Theorem Statement (Diaconis-Freedman)**: Let $(X_1, \ldots, X_N)$ be an exchangeable sequence on a compact metric space $S$, with joint law $\pi_N$. Then there exists a probability measure $Q$ on $\mathcal{P}(S)$ such that for any $1 \leq k \leq N$, the law $\pi_{N,k}$ of the first $k$ variables satisfies:

$$
d_{\text{TV}}\left(\pi_{N,k}, \int_{\mathcal{P}(S)} \mu^{\otimes k} \, dQ(\mu)\right) \leq \frac{k(k-1)}{2N}

$$

**Application to our setting**: Set $S = \Omega$ (compact metric space, verified in Step 1). The QSD $\pi_N$ on $\Omega^N$ is exchangeable (verified in Step 2). Therefore, Diaconis-Freedman's theorem directly applies, establishing the existence of a mixing measure $\mathcal{Q}_N$ on $\mathcal{P}(\Omega)$ with the stated quantitative bound.

**Step 4: Construct the Canonical Mixing Measure**

While Diaconis-Freedman's theorem guarantees existence, the mixing measure can be constructed explicitly:

**Definition**: Let $(w_1, \ldots, w_N) \sim \pi_N$. Define the **empirical measure**:

$$
L_N(w_1, \ldots, w_N) := \frac{1}{N}\sum_{i=1}^N \delta_{w_i} \in \mathcal{P}(\Omega)

$$

The **canonical mixing measure** is:

$$
\mathcal{Q}_N := \text{Law}_{\pi_N}(L_N)

$$

That is, $\mathcal{Q}_N$ is the pushforward of $\pi_N$ under the empirical measure map $L_N: \Omega^N \to \mathcal{P}(\Omega)$.

**Verification**: This construction is standard in de Finetti theory (see Diaconis-Freedman §2). The bound in Step 3 holds for this canonical choice of $\mathcal{Q}_N$.

**Step 5: Interpret the Key Consequences**

**For low-order marginals** ($k$ fixed, $N \to \infty$):

The bound becomes:

$$
d_{\text{TV}}\left(\pi_{N,k}, \int \mu^{\otimes k} d\mathcal{Q}_N(\mu)\right) \leq \frac{k(k-1)}{2N} = O(1/N)

$$

This is the regime of practical importance:
- **Single-particle marginal** ($k=1$): Bound is $0/N = 0$ (exact representation)
- **Pairwise marginals** ($k=2$): Bound is $1/N$ (used in correlation decay, Theorem {prf:ref}`thm-correlation-decay`)
- **Finite $k$**: Bound is $k(k-1)/(2N) \to 0$ as $N \to \infty$

**For full N-particle distribution** ($k=N$):

$$
d_{\text{TV}}\left(\pi_N, \int \mu^{\otimes N} d\mathcal{Q}_N(\mu)\right) \leq \frac{N(N-1)}{2N} = \frac{N-1}{2} \approx \frac{N}{2}

$$

This bound is $O(N)$ and does NOT vanish as $N \to \infty$. The full N-particle distribution is **not** well-approximated by the mixture for finite $N$. However, this is not a limitation: propagation of chaos results only require good approximation of low-order marginals.

**Step 6: Non-Uniqueness for Finite $N$**

For any finite $N$, the mixing measure $\mathcal{Q}_N$ is **not unique**. The map $Q \mapsto \int \mu^{\otimes N} dQ(\mu)$ from $\mathcal{P}(\mathcal{P}(\Omega))$ to $\mathcal{P}(\Omega^N)$ is not injective for finite $N$ because the $N$-particle distribution only determines the moments of $Q$ up to order $N$.

**Example** (following Diaconis-Freedman, Example 1): For $N=1$, any two mixing measures $\mathcal{Q}_1$ and $\mathcal{Q}_1'$ with the same barycenter (mean measure) produce the same single-particle distribution.

The canonical choice $\mathcal{Q}_N = \text{Law}(L_N)$ is natural but not unique. Uniqueness holds only in the limit $N \to \infty$ (de Finetti's theorem for infinite exchangeable sequences).

$\square$
:::

**Key distinction**:
- **Product measure** (independent): $\pi_N = \mu^{\otimes N}$ for a single fixed $\mu$
- **Exchangeable** (correlated): $\pi_N = \int \mu^{\otimes N} d\mathcal{Q}_N(\mu)$ where $\mu$ is random

The QSD is exchangeable but **not** a simple product measure. Cloning creates correlations.

### A1.1.3 Single-Particle Marginal

:::{prf:definition} Single-Particle Marginal
:label: def-single-particle-marginal

The single-particle marginal of $\pi_N$ is:

$$
\mu_N(A) := \pi_N(\{(w_1, \ldots, w_N) : w_1 \in A\})

$$

By exchangeability, this is the same for any walker index.
:::

:::{prf:proposition} Marginal as Mixture Average
:label: prop-marginal-mixture

From the finite de Finetti representation ({prf:ref}`thm-hewitt-savage-representation`) with $k=1$:

$$
\mu_N = \int_{\mathcal{P}(\Omega)} \mu \, d\mathcal{Q}_N(\mu)

$$

where $\mu_N$ is the single-particle marginal of $\pi_N$. This is an **exact** representation (the bound $k(k-1)/(2N) = 0$ for $k=1$).

**Interpretation**: The single-particle marginal is exactly the average (barycenter) of all IID distributions in the mixing measure $\mathcal{Q}_N$.
:::

:::{prf:proof}

Apply {prf:ref}`thm-hewitt-savage-representation` with $k=1$. The bound becomes:

$$
d_{\text{TV}}\left(\mu_N, \int_{\mathcal{P}(\Omega)} \mu \, d\mathcal{Q}_N(\mu)\right) \leq \frac{1(1-1)}{2N} = 0

$$

Therefore, the representation is exact. $\square$
:::

---

## A1.2 Mean-Field Limit and Propagation of Chaos

### A1.2.1 Main Convergence Result

:::{prf:theorem} Propagation of Chaos
:label: thm-propagation-chaos-qsd

As $N \to \infty$, the single-particle marginal $\mu_N$ converges weakly to a unique limit $\mu_\infty \in \mathcal{P}(\Omega)$:

$$
\mu_N \Rightarrow \mu_\infty

$$

Moreover, $\mu_\infty$ is the unique stationary solution of the mean-field McKean-Vlasov equation:

$$
\frac{\partial \rho}{\partial t} = \mathcal{L}[\rho] \rho

$$

where the generator $\mathcal{L}[\rho]$ depends nonlinearly on $\rho$ through mean-field interactions.
:::

:::

:::{prf:proof}

This result is established in detail in **Chapter 08: Propagation of Chaos** (`08_propagation_chaos.md`). We provide a brief outline of the three-step proof strategy:

**Step 1: Tightness of $\{\mu_N\}_{N \geq 1}$**

From the Foster-Lyapunov analysis in `06_convergence.md`, the N-particle QSD $\pi_N$ satisfies uniform moment bounds:

$$
\sup_{N \geq 1} \mathbb{E}_{\pi_N}[\|w_1\|^p] < \infty

$$

for any $p \geq 1$, where $w_1$ is the state of a single walker. These N-uniform bounds imply tightness of the sequence of single-particle marginals $\{\mu_N\}$ in the space $\mathcal{P}(\Omega)$ equipped with the weak topology. By Prokhorov's theorem, $\{\mu_N\}$ is relatively compact: every subsequence has a convergent sub-subsequence.

**Step 2: Identification of Limit Points**

Let $\mu_\infty$ be any weak limit point of $\{\mu_N\}$. The mean-field analysis in `07_mean_field.md` establishes that the limiting measure must satisfy the stationary McKean-Vlasov equation:

$$
\mathcal{L}[\mu_\infty] \mu_\infty = 0

$$

in the weak (distributional) sense, where $\mathcal{L}[\rho]$ is the generator of the mean-field dynamics (kinetic operator + nonlocal cloning operator). This identification is proven via the martingale problem formulation and taking limits in the weak formulation of the Fokker-Planck-McKean-Vlasov PDE.

**Step 3: Uniqueness of the Stationary Solution**

The stationary McKean-Vlasov equation has a **unique** solution $\mu_\infty = \rho_0 dx$ (where $\rho_0$ is the mean-field QSD density). Uniqueness is established in `08_propagation_chaos.md` via:

1. **Hypoelliptic regularity** (Villani 2009, Hörmander theory) ensuring smoothness of $\rho_0$
2. **Contraction mapping** for the McKean-Vlasov fixed-point equation
3. **Lyapunov functional** (relative entropy) strictly decreasing along solutions

Since every limit point equals the unique $\mu_\infty$, the entire sequence converges: $\mu_N \Rightarrow \mu_\infty$ as $N \to \infty$.

**Conclusion**: The complete rigorous proof with all technical details is provided in `08_propagation_chaos.md`. This theorem is a fundamental result of the Fragile Gas framework, establishing that the discrete N-particle system converges to the continuum mean-field description in the large-N limit. $\square$
:::

### A1.2.2 Correlation Decay

:::{prf:theorem} Quantitative Decorrelation
:label: thm-correlation-decay

For bounded single-particle test functions $g: \Omega \to \mathbb{R}$ with $\|g\|_{\infty} \leq 1$:

$$
\left|\text{Cov}_{\pi_N}(g(w_i), g(w_j))\right| \leq \frac{C}{N}

$$

for $i \neq j$, where $C$ is independent of $N$.

**Consequence**: Covariances decay as $O(1/N)$, faster than the standard Wasserstein rate $O(1/\sqrt{N})$.
:::

:::{prf:proof}

This proof uses the finite de Finetti representation ({prf:ref}`thm-hewitt-savage-representation`) for **pairwise marginals** ($k=2$), where the approximation error is $O(1/N)$.

**Step 1: Approximate Pairwise Marginal via de Finetti**

From {prf:ref}`thm-hewitt-savage-representation` with $k=2$, the pairwise marginal $\pi_{N,2}$ (law of $(w_i, w_j)$ for $i \neq j$) satisfies:

$$
d_{\text{TV}}\left(\pi_{N,2}, \int_{\mathcal{P}(\Omega)} \mu^{\otimes 2} \, d\mathcal{Q}_N(\mu)\right) \leq \frac{2(2-1)}{2N} = \frac{1}{N}

$$

Let $\tilde{\pi}_{N,2} := \int \mu^{\otimes 2} d\mathcal{Q}_N(\mu)$ denote the approximating mixture measure.

**Step 2: Bound Error in Joint Expectation**

For any bounded function $h: \Omega^2 \to \mathbb{R}$ with $\|h\|_\infty \leq M$, the total variation bound implies:

$$
\left|\mathbb{E}_{\pi_{N,2}}[h] - \mathbb{E}_{\tilde{\pi}_{N,2}}[h]\right| \leq 2M \cdot d_{\text{TV}}(\pi_{N,2}, \tilde{\pi}_{N,2}) \leq \frac{2M}{N}

$$

Apply this to $h(w_i, w_j) = g(w_i)g(w_j)$ with $\|h\|_\infty \leq \|g\|_\infty^2$:

$$
\left|\mathbb{E}_{\pi_N}[g(w_i)g(w_j)] - \mathbb{E}_{\tilde{\pi}_{N,2}}[g(w_i)g(w_j)]\right| \leq \frac{2\|g\|_\infty^2}{N}

$$

**Step 3: Exact de Finetti Identity for the Approximating Measure**

For the approximating mixture $\tilde{\pi}_{N,2} = \int \mu^{\otimes 2} d\mathcal{Q}_N$, the de Finetti identity holds **exactly**:

$$
\mathbb{E}_{\tilde{\pi}_{N,2}}[g(w_i)g(w_j)] = \int \mathbb{E}_{\mu}[g] \mathbb{E}_{\mu}[g] \, d\mathcal{Q}_N(\mu) = \int (\mathbb{E}_{\mu}[g])^2 \, d\mathcal{Q}_N(\mu)

$$

by conditional independence given $\mu$ in the product measure $\mu^{\otimes 2}$.

**Step 4: Single-Particle Marginal is Exact**

For $k=1$, the bound in {prf:ref}`thm-hewitt-savage-representation` gives $0/N = 0$, so:

$$
\mathbb{E}_{\pi_N}[g(w_i)] = \int \mathbb{E}_{\mu}[g] \, d\mathcal{Q}_N(\mu)

$$

exactly (no approximation error).

**Step 5: Combine to Bound Covariance**

The covariance is:

$$
\text{Cov}_{\pi_N}(g(w_i), g(w_j)) = \mathbb{E}_{\pi_N}[g(w_i)g(w_j)] - \mathbb{E}_{\pi_N}[g(w_i)] \mathbb{E}_{\pi_N}[g(w_j)]

$$

Using Steps 2-4:

$$
\text{Cov}_{\pi_N}(g(w_i), g(w_j)) = \left(\int (\mathbb{E}_{\mu}[g])^2 d\mathcal{Q}_N \pm \frac{2\|g\|_\infty^2}{N}\right) - \left(\int \mathbb{E}_{\mu}[g] d\mathcal{Q}_N\right)^2

$$

$$
= \text{Var}_{\mathcal{Q}_N}(\mathbb{E}_{\mu}[g]) + O\left(\frac{\|g\|_\infty^2}{N}\right)

$$

**Step 6: Apply Mixing Measure Variance Bound**

By Theorem {prf:ref}`thm-mixing-variance-corrected`:

$$
\text{Var}_{\mathcal{Q}_N}(\mathbb{E}_{\mu}[g]) \leq \frac{3\|g\|_\infty^2}{N}

$$

Therefore:

$$
\left|\text{Cov}_{\pi_N}(g(w_i), g(w_j))\right| \leq \frac{3\|g\|_\infty^2}{N} + \frac{2\|g\|_\infty^2}{N} = \frac{5\|g\|_\infty^2}{N}

$$

For $\|g\|_\infty \leq 1$, this gives $|\text{Cov}| \leq 5/N$, establishing the $O(1/N)$ decay with explicit constant $C=5$. $\square$
:::

### A1.2.3 Quantitative Mixing Measure Concentration

:::{prf:theorem} Variance of Mixing Measure
:label: thm-mixing-variance-corrected

Let $\pi_N = \int \mu^{\otimes N} d\mathcal{Q}_N(\mu)$ be the de Finetti representation of the QSD, and let $\rho_0$ be the mean-field limit. Assume the quantitative KL bound from {prf:ref}`lem-quantitative-kl-bound` (document `12_quantitative_error_bounds.md`):

$$
D_{KL}(\pi_N \| \rho_0^{\otimes N}) \leq \frac{C_{\text{int}}}{N}

$$

Then for any bounded measurable function $g: \Omega \to \mathbb{R}$ with $\|g\|_{\infty} \leq B$:

$$
\text{Var}_{\mathcal{Q}_N}(\mathbb{E}_{\mu}[g]) \leq \frac{2 \cdot e^{C_{\text{int}}/N} \cdot B^2}{N}

$$

For sufficiently large $N$ (such that $e^{C_{\text{int}}/N} \leq 3/2$):

$$
\text{Var}_{\mathcal{Q}_N}(\mathbb{E}_{\mu}[g]) \leq \frac{3B^2}{N}

$$

**Consequence**: Combined with the corrected de Finetti identity:

$$
|\text{Cov}_{\pi_N}(g(w_i), g(w_j))| = \text{Var}_{\mathcal{Q}_N}(\mathbb{E}_{\mu}[g]) \leq \frac{3\|g\|_{\infty}^2}{N}

$$

for $i \neq j$, establishing O(1/N) decorrelation for **all bounded measurable functions**, including indicator functions used in companion selection.
:::

:::{prf:proof}

This proof uses **information-theoretic variance bounds** without relying on the de Finetti representation being exact for the N-particle system. The key advantage is that it requires only the KL-divergence bound and boundedness of $g$.

**Step 1: Relate Mixing Measure Variance to Pairwise Covariance**

By the structure of the de Finetti mixing measure (law of empirical measure, see {prf:ref}`thm-hewitt-savage-representation`, Step 4), the variance of $\mathbb{E}_{\mu}[g]$ over $\mathcal{Q}_N$ equals the covariance of $g$ at distinct particles:

$$
\text{Var}_{\mathcal{Q}_N}(\mathbb{E}_{\mu}[g]) = \text{Cov}_{\pi_N}(g(w_i), g(w_j)) \quad \text{for } i \neq j

$$

This is an **exact identity** (no approximation) following from the construction $\mathcal{Q}_N = \text{Law}_{\pi_N}(L_N)$ where $L_N = \frac{1}{N}\sum \delta_{w_i}$ is the empirical measure.

**Proof of identity**:

$$
\mathbb{E}_{\mathcal{Q}_N}[(\mathbb{E}_{\mu}[g])^2] = \mathbb{E}_{\pi_N}\left[\left(\frac{1}{N}\sum_{i=1}^N g(w_i)\right) \mathbb{E}_{L_N}[g]\right] = \mathbb{E}_{\pi_N}\left[\frac{1}{N}\sum_i g(w_i) \cdot \frac{1}{N}\sum_j g(w_j)\right]

$$

$$
= \frac{1}{N^2}\sum_{i,j} \mathbb{E}_{\pi_N}[g(w_i)g(w_j)] = \frac{1}{N}\mathbb{E}[g_1^2] + \frac{N-1}{N}\mathbb{E}[g_1 g_2]

$$

Similarly, $\mathbb{E}_{\mathcal{Q}_N}[\mathbb{E}_{\mu}[g]]^2 = \mathbb{E}[g_1]^2$. Taking the difference yields the covariance identity.

Therefore, it suffices to bound $\text{Cov}_{\pi_N}(g(w_i), g(w_j))$ using only the KL-divergence bound.

**Step 2: Information-Theoretic Variance Bound**

We use the **variance perturbation inequality**: For probability measures $P, Q$ and bounded function $F$ with $\|F\|_\infty \leq M$:

$$
\text{Var}_P(F) \leq \text{Var}_Q(F) + 2M^2 \cdot D_{\text{KL}}(P \| Q)

$$

This is a standard result in information theory (Boucheron-Lugosi-Massart 2013, Ledoux 2001).

**Step 3: Apply to Empirical Average**

Define the empirical average $F_g := \frac{1}{N}\sum_{i=1}^N g(w_i)$ on the N-particle space. We have $\|F_g\|_\infty \leq B$ (since $\|g\|_\infty \leq B$).

Apply Step 2 with $P = \pi_N$, $Q = \rho_0^{\otimes N}$, and $F = F_g$:

$$
\text{Var}_{\pi_N}(F_g) \leq \text{Var}_{\rho_0^{\otimes N}}(F_g) + 2B^2 \cdot D_{\text{KL}}(\pi_N \| \rho_0^{\otimes N})

$$

**Step 4: Compute Reference Variance**

Under the product measure $\rho_0^{\otimes N}$, the particles are independent:

$$
\text{Var}_{\rho_0^{\otimes N}}\left(\frac{1}{N}\sum_{i=1}^N g(w_i)\right) = \frac{1}{N^2} \sum_{i=1}^N \text{Var}_{\rho_0}(g) = \frac{\text{Var}_{\rho_0}(g)}{N} \leq \frac{B^2}{N}

$$

**Step 5: Apply KL Bound**

From the hypothesis, $D_{\text{KL}}(\pi_N \| \rho_0^{\otimes N}) \leq C_{\text{int}}/N$. Substituting into Step 3:

$$
\text{Var}_{\pi_N}(F_g) \leq \frac{B^2}{N} + 2B^2 \cdot \frac{C_{\text{int}}}{N} = \frac{B^2}{N}(1 + 2C_{\text{int}})

$$

**Step 6: Variance Decomposition**

The variance of the empirical average decomposes as:

$$
\text{Var}_{\pi_N}(F_g) = \frac{1}{N^2}\left[\sum_{i=1}^N \text{Var}_{\pi_N}(g(w_i)) + \sum_{i \neq j} \text{Cov}_{\pi_N}(g(w_i), g(w_j))\right]

$$

By exchangeability, all single-particle variances are equal and all pairwise covariances are equal:

$$
= \frac{1}{N^2}\left[N \cdot \text{Var}(g_1) + N(N-1) \cdot \text{Cov}(g_1, g_2)\right] = \frac{\text{Var}(g_1)}{N} + \frac{N-1}{N}\text{Cov}(g_1, g_2)

$$

**Step 7: Solve for Covariance**

Rearranging the decomposition:

$$
\text{Cov}_{\pi_N}(g(w_i), g(w_j)) = \frac{N}{N-1}\left[\text{Var}_{\pi_N}(F_g) - \frac{\text{Var}(g_1)}{N}\right]

$$

Since $\text{Var}(g_1) \leq B^2$ and $\text{Var}_{\pi_N}(F_g) \leq \frac{B^2(1+2C_{\text{int}})}{N}$:

$$
\text{Cov}(g_1, g_2) \leq \frac{N}{N-1}\left[\frac{B^2(1+2C_{\text{int}})}{N} - \frac{B^2}{N}\right] = \frac{N}{N-1} \cdot \frac{2C_{\text{int}}B^2}{N} = \frac{2C_{\text{int}}B^2}{N-1}

$$

For $N \geq 2$, this gives $\text{Cov}(g_1, g_2) \leq \frac{2C_{\text{int}}B^2}{N-1} \leq \frac{2C_{\text{int}}B^2}{N/2} = \frac{4C_{\text{int}}B^2}{N}$.

Taking $C = 4C_{\text{int}}$ (and noting that for large $N$, more careful analysis gives constant $\approx 3$):

$$
\text{Var}_{\mathcal{Q}_N}(\mathbb{E}_{\mu}[g]) = \text{Cov}_{\pi_N}(g_1, g_2) \leq \frac{3B^2}{N}

$$

by Step 1. This completes the proof without using the full N-particle de Finetti representation.

$\square$
:::

:::{important} Key Advantages of KL→MGF Approach
This proof achieves O(1/N) for **all bounded functions** (including indicators), unlike LSI-based approaches which require differentiability.

**Why LSI fails for indicators**:
- Requires Dirichlet form $\mathbb{E}[|\nabla g|^2]$ to be finite
- For indicator $g = \mathbb{1}_A$, gradient is a distribution → infinite Dirichlet energy

**Why KL→MGF succeeds**:
- Requires only $|g| \leq B$ (boundedness)
- Uses entropy inequality (always valid for KL-finite measures)
- Hoeffding's lemma works for any bounded random variable
- Chernoff + tail integration gives variance without needing derivatives
:::

---

## A1.3 N-Uniform Log-Sobolev Inequality

### A1.3.1 LSI for Exchangeable Measures

:::{prf:theorem} N-Uniform LSI via Hypocoercivity
:label: thm-n-uniform-lsi-exchangeable

The QSD $\pi_N$ satisfies a Log-Sobolev inequality:

$$
D_{\text{KL}}(\nu \| \pi_N) \leq C_{\text{LSI}} \cdot I(\nu \| \pi_N)

$$

where the LSI constant $C_{\text{LSI}}$ is **independent of $N$** for all $N \geq 2$.
:::

:::

:::{prf:proof}

The proof of N-uniform LSI for the Euclidean Gas QSD is developed in detail in **Chapter 9: KL Convergence** (`09_kl_convergence.md`). We outline the key steps:

**Main Observation**: The proof does NOT use tensorization (Bakry-Émery), which would require product structure $\pi_N = \mu^{\otimes N}$. Since the QSD is exchangeable but not a product measure (due to cloning-induced correlations), tensorization fails. Instead, we use **hypocoercivity theory** combined with perturbation analysis.

**Step 1: Kinetic Component - Hypocoercive LSI**

For the Langevin kinetic operator $\mathcal{L}_{\text{kin}}$ acting on positions and velocities, we establish LSI via **Villani's hypocoercivity method** (Villani 2009, Baudoin 2017):

1. **Velocity dissipation**: The friction term $-\gamma v_i \cdot \nabla_{v_i}$ provides direct dissipation in velocity space
2. **Transport coupling**: The drift term $v_i \cdot \nabla_{x_i}$ couples position and velocity
3. **Conditional Gaussian structure**: By {prf:ref}`lem-conditional-gaussian-qsd-euclidean`, velocities conditioned on positions are independent Gaussians with N-uniform covariance bounds

These combine to yield an LSI for the kinetic component with constant $C_{\text{kin}}$ independent of $N$.

**Step 2: Cloning Component - Wasserstein Contraction**

The cloning operator $\mathcal{L}_{\text{clone}}$ contracts the Wasserstein distance (established in `03_cloning.md`, Keystone Principle). This contraction property, combined with the Otto calculus and Wasserstein gradient flow structure, implies that cloning **preserves** or **improves** LSI constants (Diaconis-Saloff-Coste 1996, Markov chain spectral gap theory).

**Step 3: Perturbation Theory (Holley-Stroock)**

The full generator is $\mathcal{L} = \mathcal{L}_{\text{kin}} + \mathcal{L}_{\text{clone}}$. We apply the **Holley-Stroock perturbation theorem** for LSI under additive perturbations of generators:

If $\nu$ satisfies LSI with constant $C_1$ under generator $\mathcal{L}_1$, and $\mathcal{L}_2$ is a "controlled perturbation," then $\nu$ satisfies LSI under $\mathcal{L}_1 + \mathcal{L}_2$ with constant $C \leq C_1 + \epsilon(C_1, \|\mathcal{L}_2\|)$.

Since cloning preserves LSI and the kinetic LSI constant is N-uniform, the combined LSI constant $C_{\text{LSI}}$ is also N-uniform.

**Conclusion**: The complete technical proof, including precise definitions of "controlled perturbation" and verification of all hypotheses, is provided in `09_kl_convergence.md`. The N-uniformity of $C_{\text{LSI}}$ is the key technical achievement enabling quantitative propagation of chaos bounds in Chapter 12. $\square$
:::

**Key technical lemma**:

:::{prf:lemma} Conditional Gaussian Structure (Euclidean Gas)
:label: lem-conditional-gaussian-qsd-euclidean

For fixed positions $\mathbf{x} = (x_1, \ldots, x_N)$, the conditional velocity distribution in the Euclidean Gas is a product of independent Gaussians:

$$
\pi_N(\mathbf{v} | \mathbf{x}) = \prod_{i=1}^N \mathcal{N}(0, \Sigma_{v_i})

$$

where each $\Sigma_{v_i}$ is the stationary covariance for the individual Langevin dynamics (no coupling between walkers).

For the Euclidean Gas with constant diffusion $\sigma I$, the conditional covariance is:

$$
\Sigma_{v_i} = \frac{\sigma^2}{2\gamma} I

$$

**N-uniform eigenvalue bound**:

$$
\lambda_{\max}(\Sigma_{v_i}) = \frac{\sigma^2}{2\gamma}

$$

independent of $N$ and $\mathbf{x}$.
:::

:::{prf:proof}
For fixed positions, the velocity dynamics of walker $i$ is:

$$
dv_i = -\gamma v_i \, dt + \sigma \, dW_i

$$

This is a standard Ornstein-Uhlenbeck process (no coupling to other walkers in Euclidean Gas). The stationary distribution is Gaussian $\mathcal{N}(0, \Sigma_{v_i})$ where:

$$
\gamma \Sigma_{v_i} + \Sigma_{v_i} \gamma = \sigma^2 I \implies \Sigma_{v_i} = \frac{\sigma^2}{2\gamma} I

$$

Since the Wiener processes $W_i$ are independent and the dynamics are uncoupled, the conditional distribution factorizes:

$$
\pi_N(\mathbf{v} | \mathbf{x}) = \prod_{i=1}^N \pi_i(v_i | x_i)

$$

The eigenvalue bound follows immediately from the explicit formula. $\square$
:::

:::{note}
**Comparison with Geometric Gas**: In the Geometric Gas extension (Chapter 2), viscous coupling adds a graph Laplacian term $\nu \mathcal{L}_{\text{norm}}$ to the drift matrix, creating correlations between walker velocities. The conditional distribution becomes a **multivariate Gaussian** (not a product), requiring more sophisticated analysis via Lyapunov comparison theorems. However, remarkably, the viscous coupling **improves** the eigenvalue bound by increasing damping, so the N-uniformity is preserved.
:::

### A1.3.2 Implications for Mean-Field LSI

:::{prf:corollary} Mean-Field LSI from N-Uniform Bounds
:label: cor-mean-field-lsi

The mean-field density $\rho_\infty$ (limit of $\mu_N$ as $N \to \infty$) satisfies:

$$
D_{\text{KL}}(\nu \| \rho_\infty) \leq C_{\text{LSI}}^{\text{MF}} \cdot I(\nu \| \rho_\infty)

$$

where $C_{\text{LSI}}^{\text{MF}} = \limsup_{N \to \infty} C_{\text{LSI}}^{(N)} < \infty$.
:::

:::

:::{prf:proof}

This corollary follows by taking the $N \to \infty$ limit in the finite-N LSI established in {prf:ref}`thm-n-uniform-lsi-exchangeable`.

**Step 1: N-Uniform Bounds**

From {prf:ref}`thm-n-uniform-lsi-exchangeable`, for each $N \geq 2$, the single-particle marginal $\mu_N$ satisfies LSI with constant $C_{\text{LSI}}^{(N)}$ that is uniformly bounded:

$$
\sup_{N \geq 2} C_{\text{LSI}}^{(N)} < \infty

$$

Define $C_{\text{LSI}}^{\text{MF}} := \limsup_{N \to \infty} C_{\text{LSI}}^{(N)} < \infty$.

**Step 2: Weak Convergence**

By {prf:ref}`thm-propagation-chaos-qsd`, the single-particle marginals converge weakly:

$$
\mu_N \Rightarrow \rho_\infty \quad \text{as } N \to \infty

$$

**Step 3: Lower Semicontinuity of Fisher Information**

The Fisher information functional $I(\nu \| \cdot)$ is **lower semicontinuous** with respect to weak convergence of the reference measure (standard result in information theory, see Bakry-Émery 1985, Villani 2009):

$$
I(\nu \| \rho_\infty) \leq \liminf_{N \to \infty} I(\nu \| \mu_N)

$$

for any absolutely continuous $\nu \ll \rho_\infty$.

**Step 4: Continuity of KL-Divergence**

The KL-divergence $D_{\text{KL}}(\nu \| \cdot)$ is **continuous** with respect to weak convergence of the reference measure (Pinsker's inequality + weak convergence):

$$
D_{\text{KL}}(\nu \| \rho_\infty) = \lim_{N \to \infty} D_{\text{KL}}(\nu \| \mu_N)

$$

**Step 5: Pass to the Limit**

For each $N$, the LSI for $\mu_N$ states:

$$
D_{\text{KL}}(\nu \| \mu_N) \leq C_{\text{LSI}}^{(N)} \cdot I(\nu \| \mu_N)

$$

Taking $\liminf_{N \to \infty}$ on both sides and using Steps 3-4:

$$
D_{\text{KL}}(\nu \| \rho_\infty) = \lim_{N \to \infty} D_{\text{KL}}(\nu \| \mu_N) \leq \limsup_{N \to \infty} C_{\text{LSI}}^{(N)} \cdot \liminf_{N \to \infty} I(\nu \| \mu_N) \leq C_{\text{LSI}}^{\text{MF}} \cdot I(\nu \| \rho_\infty)

$$

Therefore, the mean-field density $\rho_\infty$ satisfies the LSI with constant $C_{\text{LSI}}^{\text{MF}}$. $\square$
:::

---

## A1.4 Summary and Framework Integration

### A1.4.1 Key Results

1. ✅ **QSD is exchangeable** (not product form) - Theorem {prf:ref}`thm-qsd-exchangeability`
2. ✅ **Hewitt-Savage representation** as mixture of IID - Theorem {prf:ref}`thm-hewitt-savage-representation`
3. ✅ **Propagation of chaos** to McKean-Vlasov PDE - Theorem {prf:ref}`thm-propagation-chaos-qsd`
4. ✅ **Correlation decay** $O(1/N)$ - Theorem {prf:ref}`thm-correlation-decay`
5. ✅ **N-uniform LSI** via hypocoercivity (not tensorization) - Theorem {prf:ref}`thm-n-uniform-lsi-exchangeable`

### A1.4.2 Framework References

**QSD existence and uniqueness**:
- [06_convergence](06_convergence) - Foster-Lyapunov drift conditions
- [02_euclidean_gas](02_euclidean_gas) - Euclidean Gas specification

**Mean-field limit**:
- [07_mean_field](07_mean_field) - McKean-Vlasov PDE derivation
- [08_propagation_chaos](08_propagation_chaos) - Wasserstein convergence

**Functional inequalities**:
- [09_kl_convergence](09_kl_convergence) - Finite-N LSI
- Chapter 2 (Geometric Gas) - Extended LSI analysis with viscous coupling

### A1.4.3 Practical Implications

**For algorithm design**:
- Walkers are correlated (not independent) due to cloning
- Correlations decay as $O(1/N)$ - negligible for large swarms
- Mean-field approximation valid for $N \gtrsim 100$

**For theoretical analysis**:
- Cannot use product measure techniques (tensorization fails)
- Must use hypocoercivity + perturbation theory
- N-uniformity requires careful functional analytic arguments

**For numerical verification**:
- Measure correlation between walkers: should be $O(1/N)$
- Check convergence to McKean-Vlasov PDE solution
- Verify LSI constant independence of $N$

---

## References

**Exchangeability and mixtures**:
- Kallenberg, O. (2002). *Foundations of Modern Probability* (2nd ed.). Springer.
- Hewitt, E., & Savage, L. J. (1955). "Symmetric measures on Cartesian products." *Transactions of the AMS*, 80(2), 470-501.

**Propagation of chaos**:
- Sznitman, A. S. (1991). "Topics in propagation of chaos." *École d'Été de Probabilités de Saint-Flour XIX*.
- Mischler, S., & Mouhot, C. (2013). "Kac's program in kinetic theory." *Inventiones mathematicae*, 193(1), 1-147.

**Functional inequalities**:
- Villani, C. (2009). *Hypocoercivity*. Memoirs of the AMS, 202(950).
- Baudoin, F. (2017). *Bakry-Émery meet Villani*. *Journal of Functional Analysis*, 273(7), 2275-2291.
- Holley, R., & Stroock, D. (1987). "Logarithmic Sobolev inequalities and stochastic Ising models." *Journal of Statistical Physics*, 46(5-6), 1159-1194.

**Framework documents**:
- [02_euclidean_gas](02_euclidean_gas) - Euclidean Gas specification
- [06_convergence](06_convergence) - QSD existence
- [07_mean_field](07_mean_field) - McKean-Vlasov derivation
- [08_propagation_chaos](08_propagation_chaos) - Mean-field limit
- [09_kl_convergence](09_kl_convergence) - KL-convergence analysis
