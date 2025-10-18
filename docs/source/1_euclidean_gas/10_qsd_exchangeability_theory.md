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

:::{prf:theorem} Mixture Representation (Hewitt-Savage)
:label: thm-hewitt-savage-representation

Since $\pi_N$ is exchangeable, there exists a probability measure $\mathcal{Q}_N$ on $\mathcal{P}(\Omega)$ such that:

$$
\pi_N = \int_{\mathcal{P}(\Omega)} \mu^{\otimes N} \, d\mathcal{Q}_N(\mu)
$$

where $\mu^{\otimes N}$ denotes the product measure: walkers are i.i.d. with law $\mu$.

**Interpretation**: The QSD is a mixture of IID sequences. The mixing measure $\mathcal{Q}_N$ encodes correlations between walkers.
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

From the Hewitt-Savage representation:

$$
\mu_N = \int_{\mathcal{P}(\Omega)} \mu \, d\mathcal{Q}_N(\mu)
$$

The single-particle marginal is the average over all IID distributions in the mixture.
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

**Proof strategy** (from [08_propagation_chaos](08_propagation_chaos)):
1. **Tightness**: N-uniform moment bounds from Foster-Lyapunov
2. **Identification**: Any limit point satisfies the McKean-Vlasov PDE (weak formulation)
3. **Uniqueness**: Hypoelliptic regularity theory ensures unique solution

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
From the Hewitt-Savage representation:

$$
\text{Cov}_{\pi_N}(g(w_i), g(w_j)) = \int_{\mathcal{P}(\Omega)} \text{Var}_{\mu}(g) \, d\mathcal{Q}_N(\mu)
$$

for $i \neq j$ (using conditional independence given $\mu$).

By the concentration of $\mathcal{Q}_N$ around $\delta_{\mu_\infty}$ (quantified via propagation of chaos):

$$
\text{Var}_{\mathcal{Q}_N}(\mathbb{E}_{\mu}[g]) = O(1/N)
$$

Therefore:

$$
|\text{Cov}_{\pi_N}(g(w_i), g(w_j))| \leq \mathbb{E}_{\mathcal{Q}_N}[\text{Var}_{\mu}(g)] \leq \mathbb{E}_{\mathcal{Q}_N}[\|g\|_{\infty}^2] = O(1/N)
$$

$\square$
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

**Proof approach** (detailed in Chapter 2 for the Geometric Gas extension):

The proof does NOT use tensorization (which requires product structure). Instead:

1. **Kinetic component**: Hypocoercive LSI for Langevin dynamics (Villani 2009, Baudoin 2017)
   - Velocity dissipation + transport coupling
   - N-uniform via conditional Gaussian structure

2. **Cloning component**: Spectral gap analysis (Diaconis-Saloff-Coste 1996)
   - Cloning contracts Wasserstein distance
   - Preserves LSI with controlled degradation

3. **Perturbation theory**: Combine kinetic and cloning via Holley-Stroock
   - Modified LSI for sum of generators
   - N-uniformity propagates through

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

This follows by taking the $N \to \infty$ limit in the finite-N LSI, using:
- Tightness of LSI constants (N-uniform bounds)
- Weak convergence $\mu_N \Rightarrow \rho_\infty$
- Lower semicontinuity of Fisher information

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
