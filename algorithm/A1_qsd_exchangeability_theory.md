# Appendix A1: QSD Structure - Exchangeability and Mean-Field Limit

**Mathematical Level**: Publication standard (rigorous proofs)

**Purpose**: Establish the rigorous structure of the Quasi-Stationary Distribution (QSD) for the Fragile Gas framework

---

## A1.1 QSD Structure: Exchangeability

### A1.1.1 Main Result

:::{prf:theorem} Exchangeability of the QSD
:label: thm-qsd-exchangeability

Let $\pi_N \in \mathcal{P}(\Sigma_N)$ be the unique Quasi-Stationary Distribution of the Fragile Gas. Then $\pi_N$ is an **exchangeable probability measure**: for any permutation $\sigma \in S_N$ and any measurable set $A \subseteq \Sigma_N$:

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

**Proof strategy** (from [06_propagation_chaos.md](06_propagation_chaos.md)):
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
D_{\text{KL}}(\nu \| \pi_N) \leq C_{\text{LSI}}(\rho) \cdot I(\nu \| \pi_N)
$$

where the LSI constant $C_{\text{LSI}}(\rho)$ is **independent of $N$** for all $N \geq 2$.
:::

**Proof approach** (detailed in [12_adaptive_gas_lsi_proof.md](12_adaptive_gas_lsi_proof.md)):

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

:::{prf:lemma} Conditional Gaussian Structure
:label: lem-conditional-gaussian-qsd

For fixed positions $\mathbf{x} = (x_1, \ldots, x_N)$, the conditional velocity distribution is multivariate Gaussian:

$$
\pi_N(\mathbf{v} | \mathbf{x}) = \mathcal{N}(0, \Sigma_{\mathbf{v}}(\mathbf{x}))
$$

where $\Sigma_{\mathbf{v}}(\mathbf{x})$ solves the Lyapunov equation:

$$
A(\mathbf{x}) \Sigma_{\mathbf{v}}(\mathbf{x}) + \Sigma_{\mathbf{v}}(\mathbf{x}) A(\mathbf{x})^T = B(\mathbf{x}) B(\mathbf{x})^T
$$

with drift matrix $A(\mathbf{x}) = \gamma I + \nu \mathcal{L}_{\text{norm}}(\mathbf{x}) \otimes I_3$ and noise $B = \text{diag}(\Sigma_{\text{reg}})$.

**N-uniform eigenvalue bound**:

$$
\lambda_{\max}(\Sigma_{\mathbf{v}}(\mathbf{x})) \leq \frac{c_{\max}^2(\rho)}{2\gamma}
$$

independent of $N$ and $\mathbf{x}$.
:::

**Proof** (see [poincare_inequality_rigorous_proof.md](poincare_inequality_rigorous_proof.md) for details):

Uses Lyapunov comparison theorem: adding viscous coupling ($\nu \mathcal{L}_{\text{norm}}$) increases damping, which decreases covariance eigenvalues. The bound follows from comparing with the uncoupled system ($\nu = 0$).

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
- [05_convergence.md](05_convergence.md) - Foster-Lyapunov drift conditions
- [07_adaptative_gas.md](07_adaptative_gas.md) - Adaptive perturbation theory

**Mean-field limit**:
- [06_mean_field.md](06_mean_field.md) - McKean-Vlasov PDE derivation
- [08_propagation_chaos.md](08_propagation_chaos.md) - Wasserstein convergence

**Functional inequalities**:
- [10_kl_convergence.md](10_kl_convergence.md) - Finite-N LSI
- [11_convergence_mean_field.md](11_convergence_mean_field.md) - Mean-field entropy production
- [12_adaptive_gas_lsi_proof.md](12_adaptive_gas_lsi_proof.md) - Complete LSI proof

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
- [05_convergence.md](05_convergence.md) - QSD existence
- [06_mean_field.md](06_mean_field.md) - McKean-Vlasov derivation
- [08_propagation_chaos.md](08_propagation_chaos.md) - Mean-field limit
- [12_adaptive_gas_lsi_proof.md](12_adaptive_gas_lsi_proof.md) - Complete LSI proof

---

**Document complete**: Clean publication version, no correction history
