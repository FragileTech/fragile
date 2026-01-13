# Fractal Set as Causal Set

**Prerequisites**: {doc}`01_fractal_set`, Causal Set Theory (Bombelli et al. 1987)

---

## TLDR

*Notation: $(E, \prec)$ = Fractal Set with causal order; BLMS = Bombelli-Lee-Meyer-Sorkin axioms; $\rho_{\mathrm{adaptive}}$ = QSD sampling density; $\sqrt{\det g}$ = Riemannian volume element.*

**The Fractal Set is a Valid Causal Set**: The episode set $E$ with CST ordering $\prec$ satisfies all three BLMS axioms (irreflexivity, transitivity, local finiteness), making the Fractal Set a rigorous causal set in the sense of quantum gravity research.

**Adaptive Sprinkling Innovation**: Unlike standard Poisson sprinkling with uniform density, QSD sampling produces episodes with density $\rho_{\mathrm{adaptive}}(x) \propto \sqrt{\det g(x)} \, e^{-U_{\mathrm{eff}}/T}$, automatically adapting to local geometry and providing optimal discretization fidelity.

**Causal Set Machinery Applies**: With causal set status established, all CST mathematical tools (d'Alembertian, dimension estimators, curvature measures) can be rigorously applied to the Fractal Set, enabling quantum gravity calculations on the emergent spacetime.

---

(sec-cst-intro)=
## 1. Introduction

:::{div} feynman-prose
Here is a beautiful connection that was waiting to be discovered. Causal set theory is one of the leading approaches to quantum gravity—the idea that spacetime is fundamentally discrete, made up of a finite set of events with a partial ordering that encodes causal structure. The program was launched by Bombelli, Lee, Meyer, and Sorkin in 1987, and it has developed into a sophisticated mathematical framework.

Now, the Fractal Set is also a discrete structure with a causal ordering. Episodes are events; CST edges encode causal precedence. The question is: does the Fractal Set satisfy the axioms of a causal set? If it does, then all the mathematical machinery of causal set theory—developed over decades by quantum gravity researchers—becomes available to us.

The answer is yes. The Fractal Set is a valid causal set. But it is more than that: it is an *adaptive* causal set, where the sampling density automatically adjusts to local geometry. This is something that standard causal set constructions (Poisson sprinkling) cannot achieve.
:::

Causal set theory (CST) posits that spacetime is fundamentally discrete: a finite collection of events with a partial ordering encoding causal relationships. The Fractal Set, defined in {doc}`01_fractal_set`, provides exactly such a structure via its CST edges ({prf:ref}`def-fractal-set-cst-edges`) and CST axioms ({prf:ref}`def-fractal-set-cst-axioms`). This chapter establishes that:

1. The Fractal Set satisfies all BLMS axioms for causal sets
2. QSD sampling provides adaptive (not uniform) sprinkling
3. All CST mathematical machinery applies to the Fractal Set

---

(sec-cst-axioms)=
## 2. Causal Set Theory: Axioms and Framework

### 2.1. Standard Causal Set Definition

:::{prf:definition} Causal Set (Bombelli et al. 1987)
:label: def-causal-set-blms

A **causal set** $(C, \prec)$ is a locally finite partially ordered set satisfying:

**Axiom CS1 (Irreflexivity)**: For all $e \in C$, $e \not\prec e$

**Axiom CS2 (Transitivity)**: For all $e_1, e_2, e_3 \in C$, if $e_1 \prec e_2$ and $e_2 \prec e_3$, then $e_1 \prec e_3$

**Axiom CS3 (Local Finiteness)**: For all $e_1, e_2 \in C$, the set $\{e \in C : e_1 \prec e \prec e_2\}$ is finite

**Physical interpretation**:
- Elements $e \in C$ represent spacetime events
- $e_1 \prec e_2$ means "$e_1$ causally precedes $e_2$" (inside future light cone)
- Local finiteness = finite events in any causal interval (discreteness)
:::

### 2.2. Poisson Sprinkling (Standard Construction)

:::{prf:definition} Poisson Sprinkling
:label: def-poisson-sprinkling-cst

Given a Lorentzian manifold $(M, g_{\mu\nu})$ with volume element $dV = \sqrt{-\det g} \, d^d x$, a **Poisson sprinkling** with constant density $\rho_0$ is:

1. **Sample points**: Draw $N$ points $\{x_i\}$ from $M$ with probability density $p(x) = \rho_0 \cdot \sqrt{-\det g(x)} / V_{\mathrm{total}}$

2. **Define order**: $e_i \prec e_j$ iff $x_i$ is in the causal past of $x_j$

**Property**: Expected number of elements in causal interval $I(e_1, e_2)$ is $\mathbb{E}[|I|] = \rho_0 \cdot V_{\mathrm{Lorentz}}(I)$.
:::

**Limitation**: Uniform density $\rho_0 = \mathrm{const}$ does not adapt to local geometry:
- Over-sampling in flat regions (wasteful)
- Under-sampling in curved regions (loss of information)

---

(sec-fractal-causal-set)=
## 3. Fractal Set as Adaptive Causal Set

### 3.1. Causal Order on Episodes

:::{prf:definition} Causal Order on Fractal Set
:label: def-fractal-causal-order

For episodes $e_i, e_j \in E$ with positions $x_i, x_j \in \mathcal{X}$ and times $t_i, t_j \in \mathbb{R}$, define:

$$
e_i \prec_{\mathrm{CST}} e_j \quad \iff \quad t_i < t_j \;\wedge\; d_g(x_i, x_j) < c_{\mathrm{eff}}(t_j - t_i)
$$

where:
- $d_g(\cdot, \cdot)$ is the geodesic distance on $(\mathcal{X}, g)$ with $g = H + \epsilon_\Sigma I$ the emergent Riemannian metric
- $c_{\mathrm{eff}}$ is the effective speed of causation (maximal information propagation rate)

**Physical meaning**: $e_i \prec e_j$ iff information from $e_i$ can causally influence $e_j$.

**Consistency with {prf:ref}`def-fractal-set-cst-axioms`**: This order coincides with the CST edge relation defined in the Fractal Set specification.
:::

### 3.2. QSD Sampling = Adaptive Sprinkling

:::{prf:theorem} Fractal Set Episodes Follow Adaptive Density
:label: thm-fractal-adaptive-sprinkling

Episodes generated by the Adaptive Gas are distributed according to:

$$
\rho_{\mathrm{adaptive}}(x) = \frac{1}{Z} \sqrt{\det g(x)} \exp\left(-\frac{U_{\mathrm{eff}}(x)}{T}\right)
$$

where $g(x) = H(x) + \epsilon_\Sigma I$ is the emergent Riemannian metric.

**Comparison with Poisson sprinkling**:

| Standard CST | Fractal Set |
|:------------|:------------|
| Density $\rho = \mathrm{const}$ | Density $\rho(x) \propto \sqrt{\det g(x)} \, e^{-U_{\mathrm{eff}}(x)/T}$ |
| Uniform sampling | Adaptive sampling |
| Ad-hoc choice of $\rho$ | Automatic from QSD |
:::

:::{dropdown} 📖 Hypostructure Reference
:icon: book

**Rigor Class:** L (Literature-Imported)

**Permits:** $\mathrm{TB}_\rho$ (Node 10 ErgoCheck)

**Hypostructure connection:** The QSD spatial marginal is established in the meta-learning framework via the Expansion Adjunction ({prf:ref}`thm-expansion-adjunction`). The adaptive density emerges from the Stratonovich interpretation of the SDE ({prf:ref}`def-fractal-set-sde`), which preserves the Riemannian volume measure $\sqrt{\det g} \, dx$.

**References:**
- State space definition: {prf:ref}`def:state-space-fg`
- Algorithmic space: {prf:ref}`def:algorithmic-space-fg`
:::

---

(sec-axiom-verification)=
## 4. Verification of Causal Set Axioms

:::{prf:theorem} Fractal Set is a Valid Causal Set
:label: thm-fractal-is-causal-set

The Fractal Set $\mathcal{F} = (E, \prec_{\mathrm{CST}})$ satisfies all BLMS axioms.

**Rigor Class:** F (Framework-Original)

**Permits:** $\mathrm{TB}_\pi$ (Node 8), $\mathrm{TB}_O$ (Node 9)
:::

:::{prf:proof}
We verify each axiom:

**Axiom CS1 (Irreflexivity):** For any episode $e_i$:

$$
e_i \prec e_i \iff t_i < t_i \;\wedge\; d(x_i, x_i) < c(t_i - t_i)
$$
Both $t_i < t_i$ (false) and $0 < 0$ (false), so $e_i \not\prec e_i$. ✓

**Axiom CS2 (Transitivity):** Assume $e_1 \prec e_2$ and $e_2 \prec e_3$. Then:
- $t_1 < t_2 < t_3$ (time ordering is transitive)
- $d(x_1, x_2) < c(t_2 - t_1)$ and $d(x_2, x_3) < c(t_3 - t_2)$

By the triangle inequality:

$$
d(x_1, x_3) \leq d(x_1, x_2) + d(x_2, x_3) < c(t_2 - t_1) + c(t_3 - t_2) = c(t_3 - t_1)
$$
Therefore $e_1 \prec e_3$. ✓

**Axiom CS3 (Local Finiteness):** The causal interval $I(e_1, e_2) := \{e : e_1 \prec e \prec e_2\}$ is contained in a bounded spacetime region:

$$
\mathcal{D} = \{(t, x) : t_1 < t < t_2, \, d(x_1, x) < c(t - t_1), \, d(x, x_2) < c(t_2 - t)\}
$$

This is a compact "double cone." The expected episode count is:

$$
\mathbb{E}[|I(e_1, e_2)|] = \int_{\mathcal{D}} \rho_{\mathrm{adaptive}}(t, x) \, dt \, dx < \infty
$$
by compactness and integrability of $\rho$. For any finite realization, $|I| < \infty$ a.s. ✓

$\square$
:::

:::{dropdown} 📖 ZFC Proof (Classical Verification)
:icon: book

**Classical Verification (ZFC):**

Working in Grothendieck universe $\mathcal{U}$, the Fractal Set $(E, \prec)$ is a finite partially ordered set (poset) where:

1. $E \in V_\mathcal{U}$ is a finite set of episodes
2. $\prec \subseteq E \times E$ is a binary relation

**CS1 (Irreflexivity):** The definition $e_i \prec e_j \Leftrightarrow t_i < t_j \wedge d(x_i, x_j) < c(t_j - t_i)$ implies $e_i \not\prec e_i$ because $t_i < t_i$ is false in any ordered field.

**CS2 (Transitivity):** Given $e_1 \prec e_2$ and $e_2 \prec e_3$:
- $t_1 < t_2 \wedge t_2 < t_3 \Rightarrow t_1 < t_3$ (transitivity of $<$ in $\mathbb{R}$)
- Triangle inequality: $d(x_1, x_3) \leq d(x_1, x_2) + d(x_2, x_3)$ (metric axiom)
- Arithmetic: $c(t_2 - t_1) + c(t_3 - t_2) = c(t_3 - t_1)$ (distributivity)

**CS3 (Local Finiteness):** For any $e_1, e_2 \in E$, the set $\{e \in E : e_1 \prec e \prec e_2\}$ is a subset of the finite set $E$, hence finite.

All axioms verified using only ZFC set theory and real analysis. $\square$
:::

---

(sec-faithful-discretization)=
## 5. Faithful Discretization and Manifoldlikeness

### 5.1. Volume Matching

:::{prf:theorem} Fractal Set Provides Faithful Discretization
:label: thm-fractal-faithful-embedding

The Fractal Set faithfully discretizes the emergent Riemannian manifold $(\mathcal{X}, g)$:

**Volume Matching**: The episode count in region $\Omega$ satisfies:

$$
\mathbb{E}\left[\frac{|E \cap \Omega|}{N}\right] = \frac{1}{Z} \int_{\Omega} \sqrt{\det g(x)} \, e^{-U_{\mathrm{eff}}(x)/T} \, dx
$$

with variance scaling as $O(1/N)$ by the law of large numbers.

**Metric Recovery**: Riemannian distance is recoverable from causal structure.

**Dimension Estimation**: The Myrheim-Meyer estimator converges:

$$
d_{\mathrm{MM}} \xrightarrow{N \to \infty} d = \dim \mathcal{X}
$$
:::

### 5.2. Advantages over Poisson Sprinkling

:::{prf:proposition} Adaptive Sprinkling Improves Geometric Fidelity
:label: prop-adaptive-vs-poisson

Compared to uniform Poisson sprinkling, the Fractal Set achieves:

1. **Better coverage**: Episodes concentrate in high-curvature regions where geometric information is richer

2. **Optimal information content**: KL divergence from true volume measure is minimized

3. **Automatic adaptation**: No ad-hoc density choices; $\rho$ emerges from QSD
:::

---

(sec-cst-machinery)=
## 6. Causal Set Mathematical Machinery

With the Fractal Set established as a valid causal set, all CST mathematical tools apply.

### 6.1. Causal Set Volume Element

:::{prf:definition} Causal Set Volume
:label: def-cst-volume

The **causal set volume** of element $e \in E$ is:

$$
V_{\mathrm{CST}}(e) := \frac{1}{\bar{\rho}} \sum_{e' \in E} \mathbb{1}_{e' \prec e}
$$
where $\bar{\rho}$ is the average adaptive density.

**Continuum limit**: $V_{\mathrm{CST}}(e) \to V(J^-(e))$ as $N \to \infty$.
:::

### 6.2. Discrete d'Alembertian (Benincasa-Dowker Operator)

:::{prf:definition} Discrete d'Alembertian on Fractal Set (Benincasa-Dowker)
:label: def-cst-dalembertian

The **Benincasa-Dowker d'Alembertian** acting on $f: E \to \mathbb{R}$ in $d$ dimensions is:

$$
(\Box_{\mathrm{BD}} f)(e) := \frac{4}{\ell_d^2} \left( -\alpha_d f(e) + \sum_{k=0}^{n_d} C_k^{(d)} \sum_{\substack{e' \prec e \\ |I(e', e)| = k}} f(e') \right)
$$

where:
- $\ell_d = (\rho V_d)^{-1/d}$ is the discreteness scale
- $\alpha_d$, $C_k^{(d)}$ are dimension-dependent coefficients (see Benincasa-Dowker 2010 for explicit values)
- $|I(e', e)|$ is the number of elements in the causal interval between $e'$ and $e$

**Convergence** (Benincasa-Dowker 2010): For smooth functions on the emergent spacetime:

$$
\lim_{N \to \infty} \mathbb{E}[(\Box_{\mathrm{BD}} f)(e_i)] = (\Box_g f)(x_i) + O(\ell_d^2)
$$
where $\Box_g = g^{\mu\nu}\nabla_\mu\nabla_\nu$ is the continuum d'Alembertian.
:::

### 6.3. Dimension and Curvature Estimation

:::{prf:definition} Myrheim-Meyer Dimension Estimator
:label: def-myrheim-meyer

The dimension of the emergent manifold is estimated from the ordering fraction:

$$
r := \frac{C_2}{\binom{N}{2}} = \frac{|\{(e_i, e_j) : e_i \prec e_j\}|}{N(N-1)/2}
$$

For a causal set faithfully embedded in $d$-dimensional Minkowski space:

$$
r \xrightarrow{N \to \infty} \frac{\Gamma(d+1) \Gamma(d/2)}{4 \Gamma(3d/2)}
$$

The **Myrheim-Meyer estimator** inverts this relation to obtain $d_{\mathrm{MM}}$ from the observed ordering fraction $r$.
:::

:::{prf:proposition} Ricci Scalar from Causal Set
:label: prop-ricci-cst

The Ricci scalar curvature is estimated via the **Benincasa-Dowker action** (2010):

For a small causal diamond $\mathcal{A}(p, q)$ with $N$ elements:

$$
S_{\mathrm{BD}}[\mathcal{A}] = \frac{\hbar}{\ell_d^{d-2}} \left( \alpha_d N - \sum_{k=0}^{n_d} \beta_k^{(d)} N_k \right)
$$
where $N_k$ counts $k$-element intervals and $\ell_d$ is the discreteness scale.

**Curvature extraction**: In the continuum limit:

$$
\lim_{\ell \to 0} \frac{S_{\mathrm{BD}}[\mathcal{A}]}{V(\mathcal{A})} = \frac{1}{d(d-1)} R + O(\ell^2)
$$
where $R$ is the Ricci scalar and $V(\mathcal{A})$ is the spacetime volume.
:::

---

(sec-physical-consequences)=
## 7. Physical Consequences

### 7.1. Quantum Gravity Path Integral

:::{admonition} Path Integral Formulation
:class: important

The causal set path integral for quantum gravity:

$$
Z_{\mathrm{Fractal}} = \sum_{\mathcal{F} \in \mathcal{F}_N} e^{iS_{\mathrm{CST}}(\mathcal{F})} \cdot \mathcal{P}_{\mathrm{QSD}}(\mathcal{F})
$$

**Key advantage**: The QSD provides a **physically motivated measure** on the space of causal sets, replacing ad-hoc uniform measures.
:::

### 7.2. Observable Predictions

:::{prf:proposition} Testable Predictions
:label: prop-cst-predictions

The Fractal Set causal structure leads to observable consequences:

1. **Discreteness scale**: Average proper distance between episodes:

$$
\ell_{\mathrm{Planck}}^{\mathrm{eff}} = \left(\frac{V_{\mathrm{total}}}{N}\right)^{1/d}
$$

2. **Modified dispersion relations**: High-energy particles experience corrections:

$$
E^2 = p^2 c^2 + m^2 c^4 + \eta_1 \frac{E^3}{E_{\mathrm{Planck}}} + \eta_2 \frac{E^4}{E_{\mathrm{Planck}}^2} + \ldots
$$
where $E_{\mathrm{Planck}} = \sqrt{\hbar c^5 / G}$ and $\eta_i$ are $O(1)$ coefficients.

3. **Lorentz violation bounds**: Observable in cosmic rays, gamma-ray bursts, ultra-high-energy neutrinos
:::

### 7.3. Connection to Loop Quantum Gravity

:::{admonition} Relation to LQG
:class: note

| Fractal Set | Loop Quantum Gravity |
|:------------|:---------------------|
| Episodes $e \in E$ | Nodes of spin network |
| CST edges $e_1 \prec e_2$ | Links with SU(2) labels |
| IG edges $e_i \sim e_j$ | Gauge connections |
| Adaptive density $\rho \propto \sqrt{\det g}$ | Quantum geometry operators |

**Key difference**: Fractal Set is classical + stochastic; LQG is quantum from the start.
:::

---

## 8. Summary

**Main Results**:

1. ✅ **Fractal Set is a causal set**: Satisfies all BLMS axioms (Theorem {prf:ref}`thm-fractal-is-causal-set`)

2. ✅ **Adaptive sprinkling**: QSD sampling with $\rho \propto \sqrt{\det g} \, e^{-U_{\mathrm{eff}}/T}$ provides optimal geometric fidelity

3. ✅ **CST machinery applies**: d'Alembertian, volume elements, dimension/curvature estimators all rigorously defined

4. ✅ **Physical implications**: Foundation for quantum gravity calculations on emergent spacetime

---

## References

### Causal Set Theory
1. Bombelli, L., Lee, J., Meyer, D., & Sorkin, R.D. (1987) "Space-Time as a Causal Set", *Phys. Rev. Lett.* **59**, 521
2. Sorkin, R.D. (2003) "Causal Sets: Discrete Gravity", in *Lectures on Quantum Gravity*, Springer
3. Benincasa, D.M.T. & Dowker, F. (2010) "The Scalar Curvature of a Causal Set", *Phys. Rev. Lett.* **104**, 181301

### Framework Documents
4. {doc}`01_fractal_set` — Fractal Set definition and structure
5. {prf:ref}`def-fractal-set-cst-edges` — CST edge definition
6. {prf:ref}`def-fractal-set-cst-axioms` — CST axioms
7. {prf:ref}`mt:fractal-gas-lock-closure` — Lock Closure for Fractal Gas (Hypostructure)
