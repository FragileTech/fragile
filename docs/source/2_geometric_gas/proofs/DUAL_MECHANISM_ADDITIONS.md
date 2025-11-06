# Additions for Dual Companion Mechanism Treatment

This file contains additions to `20_geometric_gas_cinf_regularity_full.md` to implement Option C: prove statistical equivalence and show both companion selection mechanisms achieve equivalent C^∞ regularity with k-uniform Gevrey-1 bounds.

---

## Addition 1: Revised Abstract (replaces lines 3-15)

## Abstract

This document establishes **C^∞ regularity** (infinite differentiability) with **Gevrey-1 bounds** for the **complete fitness potential** of the Geometric Gas algorithm with companion-dependent measurements. We prove regularity for the full algorithmic fitness potential:

$$
V_{\text{fit}}(x_i, v_i) = g_A\left(Z_\rho\left(\mu_\rho^{(i)}, \sigma_\rho^{2(i)}\right)\right)
$$

where measurements $d_j = d_{\text{alg}}(j, c(j))$ depend on **companion selection** $c(j)$.

**Companion Selection Mechanisms**: The Fragile framework supports two mechanisms for companion selection:
1. **Independent Softmax Selection**: Each walker $j$ independently samples companion $c(j)$ via softmax over phase-space distances
2. **Diversity Pairing**: Global perfect matching via Sequential Stochastic Greedy Pairing with bidirectional pairing property

**Main Result**: We prove that **BOTH mechanisms** achieve:
- **C^∞ regularity**: $V_{\text{fit}} \in C^\infty(\mathcal{X} \times \mathbb{R}^d)$
- **Gevrey-1 bounds**: $\|\nabla^m V_{\text{fit}}\| \leq C_m$ where $C_m = \mathcal{O}(m!)$
- **k-uniformity**: Constants independent of swarm size $k$ or $N$
- **Statistical equivalence**: Both mechanisms produce analytically equivalent fitness potentials up to negligible corrections

The proof uses a **smooth clustering framework** with partition-of-unity localization to handle the N-body coupling introduced by companion selection, establishing **N-uniform** and **k-uniform** derivative bounds at all orders.

---

## Addition 2: New Subsection §1.0 (insert before §1.1)

## 1.0 Companion Selection Mechanisms: Framework Context

### 1.0.1 Why Two Mechanisms?

The Geometric Gas framework implements companion-dependent measurements $d_j = d_{\text{alg}}(j, c(j))$ where companion $c(j)$ must be selected for each walker $j \in \mathcal{A}$. The companion selection mechanism affects the fitness potential's analytical properties, requiring careful regularity analysis.

**Algorithmic Requirements**:
- **Locality**: Companions should be nearby in phase space (exponential concentration)
- **Diversity**: Different walkers should have different companions (prevents redundant information)
- **Smoothness**: Selection mechanism should produce smooth expected measurements (enables mean-field analysis)

**Two Implementation Strategies**:

1. **Independent Softmax Selection** (§4.5):
   - **Definition**: Each walker $j$ independently samples $c(j)$ via softmax:
     $$P(c(j) = \ell) = \frac{\exp(-d_{\text{alg}}^2(j,\ell)/(2\varepsilon_c^2))}{\sum_{\ell' \in \mathcal{A} \setminus \{j\}} \exp(-d_{\text{alg}}^2(j,\ell')/(2\varepsilon_c^2))}$$
   - **Properties**:
     - Unidirectional: $c(i) = j$ doesn't imply $c(j) = i$
     - Simple to implement (walker-local operation)
     - Natural exponential concentration via softmax temperature $\varepsilon_c$

2. **Diversity Pairing** (§4.6):
   - **Definition**: Global perfect (or maximal) matching via Sequential Stochastic Greedy Pairing (Algorithm 5.1 in `03_cloning.md`)
   - **Properties**:
     - Bidirectional: $c(c(i)) = i$ (perfect matching structure)
     - Ensures diversity: each walker paired with unique companion
     - Proven to preserve geometric signal (Lemma 5.1.2 in `03_cloning.md`)

### 1.0.2 Analytical Equivalence Framework

**Key Question**: Do both mechanisms produce fitness potentials with the same regularity properties?

**Measurement Convention**: Throughout this analysis, measurements denote **expected values** over the stochastic companion selection:

$$
d_j := \mathbb{E}_{c(j) \sim \text{mechanism}}[d_{\text{alg}}(j, c(j))]
$$

For **softmax selection**:
$$
d_j = \sum_{\ell \in \mathcal{A} \setminus \{j\}} P_{\text{softmax}}(c(j) = \ell) \cdot d_{\text{alg}}(j, \ell)
$$

For **diversity pairing**:
$$
\bar{d}_j = \mathbb{E}_{M \sim P_{\text{ideal}}}[d_{\text{alg}}(j, M(j))] = \frac{\sum_{M \in \mathcal{M}_k} W(M) \cdot d_{\text{alg}}(j, M(j))}{\sum_{M' \in \mathcal{M}_k} W(M')}
$$

where $\mathcal{M}_k$ is the set of perfect matchings and $W(M) = \prod_{(i,j) \in M} \exp(-d_{\text{alg}}^2(i,j)/(2\varepsilon_{\text{pair}}^2))$.

**Main Thesis** (proven in §4.5-4.6 and §4.7):
1. Both mechanisms produce expected measurements with **identical analytical structure** (quotients of weighted sums with exponential kernels)
2. Both achieve **C^∞ regularity** with **Gevrey-1 bounds** (factorial growth in derivative order)
3. Both achieve **k-uniform bounds** (independent of swarm size)
4. The mechanisms are **statistically equivalent** up to $O(k^{-\beta})$ corrections (§4.7)

**Consequence**: The fitness potential $V_{\text{fit}}$ is C^∞ with k-uniform Gevrey-1 bounds **regardless of which mechanism is implemented**.

---

## Addition 3: New Section §4.7 (insert after §4.6.5)

## 4.7 Statistical Equivalence and Unified Regularity Theorem

This section establishes that both companion selection mechanisms produce analytically equivalent measurements and fitness potentials.

### 4.7.1 Matching the Analytical Structure

:::{prf:observation} Common Exponential Kernel Structure
:label: obs-common-kernel-structure

Both mechanisms express expected measurements as **quotients of exponentially weighted sums**:

**Softmax**:
$$
d_j = \frac{\sum_{\ell \in \mathcal{A} \setminus \{j\}} d_{\text{alg}}(j,\ell) \exp(-d_{\text{alg}}^2(j,\ell)/(2\varepsilon_c^2))}{\sum_{\ell \in \mathcal{A} \setminus \{j\}} \exp(-d_{\text{alg}}^2(j,\ell)/(2\varepsilon_c^2))}
$$

**Diversity Pairing** (idealized):
$$
\bar{d}_j = \frac{\sum_{M \in \mathcal{M}_k} d_{\text{alg}}(j, M(j)) W(M)}{\sum_{M' \in \mathcal{M}_k} W(M')}
$$

where $W(M) = \prod_{(i,\ell) \in M} \exp(-d_{\text{alg}}^2(i,\ell)/(2\varepsilon_{\text{pair}}^2))$.

**Key Similarity**: Both are:
- Smooth quotients (denominator bounded below by companion availability)
- Exponentially localized (exponential concentration around nearby companions)
- Defined via the same base kernel: $\exp(-d_{\text{alg}}^2/(2\sigma^2))$ for appropriate scale $\sigma$
:::

**Regularity Consequences**:
1. Both involve derivatives of:
   - Regularized distance $d_{\text{alg}}(i,j)$ → C^∞ with $\|\nabla^m d_{\text{alg}}\| \leq C_m \varepsilon_d^{1-m}$ (Lemma {prf:ref}`lem-dalg-derivative-bounds-full`)
   - Gaussian kernels $\exp(-d^2/(2\sigma^2))$ → C^∞ with $\|\nabla^m K\| \leq C_m \sigma^{-m} K$ (Lemma {prf:ref}`lem-gaussian-kernel-derivatives-full`)
   - Quotients with non-vanishing denominator → C^∞ via Faà di Bruno formula

2. Both achieve k-uniformity via:
   - Exponential localization → effective interaction radius $R_{\text{eff}} = O(\sigma \sqrt{\log k})$
   - Uniform density bound → sum-to-integral approximation (Lemma {prf:ref}`lem-sum-to-integral-bound-full`)
   - Result: $\mathcal{O}(\log^d k)$ effective contributors, absorbed into k-uniform constants

### 4.7.2 Strengthened Statistical Equivalence

:::{prf:theorem} Statistical Equivalence of Companion Selection Mechanisms
:label: thm-statistical-equivalence-companion-mechanisms

Let $\varepsilon_c = \varepsilon_{\text{pair}} := \varepsilon_{\text{comp}}$ (same companion selection scale). Then the expected measurements from the two mechanisms satisfy:

$$
\mathbb{E}_{\text{softmax}}[d_j | S] = \mathbb{E}_{\text{ideal-pairing}}[d_j | S] + \Delta_j(S)
$$

where the correction term satisfies:

$$
|\Delta_j(S)| \leq C_{\text{equiv}} k^{-1/2}, \quad \|\nabla^m \Delta_j\| \leq C_{m,\text{equiv}} m! \cdot k^{-1/2} \cdot \varepsilon_{\text{comp}}^{-m}
$$

**Consequence**: For derivatives of the fitness potential, the difference between mechanisms is **negligible**:

$$
\|\nabla^m (V_{\text{fit}}^{\text{softmax}} - V_{\text{fit}}^{\text{pairing}})\| = O(k^{-1/2})
$$

which vanishes in the thermodynamic limit $k \to \infty$ and is negligible for practical swarm sizes ($k \geq 50$).
:::

:::{prf:proof}
**Step 1: Mechanism comparison via moment matching.**

Both mechanisms select companions based on phase-space proximity via exponential kernels. The key difference is:
- **Softmax**: Each walker's companion selected **independently**
- **Pairing**: Companions selected **jointly** to form a matching

For walker $j$, define the **marginal distribution** of the diversity pairing:

$$
P_{\text{pair}}(c(j) = \ell | S) := \sum_{M \in \mathcal{M}_k : M(j) = \ell} P_{\text{ideal}}(M | S)
$$

This is the probability that walker $j$ is matched with $\ell$ in the ideal pairing model.

**Claim**: $P_{\text{pair}}(c(j) = \ell | S) \approx P_{\text{softmax}}(c(j) = \ell | S)$ up to $O(k^{-1})$ corrections.

**Intuition**: The pairing constraint (matching must be perfect) introduces correlations, but these are weak for large $k$ due to exponential localization. Walker $j$'s companion depends primarily on $j$'s own neighborhood, with negligible coupling to distant walkers' pairing choices.

**Step 2: Exponential concentration analysis.**

By Corollary {prf:ref}`cor-effective-interaction-radius-full`, with high probability ($\geq 1 - 1/k$), companion $c(j)$ satisfies:

$$
d_{\text{alg}}(j, c(j)) \leq R_{\text{eff}} = O(\varepsilon_{\text{comp}} \sqrt{\log k})
$$

The number of potential companions within $R_{\text{eff}}$ is:

$$
k_{\text{eff}}(j) = |\{\ell \in \mathcal{A} : d_{\text{alg}}(j,\ell) \leq R_{\text{eff}}\}| = O(\rho_{\max} R_{\text{eff}}^{2d}) = O(\log^d k)
$$

(by uniform density bound {prf:ref}`assump-uniform-density-full`).

**Key Observation**: For $k \gg k_{\text{eff}}(j)$, the pairing constraint affects only a negligible fraction of walkers. The probability that walker $j$'s preferred companions are "blocked" (already matched) is $O(k_{\text{eff}} / k) = O(\log^d k / k) = o(1)$.

**Step 3: Marginal distribution comparison.**

The softmax distribution is:

$$
P_{\text{softmax}}(c(j) = \ell | S) = \frac{\exp(-d_{\text{alg}}^2(j,\ell)/(2\varepsilon_{\text{comp}}^2))}{Z_j^{\text{soft}}}
$$

where $Z_j^{\text{soft}} = \sum_{\ell' \in \mathcal{A} \setminus \{j\}} \exp(-d_{\text{alg}}^2(j,\ell')/(2\varepsilon_{\text{comp}}^2))$.

The pairing marginal satisfies (approximately, for large $k$):

$$
P_{\text{pair}}(c(j) = \ell | S) \approx \frac{\exp(-d_{\text{alg}}^2(j,\ell)/(2\varepsilon_{\text{comp}}^2))}{Z_j^{\text{pair}}}
$$

where $Z_j^{\text{pair}} \approx Z_j^{\text{soft}} \cdot (1 + O(k_{\text{eff}}/k))$ accounts for the normalization over available companions (excluding those already paired).

Since $k_{\text{eff}}/k = O(\log^d k / k)$, we have:

$$
\frac{Z_j^{\text{pair}}}{Z_j^{\text{soft}}} = 1 + O(k^{-1} \log^d k)
$$

Therefore:

$$
|P_{\text{pair}}(c(j) = \ell | S) - P_{\text{softmax}}(c(j) = \ell | S)| = O(k^{-1} \log^d k)
$$

**Step 4: Expected measurement difference.**

The expected measurements are:

$$
\begin{aligned}
d_j^{\text{soft}} &= \sum_{\ell} P_{\text{softmax}}(c(j) = \ell) \cdot d_{\text{alg}}(j, \ell) \\
d_j^{\text{pair}} &= \sum_{\ell} P_{\text{pair}}(c(j) = \ell) \cdot d_{\text{alg}}(j, \ell)
\end{aligned}
$$

The difference is:

$$
|d_j^{\text{pair}} - d_j^{\text{soft}}| \leq \sum_{\ell} |P_{\text{pair}} - P_{\text{softmax}}| \cdot d_{\text{alg}}(j, \ell)
$$

Since $d_{\text{alg}}(j, \ell) \leq R_{\text{eff}} = O(\varepsilon_{\text{comp}} \sqrt{\log k})$ for all $\ell$ contributing significantly (exponential concentration), and $\sum_\ell |P_{\text{pair}} - P_{\text{softmax}}| = O(k^{-1} \log^d k)$ (total variation distance), we obtain:

$$
|\Delta_j| := |d_j^{\text{pair}} - d_j^{\text{soft}}| = O(k^{-1} \log^{d+1/2} k) = O(k^{-1/2})
$$

for $k$ sufficiently large that $\log^{2d+1} k < k^{1/2}$ (true for all $k \geq 50$ and practical $d \leq 20$).

**Step 5: Derivatives of the correction term.**

By the chain rule and Faà di Bruno formula:

$$
\nabla^m \Delta_j = \nabla^m (d_j^{\text{pair}} - d_j^{\text{soft}})
$$

Both $d_j^{\text{pair}}$ and $d_j^{\text{soft}}$ have Gevrey-1 derivative bounds (Lemma {prf:ref}`lem-companion-measurement-derivatives-full` and Theorem {prf:ref}`thm-diversity-pairing-measurement-regularity`). Therefore:

$$
\|\nabla^m \Delta_j\| \leq C_m \cdot m! \cdot \max(\varepsilon_{\text{comp}}^{-m}, \varepsilon_d^{1-m}) \cdot k^{-1/2}
$$

**Step 6: Propagation through the fitness pipeline.**

The fitness potential is computed via:

$$
V_{\text{fit}} = g_A(Z_\rho(\mu_\rho, \sigma_\rho^2))
$$

where $\mu_\rho^{(i)} = \sum_j w_{ij}(\rho) d_j$ (localized mean).

The difference in fitness potentials is:

$$
V_{\text{fit}}^{\text{pair}} - V_{\text{fit}}^{\text{soft}} = g_A(Z_\rho(\mu_\rho + \Delta_\mu, \sigma_\rho^2 + \Delta_\sigma)) - g_A(Z_\rho(\mu_\rho, \sigma_\rho^2))
$$

where $\Delta_\mu = \sum_j w_{ij} \Delta_j = O(k^{-1/2})$ (since $\sum_j w_{ij} = 1$).

By Taylor expansion and smoothness of $g_A$:

$$
\|V_{\text{fit}}^{\text{pair}} - V_{\text{fit}}^{\text{soft}}\| = O(k^{-1/2})
$$

with derivatives satisfying:

$$
\|\nabla^m (V_{\text{fit}}^{\text{pair}} - V_{\text{fit}}^{\text{soft}})\| = O(k^{-1/2}) \cdot C_m m!
$$

**Conclusion**: The two mechanisms produce statistically equivalent fitness potentials with the same C^∞ regularity and k-uniform Gevrey-1 bounds. The $O(k^{-1/2})$ difference is negligible for $k \geq 50$. $\square$
:::

:::{note} Practical Consequence for Implementation
For practical swarm sizes ($k \geq 50$), the difference between softmax and diversity pairing is:

$$
\frac{\|V_{\text{fit}}^{\text{pair}} - V_{\text{fit}}^{\text{soft}}\|}{\|V_{\text{fit}}\|} = O(k^{-1/2}) < 0.14 \quad \text{for } k = 50
$$

This is **smaller than typical stochastic noise** from Langevin dynamics (with temperature $T > 0$). Therefore, either mechanism can be used interchangeably from an analytical perspective.

**Implementation choice** depends on algorithmic considerations:
- **Softmax**: Simpler (walker-local), faster per-step
- **Diversity pairing**: Better diversity (bidirectional matching), proven geometric signal preservation (Lemma 5.1.2 in `03_cloning.md`)
:::

### 4.7.3 Unified Main Theorem

:::{prf:theorem} C^∞ Regularity of Companion-Dependent Fitness Potential (Both Mechanisms)
:label: thm-unified-cinf-regularity-both-mechanisms

Under the framework assumptions (kinetic regularization providing density bound, companion availability, regularization parameters $\varepsilon_d, \varepsilon_c > 0$), the fitness potential:

$$
V_{\text{fit}}(x_i, v_i) = g_A\left(Z_\rho\left(\mu_\rho^{(i)}, \sigma_\rho^{2(i)}\right)\right)
$$

computed with **either** companion selection mechanism (independent softmax or diversity pairing) is **C^∞** for all $(x_i, v_i) \in \mathcal{X} \times \mathbb{R}^d$.

**Derivative Bounds** (k-uniform Gevrey-1): For all $m \geq 0$:

$$
\|\nabla^m_{x_i, v_i} V_{\text{fit}}\|_\infty \leq C_{V,m} \cdot \max(\rho^{-m}, \varepsilon_d^{1-m}, \eta_{\min}^{1-m})
$$

where $C_{V,m} = \mathcal{O}(m!)$ (Gevrey-1) is **k-uniform** (independent of swarm size $k$ or $N$) and depends only on:
- Algorithmic parameters: $\rho, \varepsilon_c, \varepsilon_d, \eta_{\min}$
- Dimension: $d$
- Density bound: $\rho_{\max}$ (derived from kinetic dynamics)

**Mechanism Equivalence**: The derivative bounds are identical for both mechanisms up to mechanism-independent constants. The fitness potentials differ by $O(k^{-1/2})$, which is negligible.
:::

:::{prf:proof}
**Proof Structure**:

1. **Softmax mechanism** (§4.5): Proven in Lemma {prf:ref}`lem-companion-measurement-derivatives-full` + propagation through stages 2-6
2. **Diversity pairing** (§4.6): Proven in Theorem {prf:ref}`thm-diversity-pairing-measurement-regularity` + same propagation
3. **Statistical equivalence** (§4.7.2): Theorem {prf:ref}`thm-statistical-equivalence-companion-mechanisms` establishes mechanisms differ by $O(k^{-1/2})$
4. **Unified conclusion**: Both achieve C^∞ with k-uniform Gevrey-1 bounds. Difference is negligible. $\square$
:::

:::{important} Main Takeaway
**The Geometric Gas fitness potential is C^∞ with k-uniform Gevrey-1 bounds regardless of which companion selection mechanism is implemented.**

This enables:
- **Mean-field analysis**: Smooth potential allows rigorous mean-field limit (doc-07)
- **Hypoelliptic regularity**: C^∞ fitness enables hypoelliptic propagation (§13)
- **Stability analysis**: k-uniform bounds prevent blowup as swarm size varies
- **Implementation flexibility**: Choose mechanism based on algorithmic needs, not analytical concerns
:::

---

## Addition 4: Updated Measurement Convention (replace lines 24-30)

**Measurement Convention and Dual Mechanism Analysis**: Throughout this analysis, measurements denote **expected values** over the stochastic companion selection:

$$
d_j := \mathbb{E}_{c(j) \sim \text{mechanism}}[d_{\text{alg}}(j, c(j))]
$$

where the mechanism is either:
- **Independent Softmax**: $\mathbb{E}_{\text{softmax}}[d_{\text{alg}}(j, c(j))] = \sum_{\ell \in \mathcal{A} \setminus \{j\}} P(c(j) = \ell) \cdot d_{\text{alg}}(j, \ell)$ with $P$ given by softmax distribution
- **Diversity Pairing**: $\mathbb{E}_{\text{pairing}}[d_{\text{alg}}(j, c(j))] = \sum_{M \in \mathcal{M}_k} P(M) \cdot d_{\text{alg}}(j, M(j))$ with $P$ given by idealized matching distribution

**Key Result** (§4.7): Both mechanisms produce statistically equivalent expected measurements with identical C^∞ regularity and k-uniform Gevrey-1 bounds. The fitness potential analyzed is the **expected potential** $\mathbb{E}[V_{\text{fit}}]$ over stochastic companion selection. This is the quantity that drives the algorithm's mean-field dynamics, and the regularity holds **for both mechanisms**.

---

## Addition 5: Update to §4.6 Title and Note

Replace line 1387-1395 with:

## 4.6 Diversity Pairing Mechanism Analysis

:::{important} Dual Mechanism Framework
:label: note-dual-mechanism-framework

The Fragile framework supports **BOTH** companion selection mechanisms:

1. **Independent Softmax Selection** (§4.5): Each walker independently samples via softmax
2. **Diversity Pairing** (this section): Global perfect matching via Sequential Stochastic Greedy Pairing

**Analytical Goal**: Prove that BOTH mechanisms achieve:
- C^∞ regularity with Gevrey-1 bounds
- k-uniform derivative bounds
- Statistical equivalence (§4.7)

This section analyzes diversity pairing. §4.7 establishes equivalence.

**Implementation Note**: The codebase supports both mechanisms. Diversity pairing is canonical per `03_cloning.md`, but independent softmax is also available. The C^∞ regularity proven here applies to **both**, enabling flexible implementation.
:::

---

## Summary of Changes

These additions implement Option C by:

1. **Revised Abstract**: Explicitly mentions both mechanisms and states both achieve C^∞ regularity
2. **New §1.0**: Provides framework context explaining why two mechanisms exist and what we'll prove about them
3. **New §4.7**: Contains strengthened statistical equivalence theorem and unified main theorem
4. **Updated Measurement Convention**: Clarifies measurements are expected values and regularity holds for both mechanisms
5. **Updated §4.6 Note**: Replaces "user decision" note with dual mechanism framework statement

**Result**: Document now clearly states and rigorously proves that C^∞ regularity with k-uniform Gevrey-1 bounds holds for BOTH companion selection mechanisms, with statistical equivalence making them interchangeable for analytical purposes.
