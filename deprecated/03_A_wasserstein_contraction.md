## DEPRECATED: Wasserstein-2 Contraction for the Cloning Operator via Coupling

**⚠️ DEPRECATED - DO NOT USE ⚠️**

**Status:** This document contains fundamental errors and has been superseded.

**Replacement:** See [03_wasserstein_contraction_complete.md](03_wasserstein_contraction_complete.md) for the complete, correct proof.

**Known Issues:**
- Incomplete case analysis
- Missing critical lemmas (Outlier Alignment)
- Incorrect proof structure

**Historical Note:** This was an initial draft. Kept for historical reference only.

---

## Wasserstein-2 Contraction for the Cloning Operator via Coupling (ORIGINAL DRAFT)

**This proof uses the direct coupling method as outlined by Gemini's expert guidance.**

---

### Preliminary: Synchronous Coupling of the Cloning Operator

:::{prf:definition} Synchronous Coupling for Paired Walkers
:label: def-synchronous-cloning-coupling

For two walkers $(x, y) \in \mathbb{R}^d \times \mathbb{R}^d$ in swarms $(S_1, S_2)$, the **synchronous cloning coupling** $\Psi_{\text{sync}}$ evolves them to $(x', y')$ using shared randomness:

1. **Companion Selection:** Both walkers use the same random index $I \sim \text{Unif}\{1, \ldots, N\}$ to select their companions:
   - Walker $x$ selects companion $c_x$ from swarm $S_1$ using index $I$
   - Walker $y$ selects companion $c_y$ from swarm $S_2$ using index $I$

2. **Cloning Threshold:** The same threshold $T \sim \text{Unif}(0, p_{\max})$ determines cloning for both:
   - Walker $x$ clones if $T < S_1(x, c_x)$ (cloning score in swarm 1)
   - Walker $y$ clones if $T < S_2(y, c_y)$ (cloning score in swarm 2)

3. **Gaussian Jitter:** The same noise vector $\zeta \sim \mathcal{N}(0, \delta^2 I)$ is applied:
   - If $x$ clones: $x' = c_x + \zeta$
   - If $y$ clones: $y' = c_y + \zeta$
   - If a walker doesn't clone: it stays at its current position
:::

:::{prf:remark}
This synchronous coupling maximizes the correlation between the evolution of paired walkers by eliminating all extrinsic sources of randomness. The only source of desynchronization is the difference in cloning scores $S_1(x, c_x)$ vs $S_2(y, c_y)$, which depends on the swarms' differing configurations.
:::

---

### Lemma 1: One-Step Distance Evolution for a Coupled Pair

:::{prf:lemma} Coupled Pair Distance Contraction
:label: lem-coupled-pair-contraction

For any pair of walkers $(x, y)$ evolved under the synchronous cloning coupling {prf:ref}`def-synchronous-cloning-coupling`, the expected squared distance after cloning satisfies:

$$
\mathbb{E}[\|x' - y'\|^2 \mid x, y] \leq (1 - \alpha \cdot p_{\text{joint}}(x, y)) \|x - y\|^2 + C_{\text{pair}}
$$

where:
- $p_{\text{joint}}(x, y)$ is the probability that at least one walker clones
- $\alpha > 0$ is a contraction strength parameter
- $C_{\text{pair}} = O(\delta^2 + D_{\text{valid}}^2)$ is a noise/domain constant
:::

:::{prf:proof}

**Step 1: Case decomposition**

Under the synchronous coupling, there are four possible outcomes based on the shared threshold $T$:

1. **Persist-Persist (PP):** Neither walker clones → $(x', y') = (x, y)$
2. **Clone-Clone (CC):** Both walkers clone → $(x', y') = (c_x + \zeta, c_y + \zeta)$
3. **Clone-Persist (CP):** Only $x$ clones → $(x', y') = (c_x + \zeta, y)$
4. **Persist-Clone (PC):** Only $y$ clones → $(x', y') = (x, c_y + \zeta)$

Let $p_{CC}, p_{CP}, p_{PC}, p_{PP}$ denote the probabilities of these cases.

**Step 2: Distance analysis for each case**

*Case PP:* $\|x' - y'\|^2 = \|x - y\|^2$ (no change)

*Case CC:*
$$
\|x' - y'\|^2 = \|(c_x + \zeta) - (c_y + \zeta)\|^2 = \|c_x - c_y\|^2
$$

The noise cancels due to synchronization! The distance becomes the distance between companions.

By the companion selection mechanism and the Safe Harbor axiom, companions are drawn from regions of high fitness, which are spatially concentrated near the barycenter. Therefore:

$$
\mathbb{E}[\|c_x - c_y\|^2 \mid \text{both clone}] \leq \rho \|x - y\|^2 + C_c
$$

for some $\rho < 1$ when $\|x - y\|$ is large (since companions are pulled toward the barycenter, reducing inter-swarm spread).

*Case CP:*
$$
\|x' - y'\|^2 = \|c_x + \zeta - y\|^2
$$

If $x$ is far from the barycenter of $S_1$ (which is likely when $\|x - y\|$ is large), the companion $c_x$ will be closer to the barycenter. The noise $\zeta$ has bounded second moment $\delta^2 d$.

By the triangle inequality and the Safe Harbor axiom:
$$
\mathbb{E}[\|c_x + \zeta - y\|^2] \leq 2\mathbb{E}[\|c_x - \bar{x}_1\|^2] + 2\mathbb{E}[\|\bar{x}_1 - y\|^2] + O(\delta^2)
$$

where $\bar{x}_1$ is the barycenter of $S_1$.

*Case PC:* Symmetric to CP.

**Step 3: Weighted average**

$$
\mathbb{E}[\|x' - y'\|^2] = p_{PP}\|x - y\|^2 + p_{CC}\mathbb{E}[\|c_x - c_y\|^2] + p_{CP}\mathbb{E}[\|c_x + \zeta - y\|^2] + p_{PC}\mathbb{E}[\|x - c_y - \zeta\|^2]
$$

**Step 4: Connection to Keystone Principle**

The **Keystone Principle** (Chapter 8, [03_cloning.md](03_cloning.md)) guarantees that when a walker is far from its swarm's barycenter, it has high probability of being cloned. Specifically:

- If $\|x - \bar{x}_1\|^2$ is large, then $x$ is in the high-error set, and by Theorem 8.2, $x$ has cloning probability $p_x \geq p_{\min} > 0$
- Similarly for $y$ and swarm $S_2$

When $\|x - y\|$ is large, at least one of the following must hold:
- $\|x - \bar{x}_1\|$ is large (walker $x$ is an outlier in swarm 1)
- $\|y - \bar{x}_2\|$ is large (walker $y$ is an outlier in swarm 2)
- $\|\bar{x}_1 - \bar{x}_2\|$ is large (the barycenters are far apart)

In all cases, the Keystone mechanism ensures $p_{\text{joint}}(x, y) := p_{CC} + p_{CP} + p_{PC} \geq p_{\min}$.

**Step 5: Bounding the overall expectation**

The CC case provides strong contraction ($\rho < 1$). The CP/PC cases provide moderate contraction when the walker being cloned is far from its barycenter. The PP case maintains distance.

Combining with appropriate bounds on each case:

$$
\mathbb{E}[\|x' - y'\|^2] \leq (1 - p_{PP}) \cdot [\rho \|x - y\|^2 + C'] + p_{PP} \|x - y\|^2
$$

$$
= (1 - p_{PP}(1 - \rho)) \|x - y\|^2 + (1 - p_{PP}) C'
$$

Since $p_{PP} = 1 - p_{\text{joint}}$:

$$
\mathbb{E}[\|x' - y'\|^2] \leq (1 - \alpha p_{\text{joint}}) \|x - y\|^2 + C_{\text{pair}}
$$

where $\alpha = 1 - \rho$ captures the contraction strength in the Clone-Clone case.

**Q.E.D.**
:::

---

### Lemma 2: Lower Bound on Joint Cloning Probability

:::{prf:lemma} Keystone-Based Joint Cloning Guarantee
:label: lem-keystone-joint-cloning

For walkers $(x, y)$ drawn from an optimal coupling $\gamma$ realizing $W_2^2(\mu_{S_1}, \mu_{S_2})$, the joint cloning probability is bounded below:

$$
\mathbb{E}_\gamma[p_{\text{joint}}(x, y)] \geq p_{\min} > 0
$$

where $p_{\min}$ is N-uniform and depends only on the framework axioms.
:::

:::{prf:proof}

**Step 1: High-transport-cost pairs**

By the definition of the Wasserstein distance, the optimal coupling $\gamma$ minimizes:

$$
W_2^2(\mu_{S_1}, \mu_{S_2}) = \int \|x - y\|^2 \, d\gamma(x, y)
$$

If this integral is large, there must exist a non-negligible fraction of pairs $(x, y)$ with large $\|x - y\|^2$.

**Step 2: Variance bound for large distances**

For any pair $(x, y)$ with $\|x - y\| \geq R$ for some threshold $R$:

$$
\|x - y\|^2 = \|(x - \bar{x}_1) - (y - \bar{x}_2) + (\bar{x}_1 - \bar{x}_2)\|^2
$$

By the triangle inequality, if $\|x - y\|$ is large, at least one of the following holds:
- $\|x - \bar{x}_1\| \geq R/3$ (walker $x$ is far from its barycenter)
- $\|y - \bar{x}_2\| \geq R/3$ (walker $y$ is far from its barycenter)
- $\|\bar{x}_1 - \bar{x}_2\| \geq R/3$ (barycenters are far apart)

**Step 3: Application of Keystone Principle**

From the Keystone Principle (Theorem 8.2 in [03_cloning.md](03_cloning.md)):

- If $\|x - \bar{x}_1\|^2 \geq \varepsilon_H$ (high-error threshold), then $x$ has cloning probability $\geq p_{\min}^{(1)}$
- If $\|y - \bar{x}_2\|^2 \geq \varepsilon_H$, then $y$ has cloning probability $\geq p_{\min}^{(2)}$

For pairs with large $\|x - y\|$, at least one walker is likely in the high-error set, ensuring:

$$
p_{\text{joint}}(x, y) \geq \min(p_{\min}^{(1)}, p_{\min}^{(2)}) =: p_{\min}
$$

**Step 4: Average over the coupling**

Averaging over all pairs in the optimal coupling $\gamma$:

$$
\mathbb{E}_\gamma[p_{\text{joint}}(x, y)] \geq \int p_{\text{joint}}(x, y) \, d\gamma(x, y) \geq p_{\min}
$$

The N-uniformity follows from the N-uniformity of the Keystone Principle.

**Q.E.D.**
:::

---

### Main Theorem: Wasserstein Contraction

:::{prf:theorem} Wasserstein-2 Contraction for the Cloning Operator
:label: thm-cloning-wasserstein-contraction-coupling

Under the foundational axioms (Chapter 4, [03_cloning.md](03_cloning.md)), the cloning operator $\Psi_{\text{clone}}$ contracts the 2-Wasserstein distance between empirical measures:

$$
\mathbb{E}[W_2^2(\mu_{S_1'}, \mu_{S_2'})] \leq (1 - \kappa_W) W_2^2(\mu_{S_1}, \mu_{S_2}) + C_W
$$

where:
- $\kappa_W = \alpha \cdot p_{\min} > 0$ is the Wasserstein contraction rate
- $C_W < \infty$ is a state-independent constant
- Both $\kappa_W$ and $C_W$ are N-uniform
:::

:::{prf:proof}

**Step 1: Optimal coupling and induced coupling**

Let $\gamma \in \Gamma(\mu_{S_1}, \mu_{S_2})$ be an optimal coupling realizing:

$$
W_2^2(\mu_{S_1}, \mu_{S_2}) = \int \|x - y\|^2 \, d\gamma(x, y)
$$

Apply the synchronous cloning coupling {prf:ref}`def-synchronous-cloning-coupling` to each pair $(x, y) \sim \gamma$. This induces a coupling $\gamma'$ between the post-cloning measures $\mu_{S_1'}, \mu_{S_2'}$.

**Step 2: Upper bound via induced coupling**

By the definition of the Wasserstein distance as an infimum over all couplings:

$$
W_2^2(\mu_{S_1'}, \mu_{S_2'}) \leq \int \|x' - y'\|^2 \, d\gamma'(x', y')
$$

The right-hand side is exactly the expected cost of the induced coupling.

**Step 3: Application of coupled pair contraction**

From Lemma {prf:ref}`lem-coupled-pair-contraction`:

$$
\mathbb{E}[\|x' - y'\|^2 \mid x, y] \leq (1 - \alpha p_{\text{joint}}(x, y)) \|x - y\|^2 + C_{\text{pair}}
$$

Integrating over $\gamma$:

$$
\mathbb{E}\left[\int \|x' - y'\|^2 \, d\gamma'(x', y')\right] = \int \mathbb{E}[\|x' - y'\|^2 \mid x, y] \, d\gamma(x, y)
$$

$$
\leq \int (1 - \alpha p_{\text{joint}}(x, y)) \|x - y\|^2 \, d\gamma(x, y) + C_{\text{pair}}
$$

**Step 4: Bounding the joint cloning probability**

From Lemma {prf:ref}`lem-keystone-joint-cloning`:

$$
\int p_{\text{joint}}(x, y) \, d\gamma(x, y) \geq p_{\min}
$$

Therefore:

$$
\int (1 - \alpha p_{\text{joint}}(x, y)) \|x - y\|^2 \, d\gamma(x, y) \leq (1 - \alpha p_{\min}) \int \|x - y\|^2 \, d\gamma(x, y)
$$

$$
= (1 - \alpha p_{\min}) W_2^2(\mu_{S_1}, \mu_{S_2})
$$

**Step 5: Final contraction bound**

Combining Steps 2, 3, and 4:

$$
\mathbb{E}[W_2^2(\mu_{S_1'}, \mu_{S_2'})] \leq (1 - \alpha p_{\min}) W_2^2(\mu_{S_1}, \mu_{S_2}) + C_{\text{pair}}
$$

Setting $\kappa_W = \alpha p_{\min}$ and $C_W = C_{\text{pair}}$ yields the desired result.

**N-uniformity:** Both $\alpha$ and $p_{\min}$ are N-uniform by the Keystone Principle, hence $\kappa_W$ and $C_W$ are N-uniform.

**Q.E.D.**
:::

---

### Connection to Position Variance Contraction

:::{prf:remark} Variance Contraction as a Corollary
The position variance contraction result (Theorem 10.3 in [03_cloning.md](03_cloning.md)) can be viewed as a **corollary** of this Wasserstein contraction result.

For a single swarm $S$, the variance is related to the Wasserstein distance to the barycenter:

$$
\text{Var}_x(S) = \frac{1}{N}\sum_i \|x_i - \bar{x}\|^2 = W_2^2(\mu_S, \delta_{\bar{x}})
$$

where $\delta_{\bar{x}}$ is the Dirac measure at the barycenter.

Applying the Wasserstein contraction with $\mu_{S_2} = \delta_{\bar{x}}$ (which is invariant under cloning toward the barycenter):

$$
\mathbb{E}[\text{Var}_x(S')] \leq (1 - \kappa_W) \text{Var}_x(S) + C_W
$$

This provides an alternative derivation of variance contraction from first principles.
:::

---

## Summary

**What we proved:**
- The cloning operator contracts the 2-Wasserstein distance via a **direct coupling argument**
- The proof relies fundamentally on the **Keystone Principle** to guarantee sufficient cloning of high-distance pairs
- The result is **N-uniform** and depends only on the foundational axioms

**Key innovation:**
- Using synchronous coupling to maximize correlation between paired walkers
- The Gaussian noise **cancels** in the Clone-Clone case due to synchronization
- The Keystone mechanism converts geometric error (large $\|x - y\|$) into corrective action (high cloning probability)

**This proof is ready for submission to Gemini for verification.**
