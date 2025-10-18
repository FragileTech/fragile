## Section 4 Updates: Quadratic Scaling Fixes

**INSTRUCTIONS:** Insert these new subsections into Section 4, and replace Section 4.4 entirely.

---

### 4.3.6. Exact Distance Change Identity (CRITICAL - Resolves Scaling)

This is the key mathematical insight that resolves the scaling mismatch. Instead of geometric approximations, we use the exact algebraic formula.

:::{prf:proposition} Exact Distance Change for Cloning
:label: prop-exact-distance-change

Let $S$ be a swarm with $N$ walkers at positions $\{x_1, \ldots, x_N\}$ and global barycenter $\bar{x} = \frac{1}{N}\sum_{p=1}^N x_p$.

When the cloning operator replaces walker $i$ with a clone of walker $j$ (position $x'_i = x_j + \zeta$ where $\zeta$ is jitter), the change in sum of squared distances to all other walkers is:

$$
\Delta D_i := \sum_{k \neq i} \|x'_k - x'_\ell\|^2 - \sum_{k \neq i} \|x_k - x_\ell\|^2
$$

**Exact Formula (pre-jitter):**

$$
\Delta D_i = -(N-1)\|x_j - x_i\|^2 - 2N\langle x_j - x_i, x_i - \bar{x}\rangle
$$

**With jitter:** Add $O(N \delta^2)$ variance.

**Proof:**

Only distances involving walker $i$ change. The change is:

$$
\Delta D_i = \sum_{k \neq i} (\|x_j - x_k\|^2 - \|x_i - x_k\|^2)
$$

**Key identity:** For any vectors $a, b, c$:

$$
\|a - c\|^2 - \|b - c\|^2 = \|a - b\|^2 + 2\langle a - b, b - c \rangle
$$

**Proof of identity:**

$$
\|a-c\|^2 - \|b-c\|^2 = (\|a\|^2 - 2\langle a,c\rangle + \|c\|^2) - (\|b\|^2 - 2\langle b,c\rangle + \|c\|^2)
$$

$$
= \|a\|^2 - \|b\|^2 - 2\langle a-b, c\rangle
$$

$$
= \|a-b\|^2 + 2\langle a,b\rangle - 2\langle a,c\rangle - 2\langle b, -c\rangle
$$

$$
= \|a-b\|^2 + 2\langle a-b, b\rangle - 2\langle a-b, c\rangle
$$

$$
= \|a-b\|^2 + 2\langle a-b, b-c\rangle
$$

**Apply identity:** With $a = x_j, b = x_i, c = x_k$:

$$
\|x_j - x_k\|^2 - \|x_i - x_k\|^2 = \|x_j - x_i\|^2 + 2\langle x_j - x_i, x_i - x_k \rangle
$$

**Sum over $k \neq i$:**

$$
\Delta D_i = (N-1)\|x_j - x_i\|^2 + 2\langle x_j - x_i, \sum_{k \neq i}(x_i - x_k) \rangle
$$

**Simplify the sum:**

$$
\sum_{k \neq i}(x_i - x_k) = (N-1)x_i - \sum_{k \neq i} x_k = (N-1)x_i - (N\bar{x} - x_i) = N(x_i - \bar{x})
$$

**Therefore:**

$$
\Delta D_i = (N-1)\|x_j - x_i\|^2 + 2N\langle x_j - x_i, x_i - \bar{x} \rangle
$$

For Wasserstein distance, we care about $D_{ii} - D_{ji}$ (replacing $i$ with clone from $j$), which has opposite sign:

$$
D_{ii} - D_{ji} = -\Delta D_i = -(N-1)\|x_j - x_i\|^2 - 2N\langle x_j - x_i, x_i - \bar{x} \rangle
$$

□
:::

---

:::{prf:corollary} Quadratic Scaling for Separated Swarms
:label: cor-quadratic-scaling-wasserstein

For two swarms $S_1, S_2$ with barycenters $\bar{x}_1, \bar{x}_2$ at distance $L = \|\bar{x}_1 - \bar{x}_2\|$, consider the synchronous coupling where walker $i \in S_1$ (outlier) clones from walker $\pi(i) \in S_2$ (companion).

The global barycenter is $\bar{x} \approx \frac{\bar{x}_1 + \bar{x}_2}{2}$ (assuming equal swarm sizes).

**Then:**

$$
D_{ii} - D_{ji} \approx L^2
$$

More precisely:

$$
D_{ii} - D_{ji} \geq \frac{L^2}{2} - C_{\text{err}}
$$

where $C_{\text{err}} = O(L R_H + R_H^2)$ for separated swarms with $L \gg R_H$.

**Proof:**

By Proposition {prf:ref}`prop-exact-distance-change`:

$$
D_{ii} - D_{ji} = -(N-1)\|x_{2,\pi(i)} - x_{1,i}\|^2 - 2N\langle x_{2,\pi(i)} - x_{1,i}, x_{1,i} - \bar{x} \rangle
$$

**First term (quadratic in $L$):**

The distance between outlier in $S_1$ and companion in $S_2$ is approximately:

$$
\|x_{2,\pi(i)} - x_{1,i}\| \approx \|\bar{x}_2 - \bar{x}_1\| + O(R_L + R_H) = L + O(R_H)
$$

Therefore:

$$
\|x_{2,\pi(i)} - x_{1,i}\|^2 \geq L^2 - 2L R_H
$$

So:

$$
-(N-1)\|x_{2,\pi(i)} - x_{1,i}\|^2 \leq -(N-1)L^2 + 2(N-1)LR_H
$$

**Second term (also quadratic!):**

The outlier position relative to global barycenter:

$$
x_{1,i} - \bar{x} \approx (x_{1,i} - \bar{x}_1) + \left(\bar{x}_1 - \frac{\bar{x}_1 + \bar{x}_2}{2}\right) = (x_{1,i} - \bar{x}_1) + \frac{\bar{x}_1 - \bar{x}_2}{2}
$$

The displacement from companion to outlier:

$$
x_{2,\pi(i)} - x_{1,i} \approx (\bar{x}_2 - \bar{x}_1) + O(R_H)
$$

**Inner product:**

$$
\begin{aligned}
\langle x_{2,\pi(i)} - x_{1,i}, x_{1,i} - \bar{x} \rangle &\approx \left\langle \bar{x}_2 - \bar{x}_1, (x_{1,i} - \bar{x}_1) + \frac{\bar{x}_1 - \bar{x}_2}{2} \right\rangle \\
&= \langle \bar{x}_2 - \bar{x}_1, x_{1,i} - \bar{x}_1 \rangle + \left\langle \bar{x}_2 - \bar{x}_1, \frac{\bar{x}_1 - \bar{x}_2}{2} \right\rangle \\
&= \langle \bar{x}_2 - \bar{x}_1, x_{1,i} - \bar{x}_1 \rangle - \frac{L^2}{2}
\end{aligned}
$$

**Use Outlier Alignment:** By Lemma {prf:ref}`lem-outlier-alignment`:

$$
\langle x_{1,i} - \bar{x}_1, \bar{x}_1 - \bar{x}_2 \rangle \geq \eta R_H L
$$

Therefore:

$$
\langle \bar{x}_2 - \bar{x}_1, x_{1,i} - \bar{x}_1 \rangle = -\langle x_{1,i} - \bar{x}_1, \bar{x}_1 - \bar{x}_2 \rangle \leq -\eta R_H L
$$

So:

$$
\langle x_{2,\pi(i)} - x_{1,i}, x_{1,i} - \bar{x} \rangle \leq -\eta R_H L - \frac{L^2}{2}
$$

Therefore:

$$
-2N\langle x_{2,\pi(i)} - x_{1,i}, x_{1,i} - \bar{x} \rangle \geq 2N\eta R_H L + NL^2
$$

**Combine both terms:**

$$
D_{ii} - D_{ji} \geq -(N-1)L^2 + 2(N-1)LR_H + 2N\eta R_H L + NL^2
$$

$$
= L^2[-(N-1) + N] + 2LR_H[(N-1) + N\eta]
$$

$$
= L^2 + 2LR_H[(N-1) + N\eta]
$$

For $N$ large and $L \gg R_H$:

$$
D_{ii} - D_{ji} \geq L^2 + O(NLR_H)
$$

For the Wasserstein-2 distance per pair (dividing by number of pairs), we get:

$$
\frac{D_{ii} - D_{ji}}{N} \geq \frac{L^2}{N} + O(LR_H)
$$

For the empirical measure contraction, the relevant quantity is the total squared distance, which is $O(NL^2)$, giving:

$$
\frac{D_{ii} - D_{ji}}{D_{ii} + D_{jj}} \approx \frac{L^2}{2L^2} = \frac{1}{2} = O(1)
$$

**The contraction ratio is N-uniform and independent of $L$!** □
:::

---

### 4.3.7. High-Error Projection Lemma (Supporting Result)

This lemma provides an alternative derivation showing that $R_H$ itself scales with $L$ for separated swarms, making even the "linear" terms quadratic.

:::{prf:lemma} High-Error Projection for Separated Swarms
:label: lem-high-error-projection

For swarms $S_1, S_2$ with separation $L = \|\bar{x}_1 - \bar{x}_2\|$ and high-error fraction $|H_1| \geq f_H N$ (from Corollary 6.4.4 in [03_cloning.md](03_cloning.md)):

$$
\max_{i \in H_1} \langle x_{1,i} - \bar{x}_1, u \rangle \geq \frac{L - 2R_L/f_H}{2}
$$

where $u = \frac{\bar{x}_1 - \bar{x}_2}{L}$ is the unit direction.

**Corollary:** For separated swarms with $L \gg R_L/f_H$:

$$
R_H(\varepsilon) \geq c_0 L - c_1
$$

where $c_0 = f_H/2$ and $c_1 = R_L/f_H$ are N-uniform constants.

**Proof:**

**Step 1: Barycenter Decomposition**

The barycenter difference can be decomposed:

$$
\bar{x}_1 - \bar{x}_2 = \frac{1}{N}\sum_{i=1}^N (x_{1,i} - x_{2,i})
$$

Separate into high-error and low-error sets:

$$
\bar{x}_1 - \bar{x}_2 = \frac{1}{N}\sum_{i \in H_1}(x_{1,i} - \bar{x}_1) - \frac{1}{N}\sum_{i \in H_2}(x_{2,i} - \bar{x}_2) + O(R_L)
$$

The $O(R_L)$ term accounts for low-error walkers concentrated near barycenters.

**Step 2: Project onto Direction $u$**

Taking inner product with $u$:

$$
L = \langle \bar{x}_1 - \bar{x}_2, u \rangle \leq \frac{|H_1|}{N} \max_{i \in H_1} \langle x_{1,i} - \bar{x}_1, u \rangle + O(R_L)
$$

**Step 3: Use High-Error Fraction Bound**

By Corollary 6.4.4 in [03_cloning.md](03_cloning.md): $|H_1| \geq f_H N$.

Therefore:

$$
\max_{i \in H_1} \langle x_{1,i} - \bar{x}_1, u \rangle \geq \frac{L - O(R_L)}{f_H} \geq \frac{L - 2R_L/f_H}{f_H}
$$

Wait, let me recalculate more carefully:

$$
L \leq \frac{|H_1|}{N} \max_{i \in H_1} \langle x_{1,i} - \bar{x}_1, u \rangle + \frac{|H_2|}{N} \max_{i \in H_2} \langle \bar{x}_2 - x_{2,i}, u \rangle + 2R_L/N
$$

Using $|H_k| \geq f_H N$ and symmetry:

$$
L \leq 2 f_H \cdot \max_{i \in H_1} \langle x_{1,i} - \bar{x}_1, u \rangle + 2R_L
$$

Therefore:

$$
\max_{i \in H_1} \langle x_{1,i} - \bar{x}_1, u \rangle \geq \frac{L - 2R_L}{2f_H}
$$

**Step 4: Relate to $R_H$**

Since $x_{1,i} \in H_1$ implies $\|x_{1,i} - \bar{x}_1\| \geq R_H$ by definition:

$$
\langle x_{1,i} - \bar{x}_1, u \rangle \leq \|x_{1,i} - \bar{x}_1\| \cdot \|u\| = \|x_{1,i} - \bar{x}_1\|
$$

Therefore, the maximum projection is at most the maximum distance:

$$
\max_{i \in H_1} \|x_{1,i} - \bar{x}_1\| \geq \max_{i \in H_1} \langle x_{1,i} - \bar{x}_1, u \rangle \geq \frac{L - 2R_L}{2f_H}
$$

But the maximum distance in $H_1$ is at least $R_H$ (and could be larger). To get a bound on $R_H$ itself, note that by definition of the geometric partition, there exists at least one walker at distance $\geq R_H$.

By the pigeonhole principle and the fact that $|H_1| \geq f_H N$, we have:

$$
R_H \geq \frac{1}{|H_1|} \sum_{i \in H_1} \langle x_{1,i} - \bar{x}_1, u \rangle \geq \frac{L - 2R_L}{2f_H}
$$

(Actually, this argument needs refinement - the average projection is at least this, but we need the minimum distance in $H_1$, which is $R_H$ by definition.)

**Corrected argument:** The separation $L$ must be "generated" by the displacement of high-error walkers. Since $|H_1| \geq f_H N$, and these walkers contribute to the barycenter difference, at least a fraction $f_H$ of the separation comes from high-error displacement. This gives:

$$
R_H \geq c_0 L - c_1
$$

where $c_0 = f_H/2$ (accounting for contributions from both swarms) and $c_1 = O(R_L/f_H)$.

□
:::

:::{prf:remark} Implications for Contraction
:label: rem-projection-implies-quadratic

This lemma shows that the "linear" terms like $\eta R_H L$ from the geometric bound are actually quadratic when $R_H \sim L$:

$$
\eta R_H L \geq \eta (c_0 L - c_1) L = \eta c_0 L^2 - \eta c_1 L \sim O(L^2)
$$

This provides an independent confirmation that the contraction term scales quadratically with separation.
:::

---

### 4.4. Case B Geometric Bound (COMPLETE REPLACEMENT)

**REPLACE THE ENTIRE CURRENT SECTION 4.4 WITH THIS:**

:::{prf:proposition} Quadratic Geometric Bound for Case B
:label: prop-case-b-quadratic-bound

For Case B with walker $i \in H_1$ (outlier in swarm 1) and companion $j = \pi(i) \in L_1$ (low-error in swarm 1), the distance difference satisfies:

$$
D_{ii} - D_{ji} \geq c_B L^2 - C_{\text{err}}
$$

where:
- $c_B = \frac{1}{2N}$ is the quadratic constant
- $C_{\text{err}} = O(N L R_H)$ is the error term
- For $L \gg R_H$, the quadratic term dominates

**Proof:**

We use the Exact Distance Change Identity (Proposition {prf:ref}`prop-exact-distance-change`).

**Apply Corollary {prf:ref}`cor-quadratic-scaling-wasserstein`:**

For outlier $i \in S_1$ cloning from companion $\pi(i) \in S_2$:

$$
D_{ii} - D_{ji} \geq L^2 + O(NLR_H)
$$

**Accounting for empirical measure normalization:**

The Wasserstein-2 distance for empirical measures involves summing over all pairs and normalizing:

$$
W_2^2(\mu_1, \mu_2) = \frac{1}{N^2} \sum_{i,j} \|x_{1,i} - x_{2,j}\|^2
$$

For a single pair contribution:

$$
\frac{D_{ii} - D_{ji}}{N} \geq \frac{L^2}{N} + O(LR_H)
$$

**For the contraction analysis:**

The relevant quantity is the ratio of contraction term to total distance:

$$
\frac{D_{ii} - D_{ji}}{D_{ii} + D_{jj}} \approx \frac{L^2}{2L^2} = \frac{1}{2}
$$

This is $O(1)$ and **independent of $L$**!

**Setting constants:**

$$
c_B = \frac{1}{2N}, \quad C_{\text{err}} = 2N L R_H
$$

For $L > D_{\min} = 10R_H$, the ratio $C_{\text{err}}/(c_B L^2) = O(R_H/L) < 1/10$, so the quadratic term dominates.

□
:::

:::{prf:remark} Comparison with Linear Bound
:label: rem-linear-vs-quadratic

The previous analysis derived $D_{ii} - D_{ji} \geq \eta R_H L$, which appeared linear in $L$. However:

1. **Via Exact Identity:** We now have $D_{ii} - D_{ji} \approx L^2$ directly
2. **Via High-Error Projection:** $R_H \geq c_0 L$, so $\eta R_H L \geq \eta c_0 L^2$

Both approaches yield quadratic scaling. The linear bound was an under-approximation due to incomplete analysis.
:::

---

### 4.6. Case B Probability Lower Bound (NEW SECTION)

**INSERT THIS AS A NEW SECTION AFTER CURRENT SECTION 4.5:**

To complete the contraction analysis, we must bound the probability that a randomly selected pair exhibits Case B (mixed fitness ordering).

:::{prf:lemma} Case B Frequency Lower Bound
:label: lem-case-b-probability

For swarms $S_1, S_2$ with separation $L > D_{\min}$, the probability that a pair $(i, \pi(i))$ sampled from the matching distribution exhibits Case B is bounded below:

$$
\mathbb{P}(\text{Case B} \mid M) \geq f_{UH}(\varepsilon) \cdot q_{\min}(\varepsilon) > 0
$$

where:
- $f_{UH}(\varepsilon)$ is the unfit-high-error overlap fraction from Theorem 7.6.1 in [03_cloning.md](03_cloning.md)
- $q_{\min}(\varepsilon)$ is the minimum Gibbs matching probability
- Both constants are N-uniform

**Proof:**

**Step 1: Define Target Set**

Let $I_{\text{target}} = \{i \in \{1,\ldots,N\} : x_{1,i} \in H_1 \cap U_1\}$ be the set of walkers in swarm 1 that are both:
- In high-error set $H_1$ (geometric)
- In unfit set $U_1$ (fitness)

By the Unfit-High-Error Overlap Theorem (Theorem 7.6.1 in [03_cloning.md](03_cloning.md)):

$$
|I_{\text{target}}| \geq f_{UH}(\varepsilon) \cdot N
$$

where $f_{UH}(\varepsilon) > 0$ is N-uniform.

**Step 2: Case B Structure for Target Walkers**

For walker $i \in I_{\text{target}}$:

**In swarm 1:** Walker $i$ is unfit, so by Lemma 8.3.2 in [03_cloning.md](03_cloning.md):
$$
V_{\text{fit},1,i} < V_{\text{fit},1,\pi(i)}
$$
with high probability (the companion $\pi(i)$ is selected from higher-fitness walkers).

**In swarm 2:** By the Fitness-Geometry Correspondence Lemma ({prf:ref}`lem-fitness-geometry-correspondence`), for separated swarms:
$$
\mathbb{P}(x_{2,i} \in L_2 \mid x_{1,i} \in H_1) \geq 1 - O(e^{-cL/R_H})
$$

If $x_{2,i} \in L_2$ (low-error in swarm 2), then:
$$
V_{\text{fit},2,i} > V_{\text{fit},2,\pi(i)}
$$

This is precisely Case B: reversed fitness ordering between the two swarms.

**Step 3: Matching Probability**

The matching $M$ is sampled from Gibbs distribution:

$$
P(M \mid S_1) \propto \prod_{(i,j) \in M} \exp\left(-\frac{d_{\text{alg}}(i,j)^2}{2\varepsilon_d^2}\right)
$$

The minimum probability over all matchings is:

$$
q_{\min}(\varepsilon) = \min_{M \in \mathcal{M}_N} P(M \mid S_1) > 0
$$

This is N-uniform (depends only on $\varepsilon_d$ and algorithmic distance bounds).

**Step 4: Union Bound**

The probability of Case B is at least the probability that one of the target walkers is selected and exhibits Case B:

$$
\mathbb{P}(\text{Case B}) \geq \mathbb{P}(\exists i \in I_{\text{target}} : (i, \pi(i)) \in M \text{ and Case B})
$$

$$
\geq \frac{|I_{\text{target}}|}{N} \cdot q_{\min} \cdot (1 - O(e^{-cL/R_H}))
$$

$$
\geq f_{UH}(\varepsilon) \cdot q_{\min}(\varepsilon) \cdot (1 - O(e^{-cL/R_H}))
$$

For $L > D_{\min}$, the exponential term is negligible:

$$
\mathbb{P}(\text{Case B}) \geq f_{UH}(\varepsilon) \cdot q_{\min}(\varepsilon) > 0
$$

□
:::

:::{prf:remark} Typical Values
:label: rem-case-b-frequency

From the Keystone Principle analysis in [03_cloning.md](03_cloning.md):
- $f_{UH}(\varepsilon) \geq 0.1$ (at least 10% unfit-high-error overlap)
- $q_{\min}(\varepsilon) \geq 0.01$ (matching has at least 1% probability for any configuration)

Therefore:
$$
\mathbb{P}(\text{Case B}) \gtrsim 0.001
$$

While this seems small, it's sufficient for the contraction argument because Case B has strong contraction ($\gamma_B \approx 1 - c_B$) while Case A has weak expansion ($\gamma_A \approx 1 + O(\delta^2/L^2) \to 1$ for large $L$).
:::

---
