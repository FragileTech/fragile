## DEPRECATED: Mixed Fitness Ordering Case Analysis

**⚠️ DEPRECATED - DO NOT USE ⚠️**

**Status:** Incomplete analysis, superseded by complete treatment in 03_wasserstein_contraction_complete.md

**Replacement:** See [03_wasserstein_contraction_complete.md](03_wasserstein_contraction_complete.md) Sections 3-4 for complete Case A and Case B analysis.

**Historical Note:** This was a partial analysis. The complete proof requires the Outlier Alignment Lemma and corrected scaling. Kept for historical reference only.

---

## Mixed Fitness Ordering Case Analysis (INCOMPLETE - ORIGINAL DRAFT)

**This addresses Gemini's CRITICAL Issue #1.**

---

### Setup: Two Cases Based on Fitness Ordering

For a pair $(i, j)$ where $j = \pi(i)$ from matching $M$, there are two possible fitness orderings:

**Case A (Consistent Ordering):** Both swarms have the same lower-fitness walker
- Swarm 1: $V_{\text{fit},1,i} \leq V_{\text{fit},1,j}$
- Swarm 2: $V_{\text{fit},2,i} \leq V_{\text{fit},2,j}$
- **Cloning pattern:** Walker $i$ clones in both swarms (or neither)

**Case B (Mixed Ordering):** Different walkers have lower fitness
- Swarm 1: $V_{\text{fit},1,i} \leq V_{\text{fit},1,j}$
- Swarm 2: $V_{\text{fit},2,i} > V_{\text{fit},2,j}$
- **Cloning pattern:** Walker $i$ clones in swarm 1, walker $j$ clones in swarm 2

---

### Case B: Mixed Fitness Ordering (New Analysis)

**State updates:**

| Swarm | Walker $i$ | Walker $j$ |
|:------|:-----------|:-----------|
| Swarm 1 | $x'_{1,i} = \begin{cases} x_{1,j} + \zeta_i & \text{if } U_i < p_{1,i} \\ x_{1,i} & \text{otherwise} \end{cases}$ | $x'_{1,j} = x_{1,j}$ (persists) |
| Swarm 2 | $x'_{2,i} = x_{2,i}$ (persists) | $x'_{2,j} = \begin{cases} x_{2,i} + \zeta_j & \text{if } U_j < p_{2,j} \\ x_{2,j} & \text{otherwise} \end{cases}$ |

**Key difference from Case A:**
- Different walkers clone in each swarm ($i$ vs $j$)
- Different jitter vectors used ($\zeta_i$ vs $\zeta_j$)
- These jitters are INDEPENDENT (no cancellation)

---

### Subcases for Case B

Define indicator variables:
- $I_{1,i} = \mathbb{1}[U_i < p_{1,i}]$ (walker $i$ clones in swarm 1)
- $I_{2,j} = \mathbb{1}[U_j < p_{2,j}]$ (walker $j$ clones in swarm 2)

**Subcase B1:** Neither clones ($I_{1,i} = 0, I_{2,j} = 0$)
- Probability: $(1 - p_{1,i})(1 - p_{2,j})$
- States: $x'_{1,i} = x_{1,i}, x'_{1,j} = x_{1,j}, x'_{2,i} = x_{2,i}, x'_{2,j} = x_{2,j}$
- Distance:
  $$\|x'_{1,i} - x'_{2,i}\|^2 + \|x'_{1,j} - x'_{2,j}\|^2 = \|x_{1,i} - x_{2,i}\|^2 + \|x_{1,j} - x_{2,j}\|^2$$

**Subcase B2:** Only walker $i$ clones ($I_{1,i} = 1, I_{2,j} = 0$)
- Probability: $p_{1,i}(1 - p_{2,j})$
- States: $x'_{1,i} = x_{1,j} + \zeta_i, x'_{1,j} = x_{1,j}, x'_{2,i} = x_{2,i}, x'_{2,j} = x_{2,j}$
- Distance:
  $$\begin{aligned}
  &\|x'_{1,i} - x'_{2,i}\|^2 + \|x'_{1,j} - x'_{2,j}\|^2 \\
  &= \|x_{1,j} + \zeta_i - x_{2,i}\|^2 + \|x_{1,j} - x_{2,j}\|^2
  \end{aligned}$$

  Taking expectation over $\zeta_i$:
  $$\mathbb{E}_{\zeta_i} = \|x_{1,j} - x_{2,i}\|^2 + d\delta^2 + \|x_{1,j} - x_{2,j}\|^2$$

**Subcase B3:** Only walker $j$ clones ($I_{1,i} = 0, I_{2,j} = 1$)
- Probability: $(1 - p_{1,i})p_{2,j}$
- States: $x'_{1,i} = x_{1,i}, x'_{1,j} = x_{1,j}, x'_{2,i} = x_{2,i}, x'_{2,j} = x_{2,i} + \zeta_j$
- Distance:
  $$\begin{aligned}
  &\|x'_{1,i} - x'_{2,i}\|^2 + \|x'_{1,j} - x'_{2,j}\|^2 \\
  &= \|x_{1,i} - x_{2,i}\|^2 + \|x_{1,j} - (x_{2,i} + \zeta_j)\|^2
  \end{aligned}$$

  Taking expectation over $\zeta_j$:
  $$\mathbb{E}_{\zeta_j} = \|x_{1,i} - x_{2,i}\|^2 + \|x_{1,j} - x_{2,i}\|^2 + d\delta^2$$

**Subcase B4:** Both clone ($I_{1,i} = 1, I_{2,j} = 1$)
- Probability: $p_{1,i} \cdot p_{2,j}$
- States: $x'_{1,i} = x_{1,j} + \zeta_i, x'_{1,j} = x_{1,j}, x'_{2,i} = x_{2,i}, x'_{2,j} = x_{2,i} + \zeta_j$
- Distance:
  $$\begin{aligned}
  &\|x'_{1,i} - x'_{2,i}\|^2 + \|x'_{1,j} - x'_{2,j}\|^2 \\
  &= \|x_{1,j} + \zeta_i - x_{2,i}\|^2 + \|x_{1,j} - (x_{2,i} + \zeta_j)\|^2
  \end{aligned}$$

  Since $\zeta_i$ and $\zeta_j$ are independent:
  $$\mathbb{E}_{\zeta_i, \zeta_j} = \|x_{1,j} - x_{2,i}\|^2 + d\delta^2 + \|x_{1,j} - x_{2,i}\|^2 + d\delta^2 = 2\|x_{1,j} - x_{2,i}\|^2 + 2d\delta^2$$

---

### Combined Expectation for Case B

$$\begin{aligned}
&\mathbb{E}[\|x'_{1,i} - x'_{2,i}\|^2 + \|x'_{1,j} - x'_{2,j}\|^2 \mid \text{Case B}] \\
&= (1 - p_{1,i})(1 - p_{2,j}) [\|x_{1,i} - x_{2,i}\|^2 + \|x_{1,j} - x_{2,j}\|^2] \\
&\quad + p_{1,i}(1 - p_{2,j}) [\|x_{1,j} - x_{2,i}\|^2 + \|x_{1,j} - x_{2,j}\|^2 + d\delta^2] \\
&\quad + (1 - p_{1,i})p_{2,j} [\|x_{1,i} - x_{2,i}\|^2 + \|x_{1,j} - x_{2,i}\|^2 + d\delta^2] \\
&\quad + p_{1,i} p_{2,j} [2\|x_{1,j} - x_{2,i}\|^2 + 2d\delta^2]
\end{aligned}$$

---

### Key Observation: Cross-Term Structure

Define:
- $D_{ii} := \|x_{1,i} - x_{2,i}\|^2$ (same walker, different swarms)
- $D_{jj} := \|x_{1,j} - x_{2,j}\|^2$ (same walker, different swarms)
- $D_{ji} := \|x_{1,j} - x_{2,i}\|^2$ (cross-term)

**Initial distance:** $D_0^2 = D_{ii} + D_{jj}$

**Case B involves cross-term $D_{ji}$!**

By triangle inequality:
$$D_{ji} = \|x_{1,j} - x_{2,i}\|^2 = \|(x_{1,j} - x_{1,i}) + (x_{1,i} - x_{2,i}) + (x_{2,i} - x_{2,j}) + (x_{2,j} - x_{2,i})\|^2$$

Hmm, this is getting messy. Let me use a cleaner decomposition.

**Decompose using barycenters:**
$$x_{k,i} - x_{k,j} = (x_{k,i} - \bar{x}_k) - (x_{k,j} - \bar{x}_k) = \delta_{k,i} - \delta_{k,j}$$

Then:
$$\begin{aligned}
D_{ji} &= \|(x_{1,j} - \bar{x}_1) - (x_{2,i} - \bar{x}_2) + (\bar{x}_1 - \bar{x}_2)\|^2 \\
&= \|\delta_{1,j} - \delta_{2,i}\|^2 + 2\langle \delta_{1,j} - \delta_{2,i}, \bar{x}_1 - \bar{x}_2 \rangle + \|\bar{x}_1 - \bar{x}_2\|^2
\end{aligned}$$

**Key insight from Keystone Principle:**
- If $p_{1,i}$ is large, then walker $i$ in swarm 1 is an outlier: $\|\delta_{1,i}\|$ large
- Companion $j$ is near barycenter: $\|\delta_{1,j}\|$ small
- Similarly for swarm 2

**This suggests:** When cloning probabilities are high, the cross-term $D_{ji}$ may be SMALLER than the direct term $D_{ii}$ because companions are closer to barycenters.

---

### Contraction Analysis (To Be Completed)

The full contraction bound for Case B requires:
1. Bounding $D_{ji}$ using Keystone properties (companion concentration)
2. Showing that the weighted combination contracts $D_0^2 = D_{ii} + D_{jj}$
3. Handling the barycenter separation term $\|\bar{x}_1 - \bar{x}_2\|^2$

**Status:** Framework established, rigorous bounds need Keystone lemmas.
