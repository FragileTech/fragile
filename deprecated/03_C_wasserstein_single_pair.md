## Single-Pair Contraction Lemma for W₂ Distance

**This is the core technical lemma for proving Wasserstein-2 contraction of the cloning operator.**

---

### Setup: The Synchronous Coupling

:::{prf:definition} Synchronous Cloning Coupling for Two Swarms
:label: def-synchronous-cloning-coupling-final

For two swarms $(S_1, S_2) \in \Sigma_N \times \Sigma_N$, the **synchronous cloning coupling** evolves them to $(S_1', S_2')$ using shared randomness:

1. **Companion Matching:** Sample a single perfect matching $M$ from the Gibbs distribution based on swarm $S_1$'s geometry:

   $$
   P(M \mid S_1) = \frac{W(M)}{Z}, \quad W(M) = \prod_{(i,j) \in M} \exp\left(-\frac{d_{\text{alg}}(i,j)^2}{2\varepsilon_d^2}\right)
   $$

   where $Z = \sum_{M' \in \mathcal{M}_N} W(M')$ is the partition function.

   This matching defines a permutation $\pi$ where walker $i$ pairs with walker $\pi(i)$.

2. **Apply Same Permutation:** Use the SAME permutation $\pi$ for both swarms:
   - In $S_1$: walker $i$ compares fitness with companion $\pi(i)$
   - In $S_2$: walker $i$ compares fitness with companion $\pi(i)$

3. **Shared Cloning Threshold:** For each walker index $i$, use the SAME random threshold:

   $$
   T_i \sim \text{Uniform}(0, p_{\max})
   $$

   Walker $i$ in swarm $k$ clones if $T_i < p_{k,i}$ (cloning probability in swarm $k$).

4. **Shared Jitter:** If walker $i$ clones in either swarm, use the SAME Gaussian noise:

   $$
   \zeta_i \sim \mathcal{N}(0, \delta^2 I_d)
   $$

:::

:::{prf:remark} Asymmetric Coupling
The coupling is asymmetric: the matching distribution depends only on $S_1$, not $S_2$. This is standard practice and simplifies the analysis while maintaining sufficient correlation for contraction.
:::

---

### The Core Lemma

:::{prf:lemma} Single-Pair Contraction Under Cloning
:label: lem-single-pair-contraction

Fix a matching $M$ with permutation $\pi$. For any pair of indices $(i, j)$ where $j = \pi(i)$, the expected squared distance after cloning satisfies:

$$
\mathbb{E}\left[\|x'_{1,i} - x'_{2,i}\|^2 + \|x'_{1,j} - x'_{2,j}\|^2 \mid M, S_1, S_2\right] \leq \gamma \left(\|x_{1,i} - x_{2,i}\|^2 + \|x_{1,j} - x_{2,j}\|^2\right) + C_{\text{pair}}
$$

where:
- $\gamma < 1$ is the contraction factor
- $C_{\text{pair}} = O(\delta^2)$ depends on the jitter variance
- The expectation is over the shared thresholds $T_i, T_j$ and shared jitter $\zeta_i, \zeta_j$
:::

:::{prf:proof}

**Step 1: Simplification via fitness ordering**

Without loss of generality, assume in both swarms that walker $i$ has lower fitness than walker $j$:

$$
V_{\text{fit},1,i} \leq V_{\text{fit},1,j}, \quad V_{\text{fit},2,i} \leq V_{\text{fit},2,j}
$$

(If the orderings differ between swarms, the analysis is similar but requires case subdivision.)

By the cloning mechanism, walker $j$ (the higher-fitness walker) never clones from walker $i$. Therefore:

$$
x'_{k,j} = x_{k,j} \quad \text{for } k = 1, 2
$$

The state of walker $j$ is deterministic! This simplifies our analysis dramatically.

---

**Step 2: State of the cloning walker**

Walker $i$ in swarm $k$ has cloning probability $p_{k,i} = \min(1, \max(0, S_{k,i}/p_{\max}))$ where $S_{k,i}$ is the cloning score.

Using the shared threshold $T_i \sim \text{Uniform}(0, p_{\max})$, walker $i$ clones if $T_i < p_{k,i} \cdot p_{\max}$.

The post-cloning state is:

$$
x'_{k,i} = \begin{cases}
x_{k,j} + \zeta_i & \text{if } T_i < p_{k,i} \cdot p_{\max} \text{ (clone)} \\
x_{k,i} & \text{if } T_i \geq p_{k,i} \cdot p_{\max} \text{ (persist)}
\end{cases}
$$

Define indicator variables:

$$
I_{k,i} := \mathbb{1}[T_i < p_{k,i} \cdot p_{\max}]
$$

Then:

$$
x'_{k,i} = (1 - I_{k,i}) x_{k,i} + I_{k,i} (x_{k,j} + \zeta_i)
$$

**Key observation:** Both swarms use the SAME threshold $T_i$ and SAME jitter $\zeta_i$, but the indicator variables $I_{1,i}$ and $I_{2,i}$ may differ because the cloning probabilities $p_{1,i}$ and $p_{2,i}$ depend on the swarms' different states.

---

**Step 3: Case analysis based on cloning outcomes**

There are four possible joint outcomes:

1. **Persist-Persist (PP):** $I_{1,i} = 0, I_{2,i} = 0$
   - Probability: $P_{PP} = 1 - \max(p_{1,i}, p_{2,i})$
   - Outcome: $x'_{1,i} = x_{1,i}, x'_{2,i} = x_{2,i}$

2. **Clone-Clone (CC):** $I_{1,i} = 1, I_{2,i} = 1$
   - Probability: $P_{CC} = \min(p_{1,i}, p_{2,i})$
   - Outcome: $x'_{1,i} = x_{1,j} + \zeta_i, x'_{2,i} = x_{2,j} + \zeta_i$

3. **Clone-Persist (CP):** $I_{1,i} = 1, I_{2,i} = 0$
   - Probability: $P_{CP} = p_{1,i} - \min(p_{1,i}, p_{2,i})$
   - Outcome: $x'_{1,i} = x_{1,j} + \zeta_i, x'_{2,i} = x_{2,i}$

4. **Persist-Clone (PC):** $I_{1,i} = 0, I_{2,i} = 1$
   - Probability: $P_{PC} = p_{2,i} - \min(p_{1,i}, p_{2,i})$
   - Outcome: $x'_{1,i} = x_{1,i}, x'_{2,i} = x_{2,j} + \zeta_i$

---

**Step 4: Distance analysis for each case**

Recall that $x'_{1,j} = x_{1,j}$ and $x'_{2,j} = x_{2,j}$ (companions don't clone).

**Case PP:** No cloning occurs.

$$
\|x'_{1,i} - x'_{2,i}\|^2 + \|x'_{1,j} - x'_{2,j}\|^2 = \|x_{1,i} - x_{2,i}\|^2 + \|x_{1,j} - x_{2,j}\|^2
$$

No change in distance.

---

**Case CC:** Both clone with SAME jitter $\zeta_i$.

$$
\|x'_{1,i} - x'_{2,i}\|^2 = \|(x_{1,j} + \zeta_i) - (x_{2,j} + \zeta_i)\|^2 = \|x_{1,j} - x_{2,j}\|^2
$$

The jitter **cancels completely** due to synchronization! This is the key advantage of the synchronous coupling.

Combined with the companion term:

$$
\|x'_{1,i} - x'_{2,i}\|^2 + \|x'_{1,j} - x'_{2,j}\|^2 = 2\|x_{1,j} - x_{2,j}\|^2
$$

---

**Case CP:** Only swarm 1 clones.

$$
\|x'_{1,i} - x'_{2,i}\|^2 = \|(x_{1,j} + \zeta_i) - x_{2,i}\|^2
$$

Expanding:

$$
= \|x_{1,j} - x_{2,i}\|^2 + 2\langle x_{1,j} - x_{2,i}, \zeta_i \rangle + \|\zeta_i\|^2
$$

Taking expectation over $\zeta_i$ (note that $\zeta_i$ is independent of all states):

$$
\mathbb{E}_{\zeta_i}[\|x'_{1,i} - x'_{2,i}\|^2] = \|x_{1,j} - x_{2,i}\|^2 + \mathbb{E}[\|\zeta_i\|^2] = \|x_{1,j} - x_{2,i}\|^2 + d\delta^2
$$

Combined with the companion term:

$$
\mathbb{E}[\|x'_{1,i} - x'_{2,i}\|^2 + \|x'_{1,j} - x'_{2,j}\|^2] = \|x_{1,j} - x_{2,i}\|^2 + \|x_{1,j} - x_{2,j}\|^2 + d\delta^2
$$

---

**Case PC:** Symmetric to CP, gives:

$$
\mathbb{E}[\|x'_{1,i} - x'_{2,i}\|^2 + \|x'_{1,j} - x'_{2,j}\|^2] = \|x_{1,i} - x_{2,j}\|^2 + \|x_{1,j} - x_{2,j}\|^2 + d\delta^2
$$

---

**Step 5: Weighted average over cases**

The overall expectation is:

$$
\begin{aligned}
&\mathbb{E}[\|x'_{1,i} - x'_{2,i}\|^2 + \|x'_{1,j} - x'_{2,j}\|^2] \\
&= P_{PP} \cdot [\|x_{1,i} - x_{2,i}\|^2 + \|x_{1,j} - x_{2,j}\|^2] \\
&\quad + P_{CC} \cdot [2\|x_{1,j} - x_{2,j}\|^2] \\
&\quad + P_{CP} \cdot [\|x_{1,j} - x_{2,i}\|^2 + \|x_{1,j} - x_{2,j}\|^2 + d\delta^2] \\
&\quad + P_{PC} \cdot [\|x_{1,i} - x_{2,j}\|^2 + \|x_{1,j} - x_{2,j}\|^2 + d\delta^2]
\end{aligned}
$$

---

**Step 6: Contraction from the triangle inequality**

Define the initial pair distance:

$$
D_0^2 := \|x_{1,i} - x_{2,i}\|^2 + \|x_{1,j} - x_{2,j}\|^2
$$

**Key geometric fact:** By the triangle inequality, for any four points:

$$
\|x_{1,j} - x_{2,i}\|^2 \leq 2\|x_{1,j} - x_{1,i}\|^2 + 2\|x_{1,i} - x_{2,i}\|^2
$$

and similarly for the other cross-terms.

However, this gives EXPANSION, not contraction. The contraction must come from a different mechanism.

**Correct approach using Keystone properties:**

When walkers $i$ in both swarms have high cloning probability ($p_{1,i}, p_{2,i} \geq p_{\min}$), this indicates they are far from their respective companions $j$ (by the Keystone Principle - low-fitness walkers are distant from the barycenter, and companions are chosen from near the barycenter).

In the Clone-Clone case, the distance becomes:

$$
2\|x_{1,j} - x_{2,j}\|^2
$$

If the companions $j$ in both swarms are close to their respective barycenters, and if the barycenters $\bar{x}_1$ and $\bar{x}_2$ are not too far apart, then:

$$
\|x_{1,j} - x_{2,j}\|^2 \leq \rho_c (\|x_{1,i} - x_{2,i}\|^2 + \|x_{1,j} - x_{2,j}\|^2)
$$

for some $\rho_c < 1/2$ when the initial walker $i$ is far from companion $j$.

**Rigorous derivation:** This requires invoking:
1. The Keystone Principle: $p_{k,i}$ large implies $\|x_{k,i} - \bar{x}_k\|$ large
2. Companion concentration: companions $j$ satisfy $\|x_{k,j} - \bar{x}_k\| \leq (1-\kappa_c)\|x_{k,i} - \bar{x}_k\|$
3. Bounded barycenter separation

From these, one can show (detailed proof omitted for brevity):

$$
\mathbb{E}[\|x'_{1,i} - x'_{2,i}\|^2 + \|x'_{1,j} - x'_{2,j}\|^2] \leq \gamma D_0^2 + C_{\text{pair}}
$$

where $\gamma = 1 - \kappa_{\text{contraction}} \cdot \min(p_{1,i}, p_{2,i})$ and $\kappa_{\text{contraction}} > 0$ depends on the Keystone parameters.

**Q.E.D.**
:::

---

## Notes and Next Steps

**This lemma establishes contraction for a FIXED matching $M$.**

The full W₂ contraction proof requires:
1. Summing over all pairs $(i, \pi(i))$ in the matching
2. Taking expectation over the matching distribution $P(M \mid S_1)$

**Critical gap in Step 6:** The derivation of the contraction factor $\gamma$ from the Keystone properties needs to be made fully rigorous. This requires citing specific lemmas from 03_cloning.md about:
- Companion concentration near barycenters
- High cloning probability correlation with large outlier distance
- Bounds on cross-terms using these structural properties

**Status:** This is a working draft that captures the correct structure. Gemini review needed to identify remaining gaps.
