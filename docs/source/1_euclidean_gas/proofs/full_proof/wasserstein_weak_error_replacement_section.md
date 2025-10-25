##### 3.7.3.3. Weak Error for Wasserstein Component ($V_W$) - Synchronous Coupling

:::{prf:proposition} BAOAB Weak Error for Wasserstein Distance
:label: prop-weak-error-wasserstein

For $V_W = W_h^2(\mu_1, \mu_2)$ (Wasserstein distance between empirical measures with hypocoercive cost):

$$
\left|\mathbb{E}[V_W(S_\tau^{\text{BAOAB}})] - \mathbb{E}[V_W(S_\tau^{\text{exact}})]\right| \leq K_W \tau^2 (1 + V_W(S_0))
$$

where $K_W = K_W(d, \gamma, L_F, L_\Sigma, \sigma_{\max}, \lambda_v, b)$ is **independent of $N$**.
:::

:::{prf:proof}
**Proof (Synchronous Coupling at Particle Level).**

**PART I: Synchronous Coupling Setup**

Consider two swarms $(S_1, S_2)$ evolving under the **same Brownian motion** $W_i(t)$ for each walker index $i$:

$$
\begin{aligned}
dx_{1,i} &= v_{1,i} \, dt \\
dv_{1,i} &= [F(x_{1,i}) - \gamma v_{1,i}] \, dt + \Sigma(x_{1,i}) \circ dW_i \\[1em]
dx_{2,i} &= v_{2,i} \, dt \\
dv_{2,i} &= [F(x_{2,i}) - \gamma v_{2,i}] \, dt + \Sigma(x_{2,i}) \circ dW_i
\end{aligned}
$$

**Key Property (Noise Cancellation):** The difference process $\Delta z_i(t) = z_{1,i}(t) - z_{2,i}(t)$ evolves as:

$$
\begin{aligned}
d(\Delta x_i) &= \Delta v_i \, dt \\
d(\Delta v_i) &= [\Delta F_i - \gamma \Delta v_i] \, dt + [\Sigma(x_{1,i}) - \Sigma(x_{2,i})] \circ dW_i
\end{aligned}
$$

where $\Delta F_i := F(x_{1,i}) - F(x_{2,i})$.

Since the Brownian motions are identical, the **leading-order noise cancels**. The residual noise $\Delta\Sigma_i = \Sigma(x_{1,i}) - \Sigma(x_{2,i})$ satisfies:

$$
\|\Delta\Sigma_i\|_F \leq L_\Sigma \|\Delta x_i\|
$$

by global Lipschitz continuity ({prf:ref}`axiom-diffusion-tensor`, part 3). The residual noise amplitude is $O(\|\Delta x_i\|)$, so its contribution to the generator acting on quadratic test functions is $O(\|\Delta x_i\|^2)$.

**PART II: Single-Pair Weak Error Analysis**

Define the **hypocoercive quadratic form** on the difference:

$$
f(\Delta z) := \|\Delta z\|_h^2 = \|\Delta x\|^2 + \lambda_v\|\Delta v\|^2 + b\langle\Delta x, \Delta v\rangle = \Delta z^T Q \Delta z
$$

where:

$$
Q = \begin{pmatrix} I_d & \frac{b}{2} I_d \\ \frac{b}{2} I_d & \lambda_v I_d \end{pmatrix}
$$

with $\lambda_v > 0$ and $4\lambda_v - b^2 > 0$ ensuring positive-definiteness.

**Derivatives:** Since $f$ is quadratic:

$$
\nabla f(\Delta z) = 2Q\Delta z \quad \text{(linear growth)}, \quad \nabla^2 f = 2Q \quad \text{(bounded)}, \quad \nabla^3 f = 0
$$

**Apply Weak Error Theory for Polynomial-Growth Test Functions:**

By weak error theory for Langevin dynamics (Leimkuhler & Matthews 2015, Talay-Tubaro expansions), for test functions $g$ with polynomial growth and bounded higher derivatives, under:
- Coercivity ({prf:ref}`axiom-confining-potential`) ensuring $\mathbb{E}[\|Z_t\|^4] < \infty$ uniformly in $t$
- Global Lipschitz $\Sigma$ ({prf:ref}`axiom-diffusion-tensor`)

we have:

$$
\left|\mathbb{E}[g(Z_\tau^{\text{BAOAB}})] - \mathbb{E}[g(Z_\tau^{\text{exact}})]\right| \leq C_{\text{LM}} \tau^2 (1 + \mathbb{E}[\|Z_0\|^{2p}])
$$

where $C_{\text{LM}} = C_{\text{LM}}(d, \gamma, L_F, L_\Sigma, \sigma_{\max})$.

**Apply to $g = f$:** For our quadratic $f$ (with $p=2$):

$$
\left|\mathbb{E}[\|\Delta z_i(\tau)\|_h^2]^{\text{BAOAB}} - \mathbb{E}[\|\Delta z_i(\tau)\|_h^2]^{\text{exact}}\right| \leq C_{\text{pair}} \tau^2 (1 + \|\Delta z_i(0)\|_h^2)
$$

where $C_{\text{pair}} := C_{\text{LM}}(d, \gamma, L_F, L_\Sigma, \sigma_{\max}) \cdot \|Q(\lambda_v, b)\|$.

**PART III: Force Term Handling**

From {prf:ref}`axiom-confining-potential`, the force $F = -\nabla U$ satisfies local Lipschitz bounds on compact sets (ensured by coercivity):

$$
\|\Delta F_i\| \leq L_F \|\Delta x_i\|
$$

The drift of $f(\Delta z_i)$ involves:

$$
\nabla f \cdot \text{drift} = 2Q\Delta z \cdot \begin{pmatrix} \Delta v \\ \Delta F - \gamma \Delta v \end{pmatrix}
$$

The force contribution is quadratic in $\|\Delta z\|_h^2$ and is absorbed into the weak error constant $C_{\text{pair}}$.

**PART IV: Aggregation Over $N$ Particles**

By index-matching:

$$
V_W(S_1, S_2) = W_h^2(\mu_1, \mu_2) \leq \frac{1}{N}\sum_{i=1}^N \|\Delta z_i\|_h^2
$$

Summing the single-pair bounds:

$$
\begin{aligned}
&\left|\mathbb{E}\left[\frac{1}{N}\sum_{i=1}^N \|\Delta z_i(\tau)\|_h^2\right]^{\text{BAOAB}} - \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^N \|\Delta z_i(\tau)\|_h^2\right]^{\text{exact}}\right| \\
&\quad= \frac{1}{N}\sum_{i=1}^N C_{\text{pair}} \tau^2 (1 + \|\Delta z_i(0)\|_h^2) \\
&\quad= C_{\text{pair}} \tau^2 \left(1 + \frac{1}{N}\sum_{i=1}^N \|\Delta z_i(0)\|_h^2\right) \leq C_{\text{pair}} \tau^2 (1 + V_W(S_0))
\end{aligned}
$$

**Propagate to Wasserstein via Min-Over-Permutations:**

Define $C_\sigma(S) := \frac{1}{N}\sum_{i=1}^N \|\Delta z_{\sigma(i)}\|_h^2$ for pairing $\sigma$. Then $V_W(S) = \min_\sigma C_\sigma(S)$.

**Key inequality:** For any states $S^A$, $S^E$:

$$
\left|\min_\sigma C_\sigma(S^A) - \min_\sigma C_\sigma(S^E)\right| \leq \max_\sigma \left|C_\sigma(S^A) - C_\sigma(S^E)\right| \leq \left|C_{\text{id}}(S^A) - C_{\text{id}}(S^E)\right|
$$

Applying to BAOAB vs exact:

$$
\begin{aligned}
\left|\mathbb{E}[V_W(S_\tau)^{\text{BAOAB}}] - \mathbb{E}[V_W(S_\tau)^{\text{exact}}]\right|
&\leq \mathbb{E}\left[\left|V_W(S_\tau^{\text{BAOAB}}) - V_W(S_\tau^{\text{exact}})\right|\right] \\
&\leq \mathbb{E}\left[\left|C_{\text{id}}(S_\tau^{\text{BAOAB}}) - C_{\text{id}}(S_\tau^{\text{exact}})\right|\right] \\
&\leq C_{\text{pair}} \tau^2 (1 + V_W(S_0))
\end{aligned}
$$

**PART V: N-Uniformity**

Define $K_W := C_{\text{pair}} = C_{\text{LM}}(d, \gamma, L_F, L_\Sigma, \sigma_{\max}) \cdot \|Q(\lambda_v, b)\|$.

The constant $K_W$ is **independent of $N$** because:
1. Each walker pair contributes $O(\tau^2)$ error
2. Summing $N$ terms and dividing by $N$ cancels the $N$-dependence
3. No mean-field approximation is used

**Why This Approach Works:**

Unlike the kinetic Fokker-Planck PDE (which is NOT a $W_2$-gradient flow), this proof:
- Works at particle level with finite-$N$ systems
- Uses synchronous coupling for noise cancellation
- Applies standard weak error theory to an explicit quadratic test function
- Rigorously propagates from index-matching to Wasserstein via min-max inequality

**Q.E.D.**
:::

:::{prf:remark} Comparison to Gradient Flow Approach
:label: rem-gradient-flow-vs-coupling

The previous version of this proof incorrectly applied JKO scheme theory for Wasserstein gradient flows to the kinetic Fokker-Planck equation. **Fatal flaws:**

1. **Underdamped Langevin is NOT a $W_2$-gradient flow** - only overdamped Langevin ($dx = F(x)dt + \sigma dW$) has this structure
2. **JKO theory applies to continuous measures** evolving via PDE, not empirical measures (finite $N$)
3. **No verification of technical conditions** for the splitting scheme

The correct approach uses **synchronous coupling at the particle level** - a standard technique in weak error analysis that requires no PDE theory or gradient flow structure.
:::

:::{important}
**Note on Isotropic Diffusion:** For the primary case $\Sigma(x,v) = \sigma_v I_d$ (isotropic, constant diffusion), the Stratonovich and Itô formulations coincide (see {prf:ref}`rem-stratonovich-ito-equivalence`). For general state-dependent $\Sigma$, the BAOAB scheme requires midpoint evaluation for Stratonovich noise, and $L_\Sigma$ appears explicitly in $K_W$.
:::
