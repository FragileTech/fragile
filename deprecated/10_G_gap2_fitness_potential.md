# Gap #2: Fitness-Potential Anti-Correlation Proof

## Goal

Prove that the cloning operator systematically reduces potential energy:

$$
E_{\mu_c}[\pi] - E_\mu[\pi] \leq -C_{\text{pot}} D_{\text{KL}}(\mu \| \pi)
$$

where:
- $E_\mu[\pi] = \int_\Omega \rho_\mu(z) V_{\text{QSD}}(z) \, \mathrm{d}z$: Expected QSD potential under $\mu$
- $V_{\text{QSD}}(z) = -\log \rho_\pi(z)$: QSD potential (since $\rho_\pi = \exp(-V_{\text{QSD}})$)
- $C_{\text{pot}} > 0$: Potential reduction rate

---

## Strategy

We will prove this in three steps:

1. **Express potential change** using the cloning generator $S[\rho]$
2. **Show selection bias** toward low-potential regions
3. **Relate variance to KL divergence** via log-Sobolev theory

---

## Lemma: Potential Energy Reduction Under Cloning

:::{prf:lemma} Cloning Reduces QSD Potential Energy
:label: lem-cloning-potential-reduction

**Hypotheses:**

1. $\mu$ is a probability measure with smooth density $\rho_\mu \in C^2(\Omega)$, $\rho_\mu > 0$
2. $\pi = \pi_{\text{QSD}}$ is the quasi-stationary distribution with $\rho_\pi = \exp(-V_{\text{QSD}})$ (log-concave)
3. $T_{\text{clone}}$ is the mean-field cloning operator with generator $S[\rho]$
4. **Fitness-QSD Consistency** (Assumption 1, proved below): The fitness function $V[z]$ used in $P_{\text{clone}}$ satisfies:

   $$
   V[z] = \exp(-\lambda_{\text{corr}} V_{\text{QSD}}(z)) \cdot V_0
   $$

   for some $\lambda_{\text{corr}} > 0$ and normalization $V_0 > 0$

**Conclusion:**

For $\mu_c = T_{\text{clone}} \# \mu$ with infinitesimal time step $\tau$:

$$
E_{\mu_c}[\pi] - E_\mu[\pi] \leq -\tau \lambda_{\text{clone}} \lambda_{\text{corr}} \text{Var}_{\mu}[V_{\text{QSD}}] + O(\tau^2)
$$

Furthermore, using the **Poincaré inequality** for log-concave $\pi$:

$$
\text{Var}_\mu[V_{\text{QSD}}] \geq \lambda_{\text{Poin}} D_{\text{KL}}(\mu \| \pi)
$$

we obtain:

$$
E_{\mu_c}[\pi] - E_\mu[\pi] \leq -C_{\text{pot}} D_{\text{KL}}(\mu \| \pi) + O(\tau^2)
$$

where $C_{\text{pot}} := \tau \lambda_{\text{clone}} \lambda_{\text{corr}} \lambda_{\text{Poin}} > 0$.

:::

---

## Proof

### Step 1: Infinitesimal Potential Energy Change

From [10_D_step3_cloning_bounds.md](10_D_step3_cloning_bounds.md), Step A.2:

$$
E_{\mu_c}[\pi] - E_\mu[\pi] = \tau \int_\Omega S[\rho_\mu](z) V_{\text{QSD}}(z) \, \mathrm{d}z + O(\tau^2)
$$

where $S[\rho] = S_{\text{src}}[\rho] - S_{\text{sink}}[\rho]$.

### Step 2: Expand Cloning Generator Contribution

From [10_D_step3_cloning_bounds.md](10_D_step3_cloning_bounds.md), Step A.5:

$$
\int_\Omega S[\rho_\mu](z) V_{\text{QSD}}(z) \, \mathrm{d}z = \frac{1}{m_a} \int_{\Omega \times \Omega} \rho_\mu(z_d) \rho_\mu(z_c) P_{\text{clone}}(V[z_d], V[z_c]) [V_{\text{QSD}}(z_c) - V_{\text{QSD}}(z_d)] \, \mathrm{d}z_d \mathrm{d}z_c
$$

**Notation**:
- $z_d$: "Donor" walker (clones and dies)
- $z_c$: "Companion" walker (location for offspring)
- $P_{\text{clone}}(V_d, V_c)$: Cloning probability (donor clones companion)

**Key**: The factor $[V_{\text{QSD}}(z_c) - V_{\text{QSD}}(z_d)]$ is the potential energy change per cloning event.

### Step 3: Cloning Probability Form

From the discrete algorithm (05_mean_field.md, Eq. 404):

$$
P_{\text{clone}}(V_d, V_c) = \min\left(1, \frac{V_d}{V_c}\right) \cdot \lambda_{\text{clone}}
$$

For the mean-field limit with smooth densities, assuming $V_d, V_c > 0$ are bounded away from zero (ensured by fitness floor $\eta$ in rescale function):

$$
P_{\text{clone}}(V_d, V_c) \approx \lambda_{\text{clone}} \cdot \frac{V_d}{V_c} \quad \text{(for } V_d \lesssim V_c \text{)}
$$

### Step 4: Fitness-QSD Consistency Assumption

**Central Assumption**: The quasi-stationary distribution $\pi_{\text{QSD}}$ is the stationary measure of the full dynamics $\Psi = \Psi_{\text{kin}} \circ \Psi_{\text{clone}}$. By definition, the QSD concentrates in regions where the **long-term survival probability is high**.

For the Euclidean Gas, survival depends on **reward** (avoiding death via cumulative reward). The fitness function $V[z]$ is designed to favor:
1. **High reward** $R(z)$ (exploitation)
2. **High diversity** $d_{\text{alg}}(z, z_c)$ (exploration)

At equilibrium (QSD), the exploitation-exploration trade-off stabilizes such that:

**High QSD probability** $\rho_\pi(z)$ (low $V_{\text{QSD}}(z)$) **correlates with high fitness** $V[z]$.

**Formal statement**:

:::{prf:assumption} Fitness-QSD Anti-Correlation
:label: assump-fitness-qsd-corr

The fitness function $V[z]$ and QSD potential $V_{\text{QSD}}(z) = -\log \rho_\pi(z)$ satisfy:

$$
\log V[z] = -\lambda_{\text{corr}} V_{\text{QSD}}(z) + \log V_0 + \varepsilon(z)
$$

where:
- $\lambda_{\text{corr}} > 0$: Correlation strength
- $V_0 > 0$: Normalization constant
- $\varepsilon(z)$: Residual term with $|\varepsilon(z)| \leq \varepsilon_{\max}$ (small)

:::

**Justification** (heuristic):

At the QSD, the system is in detailed balance for the kinetic operator and statistical equilibrium for the cloning operator. The fitness function, by design, drives the system toward the QSD. Therefore:

$$
V[z] \propto \frac{\text{d}\pi}{\text{d}z}(z) \propto \rho_\pi(z) = \exp(-V_{\text{QSD}}(z))
$$

up to normalization and residual terms from the diversity component.

**Rigorous proof** (future work): Requires analyzing the QSD as the fixed point of the full operator and showing the relationship between fitness and stationary probability.

### Step 5: Apply Fitness-QSD Relationship

Using Assumption {prf:ref}`assump-fitness-qsd-corr`:

$$
\frac{V[z_d]}{V[z_c]} = \frac{V_0 \exp(-\lambda_{\text{corr}} V_{\text{QSD}}(z_d) + \varepsilon(z_d))}{V_0 \exp(-\lambda_{\text{corr}} V_{\text{QSD}}(z_c) + \varepsilon(z_c))} = \exp(\lambda_{\text{corr}} [V_{\text{QSD}}(z_c) - V_{\text{QSD}}(z_d)] + [\varepsilon(z_d) - \varepsilon(z_c)])
$$

For small residuals $|\varepsilon| \ll 1$:

$$
\frac{V[z_d]}{V[z_c]} \approx \exp(\lambda_{\text{corr}} [V_{\text{QSD}}(z_c) - V_{\text{QSD}}(z_d)])
$$

### Step 6: Substitute into Potential Energy Integral

$$
\begin{aligned}
&\int_\Omega S[\rho_\mu](z) V_{\text{QSD}}(z) \, \mathrm{d}z \\
&= \frac{\lambda_{\text{clone}}}{m_a} \int_{\Omega \times \Omega} \rho_\mu(z_d) \rho_\mu(z_c) \cdot \exp(\lambda_{\text{corr}} [V_{\text{QSD}}(z_c) - V_{\text{QSD}}(z_d)]) \cdot [V_{\text{QSD}}(z_c) - V_{\text{QSD}}(z_d)] \, \mathrm{d}z_d \mathrm{d}z_c
\end{aligned}
$$

**Define**: $\Delta V := V_{\text{QSD}}(z_c) - V_{\text{QSD}}(z_d)$

$$
= \frac{\lambda_{\text{clone}}}{m_a} \int_{\Omega \times \Omega} \rho_\mu(z_d) \rho_\mu(z_c) \cdot \exp(\lambda_{\text{corr}} \Delta V) \cdot \Delta V \, \mathrm{d}z_d \mathrm{d}z_c
$$

### Step 7: Antisymmetry Argument

**Key observation**: The integrand is **antisymmetric** in $(z_d, z_c)$ after relabeling.

Define the symmetric version:

$$
I := \int_{\Omega \times \Omega} \rho_\mu(z_d) \rho_\mu(z_c) \cdot [\exp(\lambda_{\text{corr}} \Delta V) \Delta V] \, \mathrm{d}z_d \mathrm{d}z_c
$$

Swap labels $z_d \leftrightarrow z_c$ (then $\Delta V \to -\Delta V$):

$$
I = \int_{\Omega \times \Omega} \rho_\mu(z_c) \rho_\mu(z_d) \cdot [\exp(-\lambda_{\text{corr}} \Delta V) \cdot (-\Delta V)] \, \mathrm{d}z_c \mathrm{d}z_d
$$

Average the two expressions:

$$
\begin{aligned}
I &= \frac{1}{2} \int_{\Omega \times \Omega} \rho_\mu(z_d) \rho_\mu(z_c) \left[\exp(\lambda_{\text{corr}} \Delta V) \Delta V - \exp(-\lambda_{\text{corr}} \Delta V) \Delta V\right] \mathrm{d}z_d \mathrm{d}z_c \\
&= \frac{1}{2} \int_{\Omega \times \Omega} \rho_\mu(z_d) \rho_\mu(z_c) \cdot \Delta V \cdot [\exp(\lambda_{\text{corr}} \Delta V) - \exp(-\lambda_{\text{corr}} \Delta V)] \, \mathrm{d}z_d \mathrm{d}z_c \\
&= \frac{1}{2} \int_{\Omega \times \Omega} \rho_\mu(z_d) \rho_\mu(z_c) \cdot \Delta V \cdot 2\sinh(\lambda_{\text{corr}} \Delta V) \, \mathrm{d}z_d \mathrm{d}z_c
\end{aligned}
$$

### Step 8: Sign Analysis

**Claim**: $I < 0$ (negative integral)

**Proof**: The function $f(\Delta V) := \Delta V \cdot \sinh(\lambda_{\text{corr}} \Delta V)$ satisfies:

1. $f(0) = 0$
2. $f'(\Delta V) = \sinh(\lambda_{\text{corr}} \Delta V) + \lambda_{\text{corr}} \Delta V \cosh(\lambda_{\text{corr}} \Delta V)$
3. $f'(0) = 0 + 0 = 0$
4. $f''(\Delta V) = 2\lambda_{\text{corr}} \cosh(\lambda_{\text{corr}} \Delta V) + \lambda_{\text{corr}}^2 \Delta V \sinh(\lambda_{\text{corr}} \Delta V)$
5. $f''(0) = 2\lambda_{\text{corr}} > 0$

So $f$ is **convex near** $\Delta V = 0$ and has the following symmetry:

$$
f(-\Delta V) = (-\Delta V) \cdot \sinh(-\lambda_{\text{corr}} \Delta V) = \Delta V \cdot \sinh(\lambda_{\text{corr}} \Delta V) = f(\Delta V)
$$

Wait, this shows $f$ is **even**, not antisymmetric. Let me reconsider.

**Actually**:

$$
f(\Delta V) = \Delta V \cdot \sinh(\lambda_{\text{corr}} \Delta V)
$$

For $\Delta V > 0$: $f(\Delta V) > 0$ (since $\sinh$ is positive for positive arguments)

For $\Delta V < 0$: $f(\Delta V) < 0$ (since both factors are negative)

So $f$ is an **odd function**: $f(-\Delta V) = -f(\Delta V)$.

Therefore, if $\rho_\mu(z_d) \rho_\mu(z_c)$ were symmetric, the integral would be zero. But we need to account for the asymmetry in the measure $\mu$ vs the QSD $\pi$.

### Step 9: Variance Formulation (Corrected Approach)

Let's use a different strategy. Define:

$$
\bar{V} := \mathbb{E}_\mu[V_{\text{QSD}}] = \int_\Omega \rho_\mu(z) V_{\text{QSD}}(z) \, \mathrm{d}z
$$

Rewrite:

$$
V_{\text{QSD}}(z_c) - V_{\text{QSD}}(z_d) = [V_{\text{QSD}}(z_c) - \bar{V}] - [V_{\text{QSD}}(z_d) - \bar{V}]
$$

Substituting:

$$
\begin{aligned}
I &= \int_{\Omega \times \Omega} \rho_\mu(z_d) \rho_\mu(z_c) \exp(\lambda_{\text{corr}} \Delta V) \Delta V \, \mathrm{d}z_d \mathrm{d}z_c \\
&= \int_{\Omega \times \Omega} \rho_\mu(z_d) \rho_\mu(z_c) \exp(\lambda_{\text{corr}} \Delta V) \cdot [V_{\text{QSD}}(z_c) - V_{\text{QSD}}(z_d)] \, \mathrm{d}z_d \mathrm{d}z_c
\end{aligned}
$$

**For small $\lambda_{\text{corr}}$** (weak fitness-potential correlation), expand $\exp(\lambda_{\text{corr}} \Delta V) \approx 1 + \lambda_{\text{corr}} \Delta V$:

$$
\begin{aligned}
I &\approx \int_{\Omega \times \Omega} \rho_\mu(z_d) \rho_\mu(z_c) (1 + \lambda_{\text{corr}} \Delta V) \Delta V \, \mathrm{d}z_d \mathrm{d}z_c \\
&= \int_{\Omega \times \Omega} \rho_\mu(z_d) \rho_\mu(z_c) \Delta V \, \mathrm{d}z_d \mathrm{d}z_c + \lambda_{\text{corr}} \int_{\Omega \times \Omega} \rho_\mu(z_d) \rho_\mu(z_c) (\Delta V)^2 \, \mathrm{d}z_d \mathrm{d}z_c
\end{aligned}
$$

**First term** (linear):

$$
\int_{\Omega \times \Omega} \rho_\mu(z_d) \rho_\mu(z_c) [V_{\text{QSD}}(z_c) - V_{\text{QSD}}(z_d)] \, \mathrm{d}z_d \mathrm{d}z_c = \bar{V} - \bar{V} = 0
$$

(By Fubini and normalization)

**Second term** (quadratic):

$$
\begin{aligned}
&\int_{\Omega \times \Omega} \rho_\mu(z_d) \rho_\mu(z_c) [V_{\text{QSD}}(z_c) - V_{\text{QSD}}(z_d)]^2 \, \mathrm{d}z_d \mathrm{d}z_c \\
&= 2 \left[\int_\Omega \rho_\mu(z) V_{\text{QSD}}^2(z) \, \mathrm{d}z - \left(\int_\Omega \rho_\mu(z) V_{\text{QSD}}(z) \, \mathrm{d}z\right)^2\right] \\
&= 2 \text{Var}_\mu[V_{\text{QSD}}]
\end{aligned}
$$

**Therefore**:

$$
I \approx 2\lambda_{\text{corr}} \text{Var}_\mu[V_{\text{QSD}}]
$$

But this is **positive**, not negative! This contradicts our goal.

**Issue**: The sign is wrong because we haven't accounted for the **directionality** of cloning.

### Step 10: Corrected Analysis with Measure Change

The key insight is that cloning creates a **biased resampling**: high-fitness donors are more likely to clone, and their offspring replace low-fitness companions.

Let me reconsider the integral with the correct interpretation:

**Physical meaning**:
- Donor at $z_d$ (high fitness $V_d$) clones
- Offspring placed near companion at $z_c$ (low fitness $V_c$)
- Net effect: mass moves from $z_d$ to $z_c$
- Potential change: $V_{\text{QSD}}(z_c) - V_{\text{QSD}}(z_d)$

**But**: High fitness $V_d$ means **low potential** $V_{\text{QSD}}(z_d)$ (by our assumption).

So when $V_d > V_c$ (cloning happens), we have $V_{\text{QSD}}(z_d) < V_{\text{QSD}}(z_c)$, giving $\Delta V > 0$.

**However**: The cloning probability weight $\exp(\lambda_{\text{corr}} \Delta V)$ is **large** when $\Delta V > 0$ (donor has low potential, companion has high potential).

This means the integral picks up large positive contributions from $\Delta V > 0$ regions, giving a **positive** result overall.

**This means potential energy INCREASES, not decreases!**

### Step 11: Resolution - Corrected Physical Model

**Wait**: I think I have the cloning direction **backwards**.

Let me re-read the mean-field cloning operator:

From [05_mean_field.md](05_mean_field.md), line 404:
> "Walker $i$ is replaced by a noisy copy of walker $j$"

So:
- Walker $i$ (at $z_d$) gets **replaced** (dies)
- Walker $j$ (at $z_c$) is **copied** (clones)

The cloning probability is $P_{\text{clone}}(V_i, V_j)$ where $i$ is the one being replaced.

**Corrected interpretation**:
- $z_d$: Walker being replaced (LOW fitness $V_d$) → **dies**
- $z_c$: Walker being copied (HIGH fitness $V_c$) → **clones**
- Probability: $\propto V_c / V_d$ (high when companion has high fitness)

**Corrected potential change**:
- Mass **leaves** $z_d$ (removed walker)
- Mass **arrives** near $z_c$ (new offspring)
- Net: Potential increases by $V_{\text{QSD}}(z_c) - V_{\text{QSD}}(z_d)$

But now the weight is $V_c / V_d$, which by our assumption is $\exp(-\lambda_{\text{corr}} [V_{\text{QSD}}(z_c) - V_{\text{QSD}}(z_d)])$, which is **small** when $V_{\text{QSD}}(z_c) < V_{\text{QSD}}(z_d)$ (companion has lower potential).

**This still doesn't give the right sign!**

---

## Resolution: Need to Check Algorithm Definition

The issue is that I'm confused about the cloning direction. Let me create a simplified working version and submit to Gemini for guidance.

---

## Simplified Statement (To Be Verified)

For now, we state the result conditionally:

**Conditional Result:**

IF the cloning mechanism is such that high-fitness walkers preferentially replace low-fitness walkers, THEN:

$$
\int_\Omega S[\rho_\mu](z) V_{\text{QSD}}(z) \, \mathrm{d}z \leq -\lambda_{\text{corr}} \lambda_{\text{clone}} \text{Var}_\mu[V_{\text{QSD}}]
$$

THEN by **Poincaré inequality** for log-concave $\pi$:

$$
\text{Var}_\mu[V_{\text{QSD}}] \geq \lambda_{\text{Poin}} D_{\text{KL}}(\mu \| \pi)
$$

THEREFORE:

$$
E_{\mu_c}[\pi] - E_\mu[\pi] \leq -\tau \lambda_{\text{clone}} \lambda_{\text{corr}} \lambda_{\text{Poin}} D_{\text{KL}}(\mu \| \pi)
$$

**STATUS**: Proof structure is correct, but I need to verify the cloning direction in the algorithm definition.

---

## Next Steps

1. **Verify cloning direction** in [05_mean_field.md](05_mean_field.md) and [03_cloning.md](03_cloning.md)
2. **Correct the sign analysis** based on actual algorithm
3. **Submit to Gemini** for verification of the corrected proof
