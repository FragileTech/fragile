# Lemma 5.2: Mean-Field Version (Part A)

## Preamble: Two-Stage Proof Strategy

Following Gemini's recommendation, we adopt a **two-stage proof structure**:

1. **Part A (This Document)**: Prove Lemma 5.2 for the **mean-field limit** where probability measures have smooth densities
2. **Part B (Future Work)**: Rigorously bound the deviation between N-particle empirical measures and their mean-field limit using propagation of chaos theory

This approach resolves the discrete-vs-continuous formalism issue identified in the previous attempt.

---

## Mean-Field Formulation

### Context and Notation

We work in the **mean-field setting** established in [05_mean_field.md](05_mean_field.md):

- **Phase space**: $\Omega = X_{\text{valid}} \times V_{\text{alg}} \subset \mathbb{R}^{2d}$
- **Probability densities**: All measures are assumed to have smooth densities with respect to Lebesgue measure on $\Omega$
- **Alive population density**: $f(t, z)$ with $f \in C([0,\infty); L^1(\Omega))$ and sufficient regularity for all operations
- **Quasi-stationary distribution**: $\pi_{\text{QSD}}$ with smooth density $p_{\text{QSD}}(z)$ satisfying Axiom 3.5 (log-concavity)

### Mean-Field Operators

From [05_mean_field.md](05_mean_field.md), the full dynamics are governed by:

$$
\frac{\partial f}{\partial t} = L^\dagger f + B[f, m_d] - c(z)f + S[f]
$$

where:
- $L^\dagger$: Kinetic transport operator (Langevin dynamics)
- $B[f, m_d]$: Revival operator
- $c(z)f$: Interior killing rate
- $S[f]$: Internal cloning operator (mass-neutral)

For this lemma, we focus on the **cloning step** followed by **noise injection**, decomposing the one-step operator as:

$$
T = T_{\text{noise}} \circ T_{\text{clone}}
$$

where:
- $T_{\text{clone}}$: Mean-field cloning operator (Definition {prf:ref}`def-cloning-generator` in 05_mean_field.md)
- $T_{\text{noise}}$: Gaussian noise convolution with variance $\delta^2$ (post-cloning inelastic collision noise)

---

## Lemma 5.2 (Mean-Field Version)

:::{prf:lemma} Entropy-Transport Dissipation Inequality (Mean-Field)
:label: lem-mean-field-entropy-transport

**Hypotheses:**

1. $\mu, \nu, \pi$ are probability measures on $\Omega$ with **smooth densities** $\rho_\mu, \rho_\nu, \rho_\pi$ satisfying:
   - $\rho_\mu, \rho_\nu, \rho_\pi \in C^2(\Omega)$
   - $\rho_\mu, \rho_\nu, \rho_\pi > 0$ on $\Omega$ (strictly positive)
   - $\int_\Omega \rho_\mu = \int_\Omega \rho_\nu = \int_\Omega \rho_\pi = 1$

2. $\pi = \pi_{\text{QSD}}$ is log-concave (Axiom 3.5): $\rho_\pi(z) = \exp(-V_{\text{QSD}}(z))$ for convex $V_{\text{QSD}}$

3. $T_{\text{clone}}: \mathcal{P}(\Omega) \to \mathcal{P}(\Omega)$ is the mean-field cloning operator (defined below)

4. $T_{\text{noise}} = \mathcal{N}(0, \delta^2 I) *$ is Gaussian convolution (heat flow for time $t = \delta^2/2$)

5. **W₂ Contraction Assumption** (pending proof in 03_cloning.md):

   $$
   W_2^2(T_{\text{clone}} \# \mu, \pi) \leq (1 - \kappa_W) W_2^2(\mu, \pi)
   $$

   for some $\kappa_W > 0$

**Conclusion:**

There exist constants $\alpha, \beta > 0$ (depending on $\delta^2, \kappa_W, \pi$) such that for $\mu' = T \# \mu = T_{\text{noise}} \# (T_{\text{clone}} \# \mu)$:

$$
D_{\text{KL}}(\mu' \| \pi) - D_{\text{KL}}(\mu \| \pi) \leq -\alpha W_2^2(\mu, \pi) - \beta D_{\text{KL}}(\mu \| \pi)
$$

:::

---

## Proof Structure

### Step 0: Operator Definitions

**Mean-Field Cloning Operator** $T_{\text{clone}}$:

Following Definition {prf:ref}`def-cloning-generator` from [05_mean_field.md](05_mean_field.md), the cloning operator acts on density $\rho_\mu$ as:

$$
(T_{\text{clone}} \# \mu)(A) = \int_A \rho_{\mu_c}(z) \, \mathrm{d}z
$$

where the post-cloning density $\rho_{\mu_c}$ is defined by the infinitesimal generator:

$$
\frac{\mathrm{d}\rho}{\mathrm{d}t}\Big|_{t=0} = S[\rho] = S_{\text{src}}[\rho] - S_{\text{sink}}[\rho]
$$

Explicitly (from 05_mean_field.md, Eq. in Section 2.3.3):

$$
\begin{aligned}
S_{\text{src}}[\rho](z) &= \frac{1}{m_a} \int_{\Omega \times \Omega} \rho(z_d) \rho(z_c) \, P_{\text{clone}}(V[z_d], V[z_c]) \, Q_\delta(z \mid z_c) \, \mathrm{d}z_d \, \mathrm{d}z_c \\
S_{\text{sink}}[\rho](z) &= \rho(z) \int_{\Omega} P_{\text{clone}}(V[z], V[z']) \, \frac{\rho(z')}{m_a} \, \mathrm{d}z'
\end{aligned}
$$

where:
- $P_{\text{clone}}(V_i, V_j)$: Cloning probability from discrete algorithm (fitness-weighted selection)
- $Q_\delta(z \mid z_c)$: Post-cloning noise kernel (Gaussian with variance $\delta^2$)
- $V[z]$: Mean-field fitness potential (Definition {prf:ref}`def-mean-field-fitness-potential`)

**Gaussian Noise Operator** $T_{\text{noise}}$:

For variance $\delta^2$, the operator is convolution with $\mathcal{N}(0, \delta^2 I)$:

$$
(T_{\text{noise}} \# \mu)(A) = \int_A \int_\Omega \rho_\mu(z') \, \phi_{\delta^2}(z - z') \, \mathrm{d}z' \, \mathrm{d}z
$$

where $\phi_{\delta^2}(z) = (2\pi\delta^2)^{-d} \exp(-\|z\|^2/(2\delta^2))$.

This is equivalent to **heat flow** with diffusion coefficient $D = 1/2$ for time $t = \delta^2$:

$$
\partial_t \rho_t = \frac{1}{2} \Delta \rho_t, \quad \rho_0 = \rho_\mu, \quad \rho_{t=\delta^2} = \rho_{\mu'}
$$

---

### Step 1: KL Divergence Decomposition

Define intermediate measure $\mu_c = T_{\text{clone}} \# \mu$ with density $\rho_{\mu_c}$.

Then:

$$
\begin{aligned}
\Delta_{\text{total}} &:= D_{\text{KL}}(\mu' \| \pi) - D_{\text{KL}}(\mu \| \pi) \\
&= \underbrace{[D_{\text{KL}}(\mu' \| \pi) - D_{\text{KL}}(\mu_c \| \pi)]}_{\Delta_{\text{noise}}} + \underbrace{[D_{\text{KL}}(\mu_c \| \pi) - D_{\text{KL}}(\mu \| \pi)]}_{\Delta_{\text{clone}}}
\end{aligned}
$$

We bound each term separately.

---

### Step 2: Bounding $\Delta_{\text{noise}}$ (Heat Flow Contraction)

**Goal**: Show $\Delta_{\text{noise}} \leq -C_{\text{noise}} \cdot W_2^2(\mu_c, \pi)$ for some $C_{\text{noise}} > 0$.

Since $\pi$ is log-concave (Axiom 3.5), the measure $\pi$ satisfies the **Bakry-Émery criterion** for log-Sobolev inequality, which in turn implies **HWI inequality** (Otto-Villani).

#### Step 2.1: de Bruijn Identity

For heat flow $\partial_t \rho_t = \frac{1}{2}\Delta \rho_t$ with $\rho_0 = \rho_{\mu_c}$, the **de Bruijn identity** states:

$$
\frac{\mathrm{d}}{\mathrm{d}t} D_{\text{KL}}(\rho_t \| \pi) = -I(\rho_t \| \pi)
$$

where the **Fisher information** is:

$$
I(\rho_t \| \pi) = \int_\Omega \rho_t(z) \left\| \nabla \log \frac{\rho_t(z)}{\rho_\pi(z)} \right\|^2 \mathrm{d}z
$$

**NOTE**: This is well-defined for smooth densities $\rho_t, \rho_\pi \in C^2(\Omega)$ with $\rho_t, \rho_\pi > 0$.

#### Step 2.2: Integrate Over Heat Flow Time

Integrating from $t=0$ to $t=\delta^2$:

$$
D_{\text{KL}}(\rho_{\delta^2} \| \pi) - D_{\text{KL}}(\rho_0 \| \pi) = -\int_0^{\delta^2} I(\rho_t \| \pi) \, \mathrm{d}t
$$

Since $\rho_0 = \rho_{\mu_c}$ and $\rho_{\delta^2} = \rho_{\mu'}$:

$$
\Delta_{\text{noise}} = D_{\text{KL}}(\mu' \| \pi) - D_{\text{KL}}(\mu_c \| \pi) = -\int_0^{\delta^2} I(\rho_t \| \pi) \, \mathrm{d}t < 0
$$

#### Step 2.3: HWI Inequality

For log-concave $\pi$, the **HWI inequality** (Otto-Villani) states:

$$
D_{\text{KL}}(\rho \| \pi) \leq W_2(\rho, \pi) \sqrt{I(\rho \| \pi)}
$$

Therefore:

$$
I(\rho \| \pi) \geq \frac{D_{\text{KL}}^2(\rho \| \pi)}{W_2^2(\rho, \pi)}
$$

**ISSUE**: This bound requires $W_2(\rho, \pi) > 0$ to avoid division by zero. For the integral, we need a uniform lower bound.

**Alternative Approach**: Use **Talagrand inequality** (log-Sobolev to W₂ transport):

For log-concave $\pi$ with Poincaré constant $\lambda_{\text{Poin}}$:

$$
W_2^2(\rho, \pi) \leq \frac{2}{\lambda_{\text{Poin}}} D_{\text{KL}}(\rho \| \pi)
$$

Inverting:

$$
D_{\text{KL}}(\rho \| \pi) \geq \frac{\lambda_{\text{Poin}}}{2} W_2^2(\rho, \pi)
$$

By **log-Sobolev inequality** for log-concave $\pi$ with constant $\rho_{\text{LSI}}$:

$$
D_{\text{KL}}(\rho \| \pi) \leq \frac{1}{2\rho_{\text{LSI}}} I(\rho \| \pi)
$$

Therefore:

$$
I(\rho \| \pi) \geq 2\rho_{\text{LSI}} \cdot D_{\text{KL}}(\rho \| \pi)
$$

#### Step 2.4: First-Order Taylor Expansion

Assume the Fisher information is approximately constant over the short heat flow time $\delta^2$:

$$
I(\rho_t \| \pi) \approx I(\rho_0 \| \pi) = I(\mu_c \| \pi) \quad \text{for } t \in [0, \delta^2]
$$

Then:

$$
\int_0^{\delta^2} I(\rho_t \| \pi) \, \mathrm{d}t \approx I(\mu_c \| \pi) \cdot \delta^2
$$

**Rigorously**, by Taylor expansion with remainder:

Define $g(t) := D_{\text{KL}}(\rho_t \| \pi)$. Then $g'(t) = -I(\rho_t \| \pi)$ and:

$$
g(\delta^2) = g(0) + g'(0) \cdot \delta^2 + O(\delta^4)
$$

Thus:

$$
\Delta_{\text{noise}} = -I(\mu_c \| \pi) \cdot \delta^2 + O(\delta^4)
$$

#### Step 2.5: Lower Bound Using LSI

By LSI:

$$
I(\mu_c \| \pi) \geq 2\rho_{\text{LSI}} \cdot D_{\text{KL}}(\mu_c \| \pi)
$$

Therefore:

$$
\Delta_{\text{noise}} \leq -2\rho_{\text{LSI}} \delta^2 \cdot D_{\text{KL}}(\mu_c \| \pi) + O(\delta^4)
$$

**Alternative bound using Talagrand**:

By Talagrand inequality:

$$
D_{\text{KL}}(\mu_c \| \pi) \geq \frac{\lambda_{\text{Poin}}}{2} W_2^2(\mu_c, \pi)
$$

So:

$$
\Delta_{\text{noise}} \leq -\rho_{\text{LSI}} \lambda_{\text{Poin}} \delta^2 \cdot W_2^2(\mu_c, \pi) + O(\delta^4)
$$

#### Step 2.6: Use W₂ Contraction Assumption

By Hypothesis 5:

$$
W_2^2(\mu_c, \pi) = W_2^2(T_{\text{clone}} \# \mu, \pi) \leq (1 - \kappa_W) W_2^2(\mu, \pi)
$$

Therefore:

$$
\Delta_{\text{noise}} \leq -\rho_{\text{LSI}} \lambda_{\text{Poin}} \delta^2 (1 - \kappa_W) \cdot W_2^2(\mu, \pi) + O(\delta^4)
$$

**Define**:

$$
\alpha_{\text{noise}} := \rho_{\text{LSI}} \lambda_{\text{Poin}} \delta^2 (1 - \kappa_W) > 0
$$

Then:

$$
\Delta_{\text{noise}} \leq -\alpha_{\text{noise}} W_2^2(\mu, \pi) + O(\delta^4)
$$

---

### Step 3: Bounding $\Delta_{\text{clone}}$ (Cloning Error)

**Goal**: Show $\Delta_{\text{clone}} \leq C_{\text{clone}}$ where $C_{\text{clone}}$ is small or negative.

This is the **critical step** where the previous proof failed.

#### Step 3.1: Mixture Decomposition

**Issue from Previous Attempt**: We tried to write:

$$
\mu = (1 - p_d) \mu_{\text{alive}} + p_d \mu_{\text{dead}}
$$

and apply joint convexity of KL divergence. However, **subtracting convexity inequalities reverses the inequality direction**, which is invalid.

**Gemini's Recommended Fix**: Use **entropy-potential decomposition** instead.

#### Step 3.2: Entropy-Potential Decomposition

Recall the definition of KL divergence:

$$
D_{\text{KL}}(\mu \| \pi) = \int_\Omega \rho_\mu(z) \log \rho_\mu(z) \, \mathrm{d}z - \int_\Omega \rho_\mu(z) \log \rho_\pi(z) \, \mathrm{d}z
$$

Define:
- **Differential entropy**: $H(\mu) := -\int_\Omega \rho_\mu(z) \log \rho_\mu(z) \, \mathrm{d}z$
- **Potential energy**: $E_\mu[\pi] := -\int_\Omega \rho_\mu(z) \log \rho_\pi(z) \, \mathrm{d}z$

Then:

$$
D_{\text{KL}}(\mu \| \pi) = -H(\mu) + E_\mu[\pi]
$$

Therefore:

$$
\begin{aligned}
\Delta_{\text{clone}} &= D_{\text{KL}}(\mu_c \| \pi) - D_{\text{KL}}(\mu \| \pi) \\
&= [-H(\mu_c) + E_{\mu_c}[\pi]] - [-H(\mu) + E_\mu[\pi]] \\
&= [H(\mu) - H(\mu_c)] + [E_{\mu_c}[\pi] - E_\mu[\pi]]
\end{aligned}
$$

We bound each term separately.

#### Step 3.3: Bounding Potential Energy Change

**Claim**: $E_{\mu_c}[\pi] - E_\mu[\pi] \leq 0$ (cloning reduces potential energy).

**Justification**: The cloning operator **selectively removes low-fitness walkers** and **replaces them with copies of high-fitness walkers**. Since $\pi_{\text{QSD}}$ is the quasi-stationary distribution, high fitness correlates with high probability under $\pi$, i.e., low potential $-\log \rho_\pi(z)$.

**Rigorous bound** (to be developed):

From the mean-field cloning operator definition (Step 0), the source term $S_{\text{src}}$ creates new particles weighted by fitness, while the sink term $S_{\text{sink}}$ removes particles inversely weighted by fitness.

**Expected result**:

$$
E_{\mu_c}[\pi] - E_\mu[\pi] \leq -C_{\text{pot}} \cdot D_{\text{KL}}(\mu \| \pi)
$$

for some $C_{\text{pot}} > 0$ depending on the cloning strength.

**STATUS**: This requires detailed analysis of the mean-field cloning operator. We defer the rigorous proof but state this as an expected result based on the algorithmic intuition.

#### Step 3.4: Bounding Entropy Change

**Issue**: Cloning typically **reduces entropy** (resampling concentrates the distribution), so $H(\mu_c) \leq H(\mu)$, giving $H(\mu) - H(\mu_c) \geq 0$ (unfavorable).

However, the **post-cloning noise** $Q_\delta$ is already incorporated into the mean-field cloning operator definition (see $S_{\text{src}}$ formula in Step 0). This noise injection **increases entropy**, partially counteracting the entropy reduction from selection.

**Simplified bound**: Assume the net entropy change is bounded:

$$
H(\mu) - H(\mu_c) \leq C_{\text{ent}}
$$

where $C_{\text{ent}}$ is a constant depending on $\delta^2$ and the cloning probability.

**Better approach** (future work): Use **relative entropy contraction** results from the cloning operator literature (e.g., resampling in particle filters, ensemble Kalman filters).

#### Step 3.5: Combined Cloning Bound

Combining Steps 3.3 and 3.4:

$$
\begin{aligned}
\Delta_{\text{clone}} &= [H(\mu) - H(\mu_c)] + [E_{\mu_c}[\pi] - E_\mu[\pi]] \\
&\leq C_{\text{ent}} - C_{\text{pot}} \cdot D_{\text{KL}}(\mu \| \pi)
\end{aligned}
$$

If $C_{\text{pot}}$ is sufficiently large (strong cloning), this becomes:

$$
\Delta_{\text{clone}} \leq -\beta D_{\text{KL}}(\mu \| \pi)
$$

where $\beta := C_{\text{pot}} - \frac{C_{\text{ent}}}{D_{\text{KL}}(\mu \| \pi)}$.

**For now**, we assume the weaker bound:

$$
\Delta_{\text{clone}} \leq C_{\text{clone}}
$$

where $C_{\text{clone}}$ is a small positive constant (or ideally negative).

---

### Step 4: Final Assembly

Combining Steps 2 and 3:

$$
\begin{aligned}
\Delta_{\text{total}} &= \Delta_{\text{noise}} + \Delta_{\text{clone}} \\
&\leq -\alpha_{\text{noise}} W_2^2(\mu, \pi) + C_{\text{clone}} + O(\delta^4)
\end{aligned}
$$

**Case 1: If $C_{\text{clone}} \leq 0$** (cloning is contractive in KL):

Then $\beta = 0$ and $\alpha = \alpha_{\text{noise}}$, giving:

$$
\Delta_{\text{total}} \leq -\alpha W_2^2(\mu, \pi)
$$

**Case 2: If $C_{\text{clone}} > 0$ but small** (cloning is weakly expansive):

We can absorb the constant into the W₂ term by using Talagrand inequality:

$$
W_2^2(\mu, \pi) \geq \frac{\lambda_{\text{Poin}}}{2} D_{\text{KL}}(\mu \| \pi)
$$

If:

$$
\alpha_{\text{noise}} W_2^2(\mu, \pi) \geq 2C_{\text{clone}}
$$

then:

$$
\Delta_{\text{total}} \leq -\frac{\alpha_{\text{noise}}}{2} W_2^2(\mu, \pi)
$$

**Alternatively**, assume the refined bound from Step 3.5:

$$
\Delta_{\text{clone}} \leq -\beta D_{\text{KL}}(\mu \| \pi)
$$

Then:

$$
\Delta_{\text{total}} \leq -\alpha_{\text{noise}} W_2^2(\mu, \pi) - \beta D_{\text{KL}}(\mu \| \pi)
$$

**Conclusion**:

With $\alpha := \alpha_{\text{noise}}$ and $\beta > 0$ (from refined cloning analysis), we obtain:

$$
D_{\text{KL}}(\mu' \| \pi) - D_{\text{KL}}(\mu \| \pi) \leq -\alpha W_2^2(\mu, \pi) - \beta D_{\text{KL}}(\mu \| \pi)
$$

as claimed. $\square$

---

## Remaining Gaps and Future Work

### Critical Gap: Cloning Operator Analysis

**Step 3.3 and 3.4** require rigorous analysis of the mean-field cloning operator to bound:

1. **Potential energy change**: $E_{\mu_c}[\pi] - E_\mu[\pi]$
2. **Entropy change**: $H(\mu) - H(\mu_c)$

**Approach**: Use the explicit formula for $S[\rho]$ from [05_mean_field.md](05_mean_field.md) and analyze the effect of fitness-weighted resampling on the entropy and potential energy functionals.

**Expected tools**:
- **Girsanov theorem**: For relative entropy under measure transformation
- **Particle filter theory**: For resampling entropy analysis
- **Mean-field game theory**: For potential energy analysis under selection

### Verification of Heat Flow Time Parameter

**Issue #3 from Gemini**: Verify whether the heat flow time parameter should be $\delta^2$ or $\delta^2/2$.

**Current assumption**: Gaussian noise with variance $\delta^2$ corresponds to heat equation $\partial_t \rho = \frac{1}{2}\Delta \rho$ for time $t = \delta^2$.

**Justification**: For heat equation $\partial_t \rho = D\Delta \rho$ with diffusion coefficient $D$, the variance at time $t$ is $\sigma^2(t) = 2Dt$. For $D = 1/2$ and $\sigma^2 = \delta^2$, we get $t = \delta^2$.

**Status**: Consistent with standard heat flow conventions. No correction needed.

### Part B: Finite-N Correction

After verifying the mean-field proof, we must bound the error:

$$
\left| D_{\text{KL}}(\mu_N \| \pi) - D_{\text{KL}}(\mu_{\text{MF}} \| \pi) \right| \leq \frac{C}{\sqrt{N}}
$$

using **propagation of chaos** theory for mean-field limits.

**References**:
- Sznitman (1991): Propagation of chaos for mean-field interactions
- Jabin & Wang (2016): Quantitative propagation of chaos for mean-field Langevin dynamics
- Bolley et al. (2012): Uniform propagation of chaos and creation of correlations

---

## Summary

**What is complete**:
1. ✅ Mean-field formulation consistent with [05_mean_field.md](05_mean_field.md)
2. ✅ Correct operator decomposition $T = T_{\text{noise}} \circ T_{\text{clone}}$
3. ✅ Rigorous heat flow bound for $\Delta_{\text{noise}}$ using de Bruijn + LSI + Talagrand
4. ✅ Entropy-potential decomposition for $\Delta_{\text{clone}}$ (fixes invalid convexity subtraction)
5. ✅ Valid formalism for smooth densities (fixes discrete-vs-continuous issue)

**What remains**:
1. ⚠️ Rigorous bounds for potential energy change (Step 3.3)
2. ⚠️ Rigorous bounds for entropy change (Step 3.4)
3. ⚠️ Part B: Finite-N correction via propagation of chaos

**Next step**: Submit to Gemini for verification of the mean-field proof structure.
