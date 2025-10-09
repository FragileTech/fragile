# Step 3 Completion: Rigorous Cloning Operator Bounds

## Context

This document provides rigorous derivations for the two critical bounds needed in Lemma 5.2 (Mean-Field Version), Step 3:

1. **Potential Energy Change** (Step 3.3): $E_{\mu_c}[\pi] - E_\mu[\pi]$
2. **Entropy Change** (Step 3.4): $H(\mu) - H(\mu_c)$

These bounds complete the proof of $\Delta_{\text{clone}} = D_{\text{KL}}(\mu_c \| \pi) - D_{\text{KL}}(\mu \| \pi)$.

---

## Notation and Setup

From [10_C_lemma5.2_meanfield.md](10_C_lemma5.2_meanfield.md):

**Measures:**
- $\mu$: Pre-cloning measure with smooth density $\rho_\mu$
- $\mu_c$: Post-cloning measure with smooth density $\rho_{\mu_c}$
- $\pi$: QSD measure with smooth density $\rho_\pi = \exp(-V_{\text{QSD}})$ (log-concave)

**Mean-Field Cloning Operator** (from [05_mean_field.md](05_mean_field.md)):

The post-cloning density $\rho_{\mu_c}$ is determined by the infinitesimal generator $S[\rho]$:

$$
\rho_{\mu_c} = \rho_\mu + \tau \cdot S[\rho_\mu] + O(\tau^2)
$$

where for infinitesimal time step $\tau \to 0$:

$$
\begin{aligned}
S[\rho](z) &= S_{\text{src}}[\rho](z) - S_{\text{sink}}[\rho](z) \\
S_{\text{src}}[\rho](z) &= \frac{1}{m_a} \int_{\Omega \times \Omega} \rho(z_d) \rho(z_c) \, P_{\text{clone}}(V[z_d], V[z_c]) \, Q_\delta(z \mid z_c) \, \mathrm{d}z_d \, \mathrm{d}z_c \\
S_{\text{sink}}[\rho](z) &= \rho(z) \int_{\Omega} P_{\text{clone}}(V[z], V[z']) \, \frac{\rho(z')}{m_a} \, \mathrm{d}z'
\end{aligned}
$$

**Cloning Probability:**

From the discrete algorithm, the cloning probability for a "donor" walker at $z_d$ to clone and replace a "companion" at $z_c$ is fitness-weighted:

$$
P_{\text{clone}}(V_d, V_c) \propto \frac{V_d}{V_c}
$$

More precisely, from the mean-field limit:

$$
P_{\text{clone}}(V_d, V_c) = \min\left(1, \frac{V_d}{V_c}\right) \cdot \lambda_{\text{clone}}
$$

where $\lambda_{\text{clone}}$ is the cloning rate and $V[z]$ is the mean-field fitness potential.

**Key Observation:** High fitness $V[z]$ correlates with high QSD probability $\rho_\pi(z)$ (low potential $V_{\text{QSD}}(z) = -\log \rho_\pi(z)$).

---

## Part A: Potential Energy Change Bound

### Goal

Prove:

$$
E_{\mu_c}[\pi] - E_\mu[\pi] \leq -C_{\text{pot}} \cdot D_{\text{KL}}(\mu \| \pi)
$$

for some $C_{\text{pot}} > 0$.

### Strategy

The cloning operator:
1. **Removes** mass from low-fitness regions (high potential under $\pi$)
2. **Adds** mass to high-fitness regions (low potential under $\pi$)

Therefore, the expected potential energy $E_\mu[\pi] = -\mathbb{E}_\mu[\log \rho_\pi]$ should **decrease** after cloning.

### Step A.1: Potential Energy Definition

Recall:

$$
\begin{aligned}
E_\mu[\pi] &:= -\int_\Omega \rho_\mu(z) \log \rho_\pi(z) \, \mathrm{d}z \\
&= \int_\Omega \rho_\mu(z) V_{\text{QSD}}(z) \, \mathrm{d}z
\end{aligned}
$$

since $\rho_\pi(z) = \exp(-V_{\text{QSD}}(z))$.

### Step A.2: Infinitesimal Change in Potential Energy

For small time step $\tau$:

$$
\rho_{\mu_c}(z) = \rho_\mu(z) + \tau \cdot S[\rho_\mu](z) + O(\tau^2)
$$

Therefore:

$$
\begin{aligned}
E_{\mu_c}[\pi] - E_\mu[\pi] &= \int_\Omega [\rho_{\mu_c}(z) - \rho_\mu(z)] V_{\text{QSD}}(z) \, \mathrm{d}z \\
&= \tau \int_\Omega S[\rho_\mu](z) \, V_{\text{QSD}}(z) \, \mathrm{d}z + O(\tau^2) \\
&= \tau \int_\Omega [S_{\text{src}}[\rho_\mu](z) - S_{\text{sink}}[\rho_\mu](z)] V_{\text{QSD}}(z) \, \mathrm{d}z + O(\tau^2)
\end{aligned}
$$

### Step A.3: Expand Source Term

$$
\begin{aligned}
&\int_\Omega S_{\text{src}}[\rho_\mu](z) \, V_{\text{QSD}}(z) \, \mathrm{d}z \\
&= \int_\Omega \left[\frac{1}{m_a} \int_{\Omega \times \Omega} \rho_\mu(z_d) \rho_\mu(z_c) \, P_{\text{clone}}(V[z_d], V[z_c]) \, Q_\delta(z \mid z_c) \, \mathrm{d}z_d \, \mathrm{d}z_c\right] V_{\text{QSD}}(z) \, \mathrm{d}z
\end{aligned}
$$

**Rearrange integrals** (Fubini):

$$
= \frac{1}{m_a} \int_{\Omega \times \Omega} \rho_\mu(z_d) \rho_\mu(z_c) \, P_{\text{clone}}(V[z_d], V[z_c]) \left[\int_\Omega Q_\delta(z \mid z_c) V_{\text{QSD}}(z) \, \mathrm{d}z\right] \mathrm{d}z_d \, \mathrm{d}z_c
$$

**Key observation:** The inner integral is the **expected potential of offspring** given parent at $z_c$:

$$
\bar{V}(z_c) := \int_\Omega Q_\delta(z \mid z_c) V_{\text{QSD}}(z) \, \mathrm{d}z
$$

Since $Q_\delta$ is Gaussian noise with small variance $\delta^2$:

$$
\bar{V}(z_c) \approx V_{\text{QSD}}(z_c) + O(\delta^2)
$$

**First-order approximation:**

$$
\int_\Omega S_{\text{src}}[\rho_\mu](z) \, V_{\text{QSD}}(z) \, \mathrm{d}z \approx \frac{1}{m_a} \int_{\Omega \times \Omega} \rho_\mu(z_d) \rho_\mu(z_c) \, P_{\text{clone}}(V[z_d], V[z_c]) \, V_{\text{QSD}}(z_c) \, \mathrm{d}z_d \, \mathrm{d}z_c
$$

### Step A.4: Expand Sink Term

$$
\begin{aligned}
&\int_\Omega S_{\text{sink}}[\rho_\mu](z) \, V_{\text{QSD}}(z) \, \mathrm{d}z \\
&= \int_\Omega \rho_\mu(z) V_{\text{QSD}}(z) \left[\int_\Omega P_{\text{clone}}(V[z], V[z']) \, \frac{\rho_\mu(z')}{m_a} \, \mathrm{d}z'\right] \mathrm{d}z
\end{aligned}
$$

**Rearrange** (Fubini, relabel $z \to z_d$, $z' \to z_c$):

$$
= \frac{1}{m_a} \int_{\Omega \times \Omega} \rho_\mu(z_d) \rho_\mu(z_c) \, P_{\text{clone}}(V[z_d], V[z_c]) \, V_{\text{QSD}}(z_d) \, \mathrm{d}z_d \, \mathrm{d}z_c
$$

### Step A.5: Combine Source and Sink

$$
\begin{aligned}
&\int_\Omega S[\rho_\mu](z) \, V_{\text{QSD}}(z) \, \mathrm{d}z \\
&= \frac{1}{m_a} \int_{\Omega \times \Omega} \rho_\mu(z_d) \rho_\mu(z_c) \, P_{\text{clone}}(V[z_d], V[z_c]) \, [V_{\text{QSD}}(z_c) - V_{\text{QSD}}(z_d)] \, \mathrm{d}z_d \, \mathrm{d}z_c
\end{aligned}
$$

**Interpretation:**
- Cloning replaces walker at $z_d$ (donor is cloned, dies) with offspring near $z_c$ (companion's location)
- Potential energy change per cloning event: $V_{\text{QSD}}(z_c) - V_{\text{QSD}}(z_d)$

**KEY INSIGHT:** Cloning probability $P_{\text{clone}}(V_d, V_c) \propto V_d / V_c$ favors high $V_d$ (high fitness donor) and low $V_c$ (low fitness companion).

For log-concave QSD: high fitness $V_d$ correlates with LOW potential $V_{\text{QSD}}(z_d)$.

### Step A.6: Relate Fitness to Potential

**Axiom 3.5 Log-Concavity:** $\pi_{\text{QSD}}$ is the stationary distribution of the system, with:

$$
\rho_\pi(z) = \exp(-V_{\text{QSD}}(z))
$$

**Fitness Potential Correlation:** The fitness $V[z]$ is designed to drive the system toward $\pi_{\text{QSD}}$. The strongest form of this correlation is:

**Assumption A.1** (Fitness-Potential Anti-Correlation):

There exists $\kappa_{\text{corr}} > 0$ such that:

$$
V[z] \geq V_{\text{ref}} - \kappa_{\text{corr}} V_{\text{QSD}}(z)
$$

for some reference fitness $V_{\text{ref}}$.

**Rationale:** High QSD probability $\rho_\pi$ (low $V_{\text{QSD}}$) should correspond to high fitness $V$.

**Consequence:**

$$
\frac{V[z_d]}{V[z_c]} \geq \exp\left(\kappa_{\text{corr}} (V_{\text{QSD}}(z_d) - V_{\text{QSD}}(z_c))\right) \quad \text{(approximately)}
$$

So cloning favors $V_{\text{QSD}}(z_d) < V_{\text{QSD}}(z_c)$, i.e., **donor has lower potential than companion**.

### Step A.7: Simplified Bound (Heuristic)

For $P_{\text{clone}} \propto V_d / V_c$ and using Assumption A.1:

$$
\begin{aligned}
&\frac{1}{m_a} \int_{\Omega \times \Omega} \rho_\mu(z_d) \rho_\mu(z_c) \, P_{\text{clone}}(V[z_d], V[z_c]) \, [V_{\text{QSD}}(z_c) - V_{\text{QSD}}(z_d)] \, \mathrm{d}z_d \, \mathrm{d}z_c \\
&\approx \frac{\lambda_{\text{clone}}}{m_a} \int_{\Omega \times \Omega} \rho_\mu(z_d) \rho_\mu(z_c) \, \frac{V[z_d]}{V[z_c]} \, [V_{\text{QSD}}(z_c) - V_{\text{QSD}}(z_d)] \, \mathrm{d}z_d \, \mathrm{d}z_c
\end{aligned}
$$

**Claim:** This integral is **negative** because:
1. Weight $V_d / V_c$ is large when $V_d > V_c$
2. By correlation, $V_d > V_c$ implies $V_{\text{QSD}}(z_d) < V_{\text{QSD}}(z_c)$
3. Therefore $V_{\text{QSD}}(z_c) - V_{\text{QSD}}(z_d) > 0$ when weight is large
4. Net effect: negative contribution

**Rigorous bound** (to be developed):

Using the Kullback-Leibler divergence relationship and covariance bounds, one can show:

$$
\int_\Omega S[\rho_\mu](z) \, V_{\text{QSD}}(z) \, \mathrm{d}z \leq -C_{\text{pot}}' \cdot \text{Var}_\mu[V_{\text{QSD}}]
$$

where $\text{Var}_\mu[V_{\text{QSD}}]$ is the variance of the potential under $\mu$.

For measures far from equilibrium, $\text{Var}_\mu[V_{\text{QSD}}]$ is large, and we can relate it to $D_{\text{KL}}(\mu \| \pi)$ via:

$$
D_{\text{KL}}(\mu \| \pi) = \mathbb{E}_\mu[V_{\text{QSD}}] - \mathbb{E}_\pi[V_{\text{QSD}}] - H(\mu) + H(\pi)
$$

For strongly log-concave $\pi$, variance and KL divergence are related by Poincaré inequality.

**WORKING BOUND** (pending rigorous derivation):

$$
E_{\mu_c}[\pi] - E_\mu[\pi] \leq -\tau \lambda_{\text{clone}} \kappa_{\text{eff}} \cdot D_{\text{KL}}(\mu \| \pi)
$$

where $\kappa_{\text{eff}} > 0$ depends on $\kappa_{\text{corr}}$ and the log-Sobolev constant of $\pi$.

---

## Part B: Entropy Change Bound

### Goal

Bound:

$$
\Delta H := H(\mu) - H(\mu_c)
$$

where $H(\mu) = -\int_\Omega \rho_\mu(z) \log \rho_\mu(z) \, \mathrm{d}z$.

### Challenge

**Selection reduces entropy:** Pure resampling (without noise) concentrates the distribution, so $H(\mu_c) < H(\mu)$ (unfavorable).

**Noise increases entropy:** The post-cloning noise $Q_\delta$ in $S_{\text{src}}$ adds diffusion, increasing entropy.

**Net effect:** Depends on balance between selection strength and noise strength.

### Step B.1: Infinitesimal Entropy Change

For $\rho_{\mu_c} = \rho_\mu + \tau S[\rho_\mu] + O(\tau^2)$:

$$
H(\mu_c) = H(\mu) + \tau \frac{\mathrm{d}H}{\mathrm{d}t}\Big|_{\rho=\rho_\mu} + O(\tau^2)
$$

where:

$$
\frac{\mathrm{d}H}{\mathrm{d}t} = -\int_\Omega S[\rho](z) \, [\log \rho(z) + 1] \, \mathrm{d}z
$$

Therefore:

$$
\Delta H = H(\mu) - H(\mu_c) = -\tau \int_\Omega S[\rho_\mu](z) \, [\log \rho_\mu(z) + 1] \, \mathrm{d}z + O(\tau^2)
$$

### Step B.2: Decompose Cloning Operator Contribution

$$
\begin{aligned}
-\int_\Omega S[\rho](z) \, \log \rho(z) \, \mathrm{d}z &= -\int_\Omega [S_{\text{src}}[\rho](z) - S_{\text{sink}}[\rho](z)] \log \rho(z) \, \mathrm{d}z \\
&= \underbrace{-\int_\Omega S_{\text{src}}[\rho](z) \log \rho(z) \, \mathrm{d}z}_{\text{Source entropy}} + \underbrace{\int_\Omega S_{\text{sink}}[\rho](z) \log \rho(z) \, \mathrm{d}z}_{\text{Sink entropy}}
\end{aligned}
$$

### Step B.3: Source Term (Entropy Injection from Noise)

$$
\begin{aligned}
&-\int_\Omega S_{\text{src}}[\rho](z) \log \rho(z) \, \mathrm{d}z \\
&= -\frac{1}{m_a} \int_{\Omega^3} \rho(z_d) \rho(z_c) P_{\text{clone}}(V_d, V_c) Q_\delta(z \mid z_c) \log \rho(z) \, \mathrm{d}z_d \, \mathrm{d}z_c \, \mathrm{d}z
\end{aligned}
$$

**Key:** The offspring at $z$ is drawn from $Q_\delta(z \mid z_c)$ (Gaussian around parent $z_c$).

For small $\delta^2$, $\rho(z) \approx \rho(z_c)$ for $z$ in the support of $Q_\delta(\cdot | z_c)$.

**Approximation:**

$$
-\int_\Omega S_{\text{src}}[\rho](z) \log \rho(z) \, \mathrm{d}z \approx -\frac{1}{m_a} \int_{\Omega \times \Omega} \rho(z_d) \rho(z_c) P_{\text{clone}}(V_d, V_c) \log \rho(z_c) \, \mathrm{d}z_d \, \mathrm{d}z_c
$$

**But this ignores the entropy injection from noise!**

### Step B.4: Entropy Injection from Gaussian Noise

The noise kernel $Q_\delta(z \mid z_c)$ is Gaussian with variance $\delta^2$. Convolving a density with Gaussian **increases differential entropy** by:

$$
\Delta H_{\text{Gaussian}} = \frac{d}{2} \log(2\pi e \delta^2)
$$

for each dimension.

**More carefully:** For $\rho_{\text{noisy}}(z) = \int \rho_0(z') Q_\delta(z - z') \, \mathrm{d}z'$:

$$
H(\rho_{\text{noisy}}) \geq H(\rho_0) + \frac{d}{2} \log(2\pi e \delta^2)
$$

by the **entropy power inequality** (Shannon).

### Step B.5: Combined Effect

**Selection entropy loss** (sink > source without noise):

$$
\Delta H_{\text{selection}} \approx -\tau \lambda_{\text{clone}} \cdot \text{Var}_\mu[\log \rho_\mu]
$$

(Resampling reduces entropy proportional to variance of log-density)

**Noise entropy gain:**

$$
\Delta H_{\text{noise}} \approx +\tau \lambda_{\text{clone}} \cdot \frac{d}{2} \log(2\pi e \delta^2)
$$

**Net entropy change:**

$$
\Delta H = H(\mu) - H(\mu_c) \approx \tau \lambda_{\text{clone}} \left[\text{Var}_\mu[\log \rho_\mu] - \frac{d}{2} \log(2\pi e \delta^2)\right]
$$

**Bounded entropy change:**

If $\delta^2$ is sufficiently large (strong noise):

$$
\Delta H \leq \tau \lambda_{\text{clone}} C_{\text{ent}}
$$

where $C_{\text{ent}}$ can be bounded using the regularity of $\rho_\mu$ (smoothness assumption).

---

## Summary and Next Steps

### What We Have

**Part A (Potential Energy):**
- Heuristic argument that cloning reduces potential energy
- Structure for rigorous bound using fitness-potential correlation
- **Needs:** Formal proof of Assumption A.1 or alternative argument

**Part B (Entropy):**
- Decomposition into selection loss + noise gain
- Entropy power inequality for noise contribution
- **Needs:** Rigorous control of variance terms and second-order corrections

### Required for Completion

1. **Prove or assume fitness-potential correlation** (Assumption A.1)
   - Check if this follows from QSD definition
   - Or state as additional axiom with plausibility argument

2. **Rigorously bound variance terms** using:
   - Poincaré inequality for $\text{Var}_\mu[V_{\text{QSD}}]$
   - Entropy power inequality for noise injection
   - Regularity bounds for $\text{Var}_\mu[\log \rho_\mu]$

3. **Combine bounds** to show:
   $$
   \Delta_{\text{clone}} = \Delta H + [E_{\mu_c}[\pi] - E_\mu[\pi]] \leq -\beta \cdot D_{\text{KL}}(\mu \| \pi)
   $$
   for $\beta > 0$ when cloning strength dominates entropy fluctuations

### Literature to Consult

- **Particle Filter Resampling:** Del Moral et al. on entropy evolution under resampling
- **Mean-Field Games:** Lasry-Lions on potential energy in mean-field systems
- **Information Geometry:** Amari on Fisher information and entropy in selection dynamics
- **Optimal Transport:** Villani on entropy along Wasserstein geodesics

---

## Status

**Current:** Proof structure established, heuristic arguments complete.

**Next:** Develop rigorous bounds using tools from probability theory, information theory, and optimal transport.

**Estimated effort:** 5-10 iterations with Gemini to verify technical details.
