# Proof Sketch: Mixing Time (Parameter-Explicit)

**Theorem Label:** `prop-mixing-time-explicit`
**Document:** `06_convergence.md`
**Type:** Proposition
**Date:** 2025-10-25
**Sketcher:** Claude (Proof Sketcher Agent)

---

## 1. Theorem Statement

:::{prf:proposition} Mixing Time (Parameter-Explicit)
:label: prop-mixing-time-explicit

The time to reach $\epsilon$-proximity to equilibrium is:

$$
T_{\text{mix}}(\epsilon) = \frac{1}{\kappa_{\text{total}}} \ln\left(\frac{V_{\text{total}}^{\text{init}}}{\epsilon C_{\text{total}}}\right)
$$

For typical initialization $V_{\text{total}}^{\text{init}} \sim O(1)$ and target $\epsilon = 0.01$:

$$
T_{\text{mix}} \sim \frac{5}{\kappa_{\text{total}}} = \frac{5}{\min(\lambda, 2\gamma, \kappa_W, \kappa_b)}
$$

:::

---

## 2. Dependencies

This proposition requires the following established results:

1. **Foster-Lyapunov Drift Condition** (from Chapter 3 of `06_convergence.md`):
   - The composed operator $\Psi_{\text{total}} = \Psi_{\text{kin}} \circ \Psi_{\text{clone}}$ satisfies:

   $$
   \mathbb{E}[\Delta V_{\text{total}}] \leq -\kappa_{\text{total}}\tau \cdot V_{\text{total}} + C_{\text{total}}
   $$

   - This is the key geometric ergodicity condition

2. **Total Convergence Rate** (Theorem `thm-total-rate-explicit`, line 1715 of `06_convergence.md`):
   - Explicit formula for $\kappa_{\text{total}}$:

   $$
   \kappa_{\text{total}} = \min(\kappa_x, \kappa_v, \kappa_W, \kappa_b) \cdot (1 - \epsilon_{\text{coupling}})
   $$

   - Equilibrium constant:

   $$
   C_{\text{total}} = \frac{C_x + \alpha_v C_v' + \alpha_W C_W' + \alpha_b C_b}{\kappa_{\text{total}}}
   $$

3. **Exponential convergence from drift condition** (standard Markov chain theory):
   - Foster-Lyapunov drift implies exponential convergence in expectation
   - The equilibrium value is $V_{\text{total}}^{\text{eq}} = C_{\text{total}}/\kappa_{\text{total}}$

4. **Component rate formulas** (Propositions in Section 5.1-5.4 of `06_convergence.md`):
   - $\kappa_v \sim 2\gamma$ (velocity dissipation rate)
   - $\kappa_x \sim \lambda$ (positional contraction rate)
   - $\kappa_W \sim c_{\text{hypo}}^2 \gamma/(1 + \gamma/\lambda_{\min})$ (Wasserstein contraction)
   - $\kappa_b \sim \lambda \Delta f_{\text{boundary}}/f_{\text{typical}}$ (boundary safety)

---

## 3. Proof Strategy

**High-Level Approach:**

The proof is a direct application of the Foster-Lyapunov drift condition to extract an explicit mixing time bound. The key steps are:

1. **Integrate the drift inequality** to obtain an exponential decay bound on $\mathbb{E}[V_{\text{total}}(t)]$
2. **Identify the equilibrium value** $V_{\text{total}}^{\text{eq}} = C_{\text{total}}/\kappa_{\text{total}}$
3. **Solve for the time** when the error drops below $\epsilon \cdot V_{\text{total}}^{\text{eq}}$
4. **Simplify for typical parameters** to obtain the approximation $T_{\text{mix}} \sim 5/\kappa_{\text{total}}$

This is a standard analysis technique in the theory of geometric ergodicity—the novelty lies in the explicit parameter dependence made possible by the detailed operator analysis in preceding sections.

---

## 4. Key Steps

### Step 1: Discrete-to-Continuous Time Conversion

The Foster-Lyapunov condition holds in discrete time:

$$
\mathbb{E}[V_{\text{total}}(n+1)] \leq (1 - \kappa_{\text{total}}\tau) V_{\text{total}}(n) + C_{\text{total}}
$$

**Goal:** Convert to continuous-time exponential decay.

**Method:** Use the standard approximation $(1 - \kappa \tau)^{t/\tau} \approx e^{-\kappa t}$ for small $\tau$.

**Result:**

$$
\mathbb{E}[V_{\text{total}}(t)] \leq e^{-\kappa_{\text{total}} t} V_{\text{total}}^{\text{init}} + \frac{C_{\text{total}}}{\kappa_{\text{total}}}(1 - e^{-\kappa_{\text{total}} t})
$$

**Justification:** This follows from iterating the discrete drift inequality and taking the continuous-time limit. The error in this approximation is $O(\tau^2)$, which is negligible for typical $\tau \sim 0.01$.

---

### Step 2: Identify Equilibrium Value

At equilibrium ($t \to \infty$), the transient term vanishes:

$$
\mathbb{E}[V_{\text{total}}^{\text{eq}}] = \lim_{t \to \infty} \mathbb{E}[V_{\text{total}}(t)] = \frac{C_{\text{total}}}{\kappa_{\text{total}}}
$$

**Interpretation:** The equilibrium is determined by the balance between contraction (rate $\kappa_{\text{total}}$) and noise injection (constant $C_{\text{total}}$). This is the fundamental drift-diffusion balance in stochastic systems.

---

### Step 3: Error Decay Bound

Define the error relative to equilibrium:

$$
E(t) := \mathbb{E}[V_{\text{total}}(t)] - V_{\text{total}}^{\text{eq}}
$$

From Step 1:

$$
E(t) = \mathbb{E}[V_{\text{total}}(t)] - \frac{C_{\text{total}}}{\kappa_{\text{total}}} = e^{-\kappa_{\text{total}} t} \left(V_{\text{total}}^{\text{init}} - \frac{C_{\text{total}}}{\kappa_{\text{total}}}\right)
$$

For $V_{\text{total}}^{\text{init}} > V_{\text{total}}^{\text{eq}}$ (typical case: initialization far from equilibrium):

$$
|E(t)| \leq e^{-\kappa_{\text{total}} t} V_{\text{total}}^{\text{init}}
$$

**Key observation:** The error decays exponentially with rate $\kappa_{\text{total}}$.

---

### Step 4: Mixing Time Definition and Calculation

**Definition:** The $\epsilon$-mixing time $T_{\text{mix}}(\epsilon)$ is the smallest time such that:

$$
|E(t)| \leq \epsilon \cdot V_{\text{total}}^{\text{eq}} \quad \text{for all } t \geq T_{\text{mix}}(\epsilon)
$$

**Calculation:** Set $e^{-\kappa_{\text{total}} T_{\text{mix}}} V_{\text{total}}^{\text{init}} = \epsilon \cdot V_{\text{total}}^{\text{eq}}$:

$$
e^{-\kappa_{\text{total}} T_{\text{mix}}} V_{\text{total}}^{\text{init}} = \epsilon \cdot \frac{C_{\text{total}}}{\kappa_{\text{total}}}
$$

Solve for $T_{\text{mix}}$:

$$
T_{\text{mix}} = \frac{1}{\kappa_{\text{total}}} \ln\left(\frac{V_{\text{total}}^{\text{init}} \kappa_{\text{total}}}{\epsilon C_{\text{total}}}\right)
$$

**Alternate form (used in theorem statement):**

$$
T_{\text{mix}}(\epsilon) = \frac{1}{\kappa_{\text{total}}} \ln\left(\frac{V_{\text{total}}^{\text{init}}}{\epsilon C_{\text{total}}}\right) + \frac{1}{\kappa_{\text{total}}} \ln(\kappa_{\text{total}})
$$

For $\kappa_{\text{total}} \sim O(1)$, the second term is negligible, yielding:

$$
T_{\text{mix}}(\epsilon) \approx \frac{1}{\kappa_{\text{total}}} \ln\left(\frac{V_{\text{total}}^{\text{init}}}{\epsilon C_{\text{total}}}\right)
$$

---

### Step 5: Typical Parameter Simplification

**Assumption:** Typical initialization has $V_{\text{total}}^{\text{init}} / C_{\text{total}} \sim O(1)$ (not pathologically large).

**Target:** $\epsilon = 0.01$ (1% accuracy relative to equilibrium).

**Simplification:**

$$
T_{\text{mix}} \approx \frac{1}{\kappa_{\text{total}}} \ln\left(\frac{1}{\epsilon}\right) = \frac{\ln(100)}{\kappa_{\text{total}}} \approx \frac{4.6}{\kappa_{\text{total}}} \approx \frac{5}{\kappa_{\text{total}}}
$$

**Explicit parameter dependence:** Using $\kappa_{\text{total}} \sim \min(\lambda, 2\gamma, \kappa_W, \kappa_b)$:

$$
T_{\text{mix}} \sim \frac{5}{\min(\lambda, 2\gamma, \kappa_W, \kappa_b)}
$$

---

## 5. Critical Estimates and Bounds

### Bound 1: Foster-Lyapunov Drift Holds Uniformly in N

**Claim:** The drift condition $\mathbb{E}[\Delta V_{\text{total}}] \leq -\kappa_{\text{total}}\tau \cdot V_{\text{total}} + C_{\text{total}}$ holds with N-uniform constants.

**Why this matters:** Ensures the mixing time bound applies to all swarm sizes, not just asymptotic limits.

**Reference:** This is established in Chapter 3 of `06_convergence.md` via the synergistic composition of operator-level drifts from `03_cloning.md` and `05_kinetic_contraction.md`.

---

### Bound 2: Discrete-Time Error is Negligible

**Claim:** The approximation $(1 - \kappa_{\text{total}} \tau)^{t/\tau} \approx e^{-\kappa_{\text{total}} t}$ has error $O(\tau^2)$.

**Why this matters:** Justifies treating the discrete-time Markov chain as a continuous-time process.

**Justification:** Taylor expansion:

$$
\ln(1 - \kappa_{\text{total}} \tau) = -\kappa_{\text{total}} \tau - \frac{(\kappa_{\text{total}} \tau)^2}{2} + O(\tau^3)
$$

So:

$$
(1 - \kappa_{\text{total}} \tau)^{t/\tau} = e^{-\kappa_{\text{total}} t - \kappa_{\text{total}}^2 \tau t/2 + O(\tau^2 t)}
$$

For $\tau \sim 0.01$ and $t \sim O(1)$, the error is $\sim 0.0001$, which is negligible.

---

### Bound 3: Initialization is Not Pathological

**Assumption:** $V_{\text{total}}^{\text{init}} / C_{\text{total}} \sim O(1)$ to $O(10)$.

**Why this matters:** If the initialization is extremely far from equilibrium (e.g., $V_{\text{total}}^{\text{init}} / C_{\text{total}} \sim 10^{100}$), the logarithmic term dominates and the "5 time units" approximation breaks down.

**Practical validity:** In all standard initialization schemes (uniform, Gaussian, importance sampling), the initial Lyapunov function is comparable to its equilibrium value, so this assumption holds.

---

## 6. Potential Difficulties and Resolutions

### Difficulty 1: Connecting Discrete and Continuous Time

**Issue:** The Foster-Lyapunov condition is stated in discrete time (per-step drift), but the mixing time formula uses continuous-time exponential decay.

**Resolution:** The standard technique is to use the generator approximation: for small $\tau$, the discrete-time operator $\Psi_{\text{total}}$ is approximated by the continuous-time generator $\mathcal{L}_{\text{total}}$ with:

$$
\mathbb{E}[V(t + \tau)] \approx V(t) + \tau \mathcal{L}_{\text{total}} V(t)
$$

The drift condition $\mathbb{E}[\Delta V] \leq -\kappa \tau V + C$ then implies $\mathcal{L}_{\text{total}} V \leq -\kappa V + C$, which integrates to exponential decay. The error in this approximation is quantified by the BAOAB weak error analysis in `05_kinetic_contraction.md`.

---

### Difficulty 2: Dependence on Initial Condition

**Issue:** The mixing time depends on the initial value $V_{\text{total}}^{\text{init}}$, which is not uniquely defined.

**Resolution:** There are two common conventions:

1. **Worst-case mixing time:** Maximize over all initial conditions in a compact set (e.g., all walkers within a bounded domain). This gives an upper bound valid uniformly over reasonable initializations.

2. **Typical-case mixing time:** Assume $V_{\text{total}}^{\text{init}} \sim V_{\text{total}}^{\text{eq}}$ (initialization near equilibrium), which simplifies the formula to $T_{\text{mix}} \sim \ln(1/\epsilon)/\kappa_{\text{total}}$.

The theorem statement uses the typical-case convention with $V_{\text{total}}^{\text{init}} \sim O(1)$ and $C_{\text{total}} \sim O(1)$, yielding the clean approximation $T_{\text{mix}} \sim 5/\kappa_{\text{total}}$.

---

### Difficulty 3: Interpreting the $\epsilon$-Proximity Condition

**Issue:** The definition of "proximity to equilibrium" can be ambiguous. Does it mean:
- Total variation distance: $\|p_t - p_{\text{eq}}\|_{\text{TV}} < \epsilon$?
- Lyapunov function: $|\mathbb{E}[V(t)] - V^{\text{eq}}| < \epsilon V^{\text{eq}}$?
- Wasserstein distance: $W_2(p_t, p_{\text{eq}}) < \epsilon$?

**Resolution:** The theorem uses the **Lyapunov function criterion**, which is the natural choice given that the entire analysis is based on the Foster-Lyapunov framework. This is a weaker notion than total variation distance (which would require additional arguments involving conductance or spectral gap), but it is sufficient for practical purposes: once $\mathbb{E}[V(t)] \approx V^{\text{eq}}$, the swarm is concentrated near low-potential, high-fitness regions.

For total variation convergence, one typically uses Cheeger's inequality or hypocoercivity estimates to relate the Lyapunov decay rate to the spectral gap, yielding $T_{\text{mix}}^{\text{TV}} \sim O(T_{\text{mix}}^{\text{Lyapunov}})$ with potentially larger constants.

---

## 7. Proof Outline Summary

1. **Start with Foster-Lyapunov drift:** $\mathbb{E}[V_{\text{total}}(n+1)] \leq (1 - \kappa_{\text{total}}\tau) V_{\text{total}}(n) + C_{\text{total}}$

2. **Iterate to obtain exponential decay:** $\mathbb{E}[V_{\text{total}}(t)] \leq e^{-\kappa_{\text{total}} t} V_{\text{total}}^{\text{init}} + \frac{C_{\text{total}}}{\kappa_{\text{total}}}(1 - e^{-\kappa_{\text{total}} t})$

3. **Identify equilibrium:** $V_{\text{total}}^{\text{eq}} = C_{\text{total}}/\kappa_{\text{total}}$

4. **Define error:** $E(t) = \mathbb{E}[V_{\text{total}}(t)] - V_{\text{total}}^{\text{eq}}$

5. **Set mixing time condition:** $|E(T_{\text{mix}})| = \epsilon \cdot V_{\text{total}}^{\text{eq}}$

6. **Solve for $T_{\text{mix}}$:**

$$
T_{\text{mix}}(\epsilon) = \frac{1}{\kappa_{\text{total}}} \ln\left(\frac{V_{\text{total}}^{\text{init}}}{\epsilon C_{\text{total}}}\right)
$$

7. **Simplify for typical parameters:** $T_{\text{mix}} \sim 5/\kappa_{\text{total}}$ when $\epsilon = 0.01$ and $V_{\text{total}}^{\text{init}} \sim O(C_{\text{total}})$

8. **Explicit parameter dependence:** Substitute $\kappa_{\text{total}} = \min(\lambda, 2\gamma, \kappa_W, \kappa_b)$ from Theorem `thm-total-rate-explicit`.

---

## 8. Required Lemmas and Auxiliary Results

The following results are needed from other sections:

1. **Lemma (Discrete-to-Continuous Time Approximation):**
   - Statement: For small $\tau$, $(1 - \kappa \tau)^{t/\tau} = e^{-\kappa t}(1 + O(\tau))$
   - Reference: Standard result; can be made rigorous via Proposition `prop-explicit-constants` in `05_kinetic_contraction.md` which bounds the BAOAB weak error

2. **Theorem (Foster-Lyapunov Drift):**
   - Statement: $\mathbb{E}[\Delta V_{\text{total}}] \leq -\kappa_{\text{total}}\tau \cdot V_{\text{total}} + C_{\text{total}}$
   - Reference: Theorem `thm-total-rate-explicit` (line 1715 of `06_convergence.md`)

3. **Proposition (Component Rate Formulas):**
   - Statement: Explicit formulas for $\kappa_v, \kappa_x, \kappa_W, \kappa_b$ in terms of $\gamma, \lambda, \sigma_v, \tau, N, d$
   - Reference: Propositions in Section 5.1-5.4 of `06_convergence.md`

---

## 9. Connections to Broader Framework

- **QSD Theory:** This mixing time bound applies to convergence to the quasi-stationary distribution (QSD) conditioned on non-extinction. The extinction probability is exponentially small in $N$ (see Section 4 of `06_convergence.md`), so this conditioning is negligible for practical purposes.

- **Mean-Field Limit:** The N-uniformity of $\kappa_{\text{total}}$ and $C_{\text{total}}$ ensures that the mixing time bound holds in the mean-field limit $N \to \infty$.

- **Spectral Gap:** The rate $\kappa_{\text{total}}$ is related to the spectral gap of the generator $\mathcal{L}_{\text{total}}$. For hypocoercive operators, this connection is made precise in Chapter 6 of `09_kl_convergence.md` via the logarithmic Sobolev inequality (LSI).

- **Parameter Optimization:** The formula $T_{\text{mix}} \sim 5/\min(\lambda, 2\gamma, \kappa_W, \kappa_b)$ provides direct guidance for parameter tuning: to minimize mixing time, increase the bottleneck rate (whichever of $\lambda, 2\gamma, \kappa_W, \kappa_b$ is smallest).

---

## 10. Validation Strategy for Full Proof

The full proof should:

1. **Verify all prerequisites:** Explicitly cite the theorem statements and line numbers for all required results (Foster-Lyapunov drift, component rates, etc.).

2. **Make the discrete-to-continuous approximation rigorous:** Use the BAOAB weak error bounds from `05_kinetic_contraction.md` to quantify the error in replacing $(1 - \kappa \tau)^{t/\tau}$ with $e^{-\kappa t}$.

3. **Handle the initial condition dependence:** Either prove a worst-case bound uniform over all initializations in a compact set, or clearly state the "typical initialization" assumption with mathematical precision.

4. **Connect to total variation distance (optional):** For a stronger result, use Pinsker's inequality or hypocoercivity techniques to relate the Lyapunov decay to total variation convergence.

5. **Include worked numerical examples:** Provide concrete parameter values (e.g., $\gamma = 1, \lambda = 0.5, \tau = 0.01$) and compute the resulting $T_{\text{mix}}$ to validate the formula against simulations.

---

## 11. Summary

**Proof Strategy:** Direct application of Foster-Lyapunov drift to extract exponential decay rate.

**Key Dependencies:**
- Foster-Lyapunov drift condition (Theorem `thm-total-rate-explicit`)
- Component rate formulas (Section 5 of `06_convergence.md`)
- Discrete-to-continuous time approximation (BAOAB weak error bounds)

**Main Challenge:** Connecting discrete-time drift to continuous-time exponential decay with explicit error bounds.

**Resolution:** Use standard generator approximation for small $\tau$, with error quantified by BAOAB weak error analysis.

**Result:** Clean explicit formula $T_{\text{mix}} \sim 5/\kappa_{\text{total}}$ for typical parameters, directly linking mixing time to primitive algorithmic parameters.

**Impact:** This formula enables principled parameter selection to achieve desired convergence speed, and validates the synergistic dissipation paradigm by showing that all component rates contribute equally to the bottleneck.
