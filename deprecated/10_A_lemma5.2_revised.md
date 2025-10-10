## Revised Proof of Lemma 5.2: Entropy-Transport Dissipation Inequality

**This is a complete rewrite following Gemini's guidance to fix the critical errors in the original proof.**

---

:::{prf:lemma} Entropy-Transport Dissipation Inequality (REVISED)
:label: lem-entropy-transport-dissipation-revised

For the cloning operator $\Psi_{\text{clone}}$ with parameters satisfying the Keystone Principle (Theorem 8.1 in [03_cloning.md](03_cloning.md)), there exists $\alpha > 0$ such that:

$$
D_{\text{KL}}(\mu' \| \pi) \le D_{\text{KL}}(\mu \| \pi) - \alpha \cdot W_2^2(\mu, \pi) + C_{\text{clone}}
$$

where $\mu' = (\Psi_{\text{clone}})_* \mu$, $\pi = \pi_{\text{QSD}}$ is the quasi-stationary distribution (assumed log-concave by Axiom {prf:ref}`ax-qsd-log-concave`), and $C_{\text{clone}}$ is a state-independent constant.
:::

:::{prf:proof}

**Proof Strategy:** We decompose the cloning operator into two sequential steps and analyze each separately:

1. **Cloning/resampling step** $T_{\text{clone}}$: Dead walkers are replaced by copies of high-fitness alive walkers
2. **Noise step** $T_{\text{noise}}$: Gaussian noise $\mathcal{N}(0, \delta^2 I)$ is added to cloned positions

Thus $\mu' = T_{\text{noise}} \circ T_{\text{clone}}(\mu)$. Let $\mu_c := T_{\text{clone}}(\mu)$ denote the intermediate distribution after cloning but before noise.

---

### Step 1: Decomposition of KL Divergence Change

The total change in KL divergence is:

$$
\begin{aligned}
D_{\text{KL}}(\mu' \| \pi) - D_{\text{KL}}(\mu \| \pi)
&= [D_{\text{KL}}(T_{\text{noise}} \# \mu_c \| \pi) - D_{\text{KL}}(\mu_c \| \pi)] \\
&\quad + [D_{\text{KL}}(\mu_c \| \pi) - D_{\text{KL}}(\mu \| \pi)]
\end{aligned}
$$

Define:
- **Term 1 (Noise effect):** $\Delta_{\text{noise}} := D_{\text{KL}}(T_{\text{noise}} \# \mu_c \| \pi) - D_{\text{KL}}(\mu_c \| \pi)$
- **Term 2 (Cloning effect):** $\Delta_{\text{clone}} := D_{\text{KL}}(\mu_c \| \pi) - D_{\text{KL}}(\mu \| \pi)$

We will show:
- Term 1 provides **contraction**: $\Delta_{\text{noise}} \leq -\alpha W_2^2(\mu, \pi) + O(\delta^4)$
- Term 2 is **bounded**: $\Delta_{\text{clone}} \leq C_{\text{clone}}$

---

### Step 2: Analysis of Term 1 - Noise Contraction

**Setup:** The noise operator is Gaussian convolution:

$$
(T_{\text{noise}} \# \mu_c)(x) = \int \mu_c(y) \cdot \frac{1}{(2\pi\delta^2)^{d/2}} \exp\left(-\frac{\|x-y\|^2}{2\delta^2}\right) dy = (\mu_c * G_\delta)(x)
$$

where $G_\delta = \mathcal{N}(0, \delta^2 I)$.

**Key tool:** For the heat flow $\partial_t \rho_t = \frac{1}{2}\Delta \rho_t$, the **de Bruijn identity** (Cover & Thomas, 2006) states:

$$
\frac{d}{dt} D_{\text{KL}}(\rho_t \| \pi) = -I(\rho_t \| \pi)
$$

where $I(\rho \| \pi) := \int \rho(x) \left\|\nabla \log \frac{\rho(x)}{\pi(x)}\right\|^2 dx$ is the relative Fisher information.

**Discrete-time version:** Adding Gaussian noise with variance $2t$ (i.e., one heat flow step of duration $t$) gives:

$$
D_{\text{KL}}(\mu * G_{\sqrt{2t}} \| \pi) = D_{\text{KL}}(\mu \| \pi) - \int_0^t I(\rho_s \| \pi) ds
$$

where $\rho_s$ is the heat flow from $\mu$ at time $s$.

**Application:** With $\delta^2 = 2t$, we have $t = \delta^2/2$:

$$
\Delta_{\text{noise}} = D_{\text{KL}}(\mu_c * G_\delta \| \pi) - D_{\text{KL}}(\mu_c \| \pi) = -\int_0^{\delta^2/2} I(\rho_s \| \pi) ds
$$

**Bounding the integral:** We need a lower bound on the Fisher information. This is where the log-concavity of $\pi$ becomes crucial.

**Talagrand's inequality for log-concave measures** (Villani, 2009, Theorem 22.17): For $\pi$ satisfying a log-Sobolev inequality with constant $C_{\text{LSI}}$, we have:

$$
W_2^2(\rho \| \pi) \leq \frac{2}{C_{\text{LSI}}} D_{\text{KL}}(\rho \| \pi)
$$

This is the **reverse** direction. We need the **HWI inequality** (Otto-Villani, 2000):

$$
H(\rho \| \pi) \leq W_2(\rho, \pi) \sqrt{I(\rho \| \pi)}
$$

where $H = D_{\text{KL}}$. Squaring both sides:

$$
H^2(\rho \| \pi) \leq W_2^2(\rho, \pi) \cdot I(\rho \| \pi)
$$

Rearranging:

$$
I(\rho \| \pi) \geq \frac{H^2(\rho \| \pi)}{W_2^2(\rho, \pi)}
$$

**Issue:** This doesn't directly give us $I \geq \alpha W_2^2$ for some constant $\alpha$.

**Alternative approach - LSI directly:** If $\pi$ satisfies a log-Sobolev inequality with constant $C_{\text{LSI}}$:

$$
D_{\text{KL}}(\rho \| \pi) \leq \frac{1}{2C_{\text{LSI}}} I(\rho \| \pi)
$$

Then:

$$
I(\rho \| \pi) \geq 2C_{\text{LSI}} D_{\text{KL}}(\rho \| \pi)
$$

**Connecting to Wasserstein:** For log-concave $\pi$ with strong convexity parameter $\kappa_{\text{conf}}$, the **Talagrand inequality** (Bobkov-Götze, 1999) gives:

$$
W_2^2(\rho, \pi) \leq \frac{2}{\kappa_{\text{conf}}} D_{\text{KL}}(\rho \| \pi)
$$

Inverting (assuming $D_{\text{KL}} \geq c W_2^2$ for well-behaved measures):

$$
D_{\text{KL}}(\rho \| \pi) \geq \frac{\kappa_{\text{conf}}}{2} W_2^2(\rho, \pi)
$$

Combining with the LSI:

$$
I(\rho \| \pi) \geq 2C_{\text{LSI}} D_{\text{KL}}(\rho \| \pi) \geq 2C_{\text{LSI}} \cdot \frac{\kappa_{\text{conf}}}{2} W_2^2(\rho, \pi) = C_{\text{LSI}} \kappa_{\text{conf}} W_2^2(\rho, \pi)
$$

**Applying to the integral:** Assuming $I(\rho_s \| \pi) \geq C_{\text{LSI}} \kappa_{\text{conf}} W_2^2(\rho_s, \pi)$ for all $s \in [0, \delta^2/2]$, and using that heat flow contracts Wasserstein distance:

$$
W_2^2(\rho_s, \pi) \geq W_2^2(\mu_c, \pi) \quad \text{(heat flow is contractive)}
$$

Therefore:

$$
\Delta_{\text{noise}} = -\int_0^{\delta^2/2} I(\rho_s \| \pi) ds \leq -\int_0^{\delta^2/2} C_{\text{LSI}} \kappa_{\text{conf}} W_2^2(\rho_s, \pi) ds
$$

**Issue:** This integral depends on the entire trajectory $\rho_s$, not just the initial Wasserstein distance $W_2^2(\mu, \pi)$.

**Simplified bound (first-order approximation):** For small $\delta^2$, we can approximate:

$$
\int_0^{\delta^2/2} I(\rho_s \| \pi) ds \approx \frac{\delta^2}{2} I(\mu_c \| \pi) + O(\delta^4)
$$

Using $I(\mu_c \| \pi) \geq C_{\text{LSI}} \kappa_{\text{conf}} W_2^2(\mu_c, \pi)$:

$$
\Delta_{\text{noise}} \approx -\frac{\delta^2}{2} C_{\text{LSI}} \kappa_{\text{conf}} W_2^2(\mu_c, \pi) + O(\delta^4)
$$

**Relating $W_2(\mu_c, \pi)$ back to $W_2(\mu, \pi)$:** From the Wasserstein contraction property of cloning (the result being proven by the other agent), we expect:

$$
W_2^2(\mu_c, \pi) \approx W_2^2(\mu, \pi) \quad \text{(or slightly contracted)}
$$

Thus:

$$
\boxed{\Delta_{\text{noise}} \leq -\alpha_{\text{noise}} W_2^2(\mu, \pi) + O(\delta^4)}
$$

where $\alpha_{\text{noise}} = \frac{\delta^2}{2} C_{\text{LSI}} \kappa_{\text{conf}}$.

---

### Step 3: Analysis of Term 2 - Cloning Error Bound

**Setup:** The cloning operator $T_{\text{clone}}$ performs resampling:
- Dead walkers (status $s_i = 0$) are replaced by copies of alive walkers
- Selection probabilities proportional to fitness: $p_i \propto \exp(V_{\text{fit},i})$

**Challenge:** $T_{\text{clone}}$ is not a simple transport map - it's a stochastic resampling operator. It can increase KL divergence.

**Strategy:** Bound the worst-case increase in KL divergence due to resampling.

**Key observation:** The cloning operator preserves the total mass and operates within a bounded domain $\mathcal{X}_{\text{valid}}$ (by the Safe Harbor axiom).

**Informal argument:** The resampling step can be viewed as:
1. Remove mass from low-fitness regions (dead walkers)
2. Add mass to high-fitness regions (cloned walkers)

The KL divergence change depends on how these densities compare to $\pi$.

**Formal bound (to be proven):** For the cloning operator with bounded cloning probabilities $p_i \in [0, 1]$ and bounded fitness range:

$$
D_{\text{KL}}(\mu_c \| \pi) \leq D_{\text{KL}}(\mu \| \pi) + C_{\text{clone}}
$$

where $C_{\text{clone}}$ depends on:
- The death rate: $\mathbb{E}[\text{# dead walkers}] / N$
- The fitness variance: $\text{Var}(V_{\text{fit}})$
- The domain size: $\text{diam}(\mathcal{X}_{\text{valid}})$

**Rigorous derivation needed:** This bound requires careful analysis of how resampling affects the Radon-Nikodym derivative $d\mu_c / d\pi$.

**STATUS:** This step requires further development. The bound is plausible but needs rigorous proof.

---

### Step 4: Combining the Bounds

Assuming both Term 1 and Term 2 bounds hold:

$$
\begin{aligned}
D_{\text{KL}}(\mu' \| \pi) - D_{\text{KL}}(\mu \| \pi)
&= \Delta_{\text{noise}} + \Delta_{\text{clone}} \\
&\leq -\alpha_{\text{noise}} W_2^2(\mu, \pi) + O(\delta^4) + C_{\text{clone}} \\
&\leq -\alpha W_2^2(\mu, \pi) + C'_{\text{clone}}
\end{aligned}
$$

where $\alpha = \alpha_{\text{noise}} = \frac{\delta^2}{2} C_{\text{LSI}} \kappa_{\text{conf}}$ and $C'_{\text{clone}} = C_{\text{clone}} + O(\delta^4)$.

Rearranging:

$$
\boxed{D_{\text{KL}}(\mu' \| \pi) \leq D_{\text{KL}}(\mu \| \pi) - \alpha W_2^2(\mu, \pi) + C_{\text{clone}}}
$$

**Q.E.D.** (modulo completing the rigorous bound for Term 2)
:::

---

## Notes and Open Issues

**Completed:**
- ✅ Correct decomposition into $T_{\text{noise}} \circ T_{\text{clone}}$
- ✅ Term 1 analysis using heat flow and de Bruijn identity
- ✅ Connection to LSI and Talagrand inequality
- ✅ First-order approximation for small $\delta^2$

**Requires Completion:**
- ❌ **Term 2 rigorous bound:** Need to prove $\Delta_{\text{clone}} \leq C_{\text{clone}}$ rigorously
- ❌ **Integral approximation:** The $\int_0^{\delta^2/2} I(\rho_s \| \pi) ds \approx \frac{\delta^2}{2} I(\mu_c \| \pi)$ step needs more justification
- ❌ **Wasserstein contraction dependency:** Currently assumes $W_2(\mu_c, \pi) \approx W_2(\mu, \pi)$ - needs the W₂ contraction result from the other agent

**Next Steps:**
1. Submit this draft to Gemini for verification of the overall structure
2. Get guidance on proving the Term 2 bound rigorously
3. Refine the integral approximation if needed
