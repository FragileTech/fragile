# Fisher-Rao Convergence: Feasibility Roadmap and Strategic Analysis

 $p_t(g) = \sum_R d_R e^{-t C_2(R)} \chi_R(g)$

**Purpose:** This document analyzes the feasibility of proving convergence to the quasi-stationary distribution (QSD) in the **Fisher-Rao metric** (also called the Fisher information metric), building on the existing KL-divergence convergence results in [10_kl_convergence.md](10_kl_convergence/10_kl_convergence.md). We evaluate different proof approaches, assess their technical requirements, difficulty levels, and potential payoffs.

**Status:** Strategic analysis and roadmap

**Relationship to Existing Results:**
- [04_convergence.md](04_convergence.md): Total variation convergence via Foster-Lyapunov
- [10_kl_convergence.md](10_kl_convergence/10_kl_convergence.md): KL-divergence convergence via LSI
- **This document**: Fisher-Rao metric convergence (gradient flow perspective)

---

## 0. Executive Summary

### 0.1. What is the Fisher-Rao Metric?

The Fisher-Rao metric is a Riemannian metric on the space of probability distributions $\mathcal{P}(\mathcal{X})$ that measures the "infinitesimal distinguishability" of nearby distributions. It provides a geometric structure complementary to the information-theoretic KL divergence.

**Key Properties:**
1. **Riemannian structure:** Endows $\mathcal{P}(\mathcal{X})$ with a smooth manifold structure
2. **Second-order approximation to KL:** For nearby distributions $p_\theta$ and $p_{\theta+d\theta}$:

$$
D_{\text{KL}}(p_{\theta+d\theta} \| p_\theta) \approx \frac{1}{2} \sum_{i,j} g_{ij}(\theta) \, d\theta_i d\theta_j
$$

where $g_{ij}$ is the Fisher information matrix

3. **Intrinsic geometry:** Invariant under sufficient statistics (Chentsov's theorem)
4. **Connects to gradient flows:** PDEs can be viewed as steepest descent on $(\mathcal{P}(\mathcal{X}), g_{\text{FR}})$

### 0.2. Why Pursue Fisher-Rao Convergence?

**Scientific Payoffs:**
- **Geometric interpretation:** Understand the Euclidean Gas as following geodesics in information space
- **Stronger structure:** Fisher-Rao convergence implies concentration inequalities and tail bounds
- **Curvature insights:** Positive curvature $\Rightarrow$ uniqueness and stability of QSD
- **Connection to physics:** Natural framework for thermodynamic limits and phase transitions

**Technical Payoffs:**
- **Unified framework:** KL, Wasserstein, and Fisher-Rao metrics in one geometric picture
- **Operator splitting:** Better understanding of $\Psi_{\text{kin}} \circ \Psi_{\text{clone}}$ as composition of gradient flows
- **Adaptive Gas extensions:** Essential for analyzing Hessian-based anisotropic diffusion

**Publication Impact:**
- Novel contribution to stochastic optimization literature
- Bridges particle methods and information geometry
- Opens path to Riemannian Euclidean Gas variants

### 0.3. Strategic Assessment (REVISED AFTER CRITICAL REVIEW)

We identify **four distinct approaches** with varying feasibility. The table below reflects a **critical re-evaluation** after discovering fundamental issues with the initial assessment:

| Approach | Difficulty | Feasibility | Payoff | Time Est. | Status |
|:---------|:-----------|:------------|:-------|:----------|:-------|
| **A1:** Direct LSI $\Rightarrow$ Fisher-Rao | ★★★☆☆ | MEDIUM | Medium | 1-2 months | ⚠️ Requires reverse LSI (non-standard) |
| **A2:** Bakry-Émery Curvature | ★★★★☆ | MEDIUM-LOW | High | 2-3 months | ⚠️ Core difficulty: non-convex potential |
| **A3:** Otto Calculus (Gradient Flow) | ★★★★★ | LOW | Very High | 4-6 months | ❌ Research frontier, PhD-level |
| **A4:** Hybrid (KL + Wasserstein) | ★★★★☆ | MEDIUM-LOW | High | 2-3 months | ⚠️ **Inherits A2's curvature requirement** |

**Critical Findings:**
1. **A1 is NOT straightforward:** Requires proving a "reverse LSI" (upper bound on Fisher information), which is non-standard
2. **A4 does NOT bypass A2:** The HWI inequality requires the same curvature condition as A2, creating hidden circularity
3. **All approaches face the curvature bottleneck:** Fisher-Rao convergence fundamentally requires understanding the geometry of $\pi_{\text{QSD}}$

**Revised Recommended Strategy:**
- **Priority 1:** Investigate whether $\pi_{\text{QSD}}$ satisfies any curvature bounds (even weak ones like $\text{CD}(0, \infty)$)
- **Priority 2:** Focus on **Wasserstein convergence only** (Alt-A4a) as a standalone, publishable result
- **Priority 3:** Reserve full Fisher-Rao convergence for future work once curvature properties are understood

---

## 1. Mathematical Foundations

### 1.1. The Fisher-Rao Metric: Formal Definition

:::{prf:definition} Fisher Information Matrix
:label: def-fisher-information-matrix

Let $\mathcal{M} = \{p_\theta : \theta \in \Theta \subset \mathbb{R}^k\}$ be a parametric family of probability densities on $\mathcal{X}$. The **Fisher information matrix** $g(\theta) \in \mathbb{R}^{k \times k}$ has components:

$$
g_{ij}(\theta) := \mathbb{E}_{x \sim p_\theta} \left[ \frac{\partial \log p_\theta(x)}{\partial \theta_i} \cdot \frac{\partial \log p_\theta(x)}{\partial \theta_j} \right]
$$

Equivalently, using the score function $s_i(\theta; x) := \partial_{\theta_i} \log p_\theta(x)$:

$$
g_{ij}(\theta) = \int_{\mathcal{X}} s_i(\theta; x) \cdot s_j(\theta; x) \, p_\theta(x) \, dx
$$

The Fisher information is the **covariance matrix of the score**.
:::

:::{prf:definition} Fisher-Rao Distance
:label: def-fisher-rao-distance

The **Fisher-Rao distance** between two distributions $p, q \in \mathcal{M}$ is the geodesic distance in the Riemannian manifold $(\mathcal{M}, g)$:

$$
d_{\text{FR}}(p, q) := \inf_{\gamma} \int_0^1 \sqrt{\sum_{i,j} g_{ij}(\gamma(t)) \dot{\gamma}_i(t) \dot{\gamma}_j(t)} \, dt
$$

where the infimum is over all smooth curves $\gamma: [0,1] \to \Theta$ with $\gamma(0) = \theta_p$, $\gamma(1) = \theta_q$.

For **infinitesimally close** distributions $p_\theta$ and $p_{\theta + d\theta}$:

$$
d_{\text{FR}}^2(p_\theta, p_{\theta+d\theta}) = \sum_{i,j} g_{ij}(\theta) \, d\theta_i d\theta_j
$$

:::

### 1.2. Relationship to KL Divergence

:::{prf:theorem} KL Divergence as Second-Order Approximation
:label: thm-kl-fisher-rao-relation

For a smooth parametric family $\{p_\theta\}$, the KL divergence admits the Taylor expansion:

$$
D_{\text{KL}}(p_{\theta+d\theta} \| p_\theta) = \frac{1}{2} \sum_{i,j} g_{ij}(\theta) \, d\theta_i d\theta_j + O(\|d\theta\|^3)
$$

Thus:

$$
D_{\text{KL}}(p_{\theta+d\theta} \| p_\theta) = \frac{1}{2} d_{\text{FR}}^2(p_\theta, p_{\theta+d\theta}) + O(\|d\theta\|^3)
$$

:::

:::{prf:proof}
The KL divergence is:

$$
D_{\text{KL}}(p_{\theta+d\theta} \| p_\theta) = \int p_{\theta+d\theta}(x) \log \frac{p_{\theta+d\theta}(x)}{p_\theta(x)} \, dx
$$

We expand the logarithm:

$$
\log p_{\theta+d\theta}(x) = \log p_\theta(x) + s_i d\theta_i + \frac{1}{2} H_{ij} d\theta_i d\theta_j + O(\|d\theta\|^3)
$$

where $s_i = \partial_i \log p_\theta(x)$ and $H_{ij} = \partial_i \partial_j \log p_\theta(x)$.

The log-ratio is:

$$
\log \frac{p_{\theta+d\theta}(x)}{p_\theta(x)} = s_i d\theta_i + \frac{1}{2} H_{ij} d\theta_i d\theta_j + O(\|d\theta\|^3)
$$

We also expand the density:

$$
p_{\theta+d\theta}(x) = p_\theta(x) (1 + s_k d\theta_k + O(\|d\theta\|^2))
$$

Substituting into the integral:

$$
D_{\text{KL}} = \int p_\theta(1 + s_k d\theta_k) \left[ s_i d\theta_i + \frac{1}{2} H_{ij} d\theta_i d\theta_j \right] dx + O(\|d\theta\|^3)
$$

The terms are:
1. $\int p_\theta s_i d\theta_i \, dx = \mathbb{E}[s_i]d\theta_i = 0$ (score has zero expectation)
2. $\int p_\theta (s_k d\theta_k)(s_i d\theta_i) \, dx = \mathbb{E}[s_i s_k] d\theta_i d\theta_k = g_{ik} d\theta_i d\theta_k$
3. $\int p_\theta \left(\frac{1}{2} H_{ij} d\theta_i d\theta_j\right) dx = \frac{1}{2} \mathbb{E}[H_{ij}] d\theta_i d\theta_j$

Using the identity $\mathbb{E}_{p_\theta}[H_{ij}(\theta; x)] = -g_{ij}(\theta)$ (derived from differentiating $\int p_\theta(x) dx = 1$ twice), the sum of second-order terms is:

$$
g_{ij} d\theta_i d\theta_j - \frac{1}{2}g_{ij} d\theta_i d\theta_j = \frac{1}{2} g_{ij} d\theta_i d\theta_j
$$

Therefore:

$$
D_{\text{KL}}(p_{\theta+d\theta} \| p_\theta) = \frac{1}{2} \sum_{i,j} g_{ij}(\theta) \, d\theta_i d\theta_j + O(\|d\theta\|^3)
$$

:::

**Implication:** The Fisher-Rao metric captures the **local geometry** of the KL divergence. Proving convergence in KL does not immediately imply Fisher-Rao convergence (which requires understanding the full geodesic structure), but it provides strong evidence.

### 1.3. Fisher Information as a Functional

For the **non-parametric setting** (where distributions are not restricted to a finite-dimensional family), the Fisher information is defined as a functional:

:::{prf:definition} Fisher Information Functional
:label: def-fisher-info-functional

For probability measures $\mu, \nu$ on $\mathcal{X}$ with $\mu \ll \nu$ and density $\rho = d\mu/d\nu$, the **Fisher information** of $\mu$ relative to $\nu$ is:

$$
I(\mu \| \nu) := \int_{\mathcal{X}} \left| \nabla \log \frac{d\mu}{d\nu} \right|^2 \frac{d\mu}{d\nu} \, d\nu = \int_{\mathcal{X}} \frac{|\nabla \rho|^2}{\rho} \, d\nu
$$

Equivalently (using the Cauchy-Schwarz identity):

$$
I(\mu \| \nu) = 4 \int_{\mathcal{X}} \left| \nabla \sqrt{\rho} \right|^2 d\nu
$$

:::

**Connection to LSI:** The logarithmic Sobolev inequality (LSI) bounds relative entropy by Fisher information:

$$
D_{\text{KL}}(\mu \| \pi) \le \frac{C_{\text{LSI}}}{2} I(\mu \| \pi)
$$

Our existing proof in [10_kl_convergence.md](10_kl_convergence/10_kl_convergence.md) establishes this for the Euclidean Gas.

---

## 2. Approach A1: Direct LSI Implications

### 2.1. Strategy Overview

**Main Idea:** Leverage the existing LSI from [10_kl_convergence.md](10_kl_convergence/10_kl_convergence.md) to derive Fisher-Rao metric properties without proving a full gradient flow structure.

**Key Observation:** If $\Psi_{\text{total}}$ satisfies an LSI, then:
1. The Fisher information $I(\mu_t \| \pi_{\text{QSD}})$ is **lower-bounded** by entropy
2. The entropy decays exponentially
3. Therefore, Fisher information must decay at least as fast (but we need an *upper* bound to control distance)

**Critical Gap:** The LSI provides $I(\mu \| \pi) \ge \frac{2}{C_{\text{LSI}}} D_{\text{KL}}(\mu \| \pi)$, which is the **wrong direction** for bounding Fisher-Rao distance. We need a "reverse LSI" or Poincaré-type inequality.

**Feasibility:** ★★★☆☆ **MEDIUM** — Requires proving a non-standard functional inequality (reverse LSI)

**Difficulty:** ★★★☆☆ **MEDIUM-HIGH** — More challenging than initially assessed

### 2.2. Technical Development

:::{prf:remark} LSI Direction and Fisher Information
:label: rem-lsi-direction-issue

The standard logarithmic Sobolev inequality states:

$$
D_{\text{KL}}(\mu \| \pi) \le \frac{C_{\text{LSI}}}{2} I(\mu \| \pi)
$$

Rearranging gives a **lower bound** on Fisher information:

$$
I(\mu \| \pi) \ge \frac{2}{C_{\text{LSI}}} D_{\text{KL}}(\mu \| \pi)
$$

This tells us that if the KL divergence is large, the Fisher information must also be large. However, for proving convergence in Fisher-Rao metric, we need an **upper bound** on $I(\mu_t \| \pi)$—we need to show that Fisher information *decreases* as the distribution approaches $\pi$.

**What we have:** $I(\mu_t \| \pi) \ge \frac{2}{C_{\text{LSI}}} D_{\text{KL}}(\mu_t \| \pi) \ge \frac{2}{C_{\text{LSI}}} e^{-t/C_{\text{LSI}}} D_{\text{KL}}(\mu_0 \| \pi)$

This shows that Fisher information decays exponentially *from below*, but does not prevent it from being arbitrarily large above this bound.
:::

:::{prf:conjecture} Reverse LSI for Euclidean Gas
:label: conj-reverse-lsi

For the Euclidean Gas operator $\Psi_{\text{total}}$ with quasi-stationary distribution $\pi_{\text{QSD}}$, there exists a constant $C_{\text{reverse}} > 0$ such that for all probability measures $\mu$ with $D_{\text{KL}}(\mu \| \pi_{\text{QSD}}) < \infty$:

$$
I(\mu \| \pi_{\text{QSD}}) \le C_{\text{reverse}} \cdot D_{\text{KL}}(\mu \| \pi_{\text{QSD}})
$$

**If true, this would imply:** Combined with exponential KL decay, Fisher information decays exponentially from above, enabling bounds on Fisher-Rao distance.

**Proof strategy:** Such inequalities are known for:
- Gaussian measures (where $C_{\text{reverse}}$ depends on covariance)
- Log-concave measures under strong convexity (Bakry-Émery theory)
- Measures satisfying Talagrand's $T_2$ transportation inequality

For the Euclidean Gas, one would need to verify that $\pi_{\text{QSD}}$ satisfies one of these conditions.
:::

:::{prf:remark} Fisher Information vs. Fisher-Rao Distance
:label: rem-fisher-info-vs-distance

Even if we establish that the **Fisher information functional** $I(\mu_t \| \pi)$ decays exponentially (via Conjecture {prf:ref}`conj-reverse-lsi`), this is **not the same** as showing that the **Fisher-Rao distance** $d_{\text{FR}}(\mu_t, \pi)$ decays.

**The gap:** Fisher information is a *second-order* quantity (gradient squared), while Fisher-Rao distance involves integration along geodesics. The relationship:

$$
I(\mu \| \pi) = \left\| \nabla_{\text{FR}} \log \frac{d\mu}{d\pi} \right\|_{L^2(\mu)}^2
$$

suggests that $I(\mu \| \pi)$ controls the "velocity" of convergence in Fisher-Rao geometry, but bounding $d_{\text{FR}}(\mu_t, \pi)$ requires integrating this velocity along the flow.
:::

### 2.3. Potential Strengthening: De Bruijn's Identity

To bridge the gap between Fisher information decay and Fisher-Rao distance convergence, we might use **de Bruijn's identity** from information theory.

:::{prf:theorem} De Bruijn's Identity (Standard Form)
:label: thm-de-bruijn-identity

Let $X$ be a random variable with density $p$ and $Z \sim \mathcal{N}(0, \sigma^2 I_d)$ be independent Gaussian noise. Define:

$$
h(\sigma) := H(X + \sigma Z) = -\int p_\sigma(x) \log p_\sigma(x) \, dx
$$

where $p_\sigma = p * \phi_\sigma$ is the Gaussian-convolved density. Then:

$$
\frac{d}{d\sigma} h(\sigma) = \frac{1}{2\sigma} I(p_\sigma)
$$

where $I(p_\sigma)$ is the Fisher information of $p_\sigma$.
:::

:::{prf:conjecture} De Bruijn Identity for BAOAB Integrator
:label: conj-de-bruijn-baoab

The BAOAB integrator's noise steps can be approximated by Gaussian convolution in a way that preserves the de Bruijn identity structure. Specifically, for the velocity noise step with parameter $\sigma_v$:

$$
\frac{d}{dt} D_{\text{KL}}(\mu_t \| \pi_{\text{QSD}}) \sim -\alpha I(\mu_t \| \pi_{\text{QSD}}) + O(\text{discretization error})
$$

for some constant $\alpha > 0$ related to the friction coefficient $\gamma$ and noise scale $\sigma_v$.

**Proof requirements:** Show that the discrete noise update in BAOAB is sufficiently close to continuous Gaussian convolution for a variant of de Bruijn's identity to hold with controlled error terms.
:::

### 2.4. Feasibility Assessment for A1

**Required Steps:**
1. ✅ **Already have:** LSI for $\Psi_{\text{total}}$ ({prf:ref}`thm-main-kl-convergence`)
2. ❌ **Major gap:** Need to prove Conjecture {prf:ref}`conj-reverse-lsi` (reverse LSI), which is **non-standard**
3. ⚠️ **Difficult:** Prove Conjecture {prf:ref}`conj-de-bruijn-baoab` for BAOAB integrator
4. ⚠️ **Technical:** Show that Fisher information decay $\Rightarrow$ Fisher-Rao distance decay via integration

**Gaps to Address:**
- **Gap 1 (CRITICAL):** The standard LSI gives the wrong inequality direction—need a "reverse LSI" or Poincaré-type bound
- **Gap 2:** The Euclidean Gas operates on the **swarm state space** $\Sigma_N$, not just $\mathcal{X}$. Need to define Fisher-Rao metric on $\Sigma_N$ (likely product metric)
- **Gap 3:** The kinetic operator has *two* noise sources: position diffusion ($\sigma_x$) and velocity diffusion ($\sigma_v$). Need careful decomposition
- **Gap 4:** Cloning operator is **not** a diffusion—it's a discrete jump. Need to understand its Fisher-Rao geometry

**Time Estimate:** 1-2 months (revised upward due to reverse LSI requirement)

**Payoff:** Medium—establishes Fisher information control but may not yield full Fisher-Rao convergence

**Recommendation:** ⚠️ **Reconsider as first step**—requires proving non-standard functional inequalities; moderate-to-high risk

---

## 3. Approach A2: Bakry-Émery Curvature Condition

### 3.1. Strategy Overview

**Main Idea:** Prove that the quasi-stationary distribution $\pi_{\text{QSD}}$ satisfies a **Bakry-Émery curvature condition** $\text{CD}(\rho, \infty)$, which implies exponential convergence in Fisher-Rao distance via the **HWI inequality**.

**Theoretical Foundation:**
- **Bakry-Émery criterion:** A measure $\pi$ has Ricci curvature $\ge \rho > 0$ if its generator $\mathcal{L}$ satisfies:

$$
\mathcal{L}(\Gamma(f)) \ge \frac{1}{d}(\mathcal{L} f)^2 + \rho \Gamma(f)
$$

where $\Gamma(f) := |\nabla f|^2$ is the carré du champ operator.

- **Consequence:** If $\text{CD}(\rho, \infty)$ holds, then the Fokker-Planck evolution contracts in Wasserstein distance:

$$
W_2(\mu_t, \pi) \le e^{-\rho t} W_2(\mu_0, \pi)
$$

and (by the **HWI inequality**) also in KL divergence with explicit constants.

**Feasibility:** ★★★☆☆ **MEDIUM** — Requires proving curvature bound for hybrid kinetic-cloning dynamics

**Difficulty:** ★★★★☆ **HIGH** — Involves second-order analysis of generators

### 3.2. Technical Requirements

To apply the Bakry-Émery framework to the Euclidean Gas, we must:

#### 3.2.1. Define the Generator on Swarm Space

The Euclidean Gas alternates between two operators:

$$
\Psi_{\text{total}} = \Psi_{\text{kin}}(\tau) \circ \Psi_{\text{clone}}
$$

Each has an associated infinitesimal generator:
- $\mathcal{L}_{\text{kin}}$: Underdamped Langevin (hypoelliptic)
- $\mathcal{L}_{\text{clone}}$: Jump process (non-local)

The **composite generator** does not exist in the usual sense because we're composing *finite-time* maps. However, we can define an **effective generator** via the Trotter-Kato splitting:

$$
\mathcal{L}_{\text{eff}} := \lim_{\tau \to 0} \frac{\Psi_{\text{total}}^{\tau} - I}{\tau}
$$

**Challenge:** This limit may not exist due to the non-commutativity of $\mathcal{L}_{\text{kin}}$ and $\mathcal{L}_{\text{clone}}$.

#### 3.2.2. Verify Bakry-Émery Criterion

For the kinetic operator alone, hypocoercivity theory (Villani 2009) provides a **modified Bakry-Émery criterion**:

:::{prf:theorem} Hypocoercive Curvature (Villani 2009)
:label: thm-villani-hypocoercivity

For the underdamped Langevin dynamics:

$$
dx = v \, dt, \quad dv = -\gamma v \, dt - \nabla U(x) \, dt + \sigma_v \, dW_t
$$

with $U$ satisfying $\text{Hess}(U) \ge \kappa I_d$ (uniformly convex potential), the kinetic generator satisfies a **modified Bakry-Émery condition**:

$$
\mathcal{L}_{\text{kin}}(\Gamma_{\text{hypo}}(f)) \ge \rho_{\text{eff}} \Gamma_{\text{hypo}}(f)
$$

where $\Gamma_{\text{hypo}}$ is the hypocoercive carré du champ and:

$$
\rho_{\text{eff}} = \min\left(\gamma, \frac{\kappa}{2}\right)
$$

:::

**Our setting:** The confining potential $U$ in the Euclidean Gas is **not** uniformly convex—it's designed to confine walkers to $\mathcal{X}_{\text{valid}}$ but may be flat in the interior. This violates the standard Bakry-Émery assumptions.

#### 3.2.3. Handle the Cloning Operator

The cloning operator is a **non-local jump process**, which is fundamentally different from diffusion. Standard Bakry-Émery theory assumes the generator is a second-order differential operator.

**Possible Approaches:**
1. **Conditional curvature:** Prove curvature bounds *conditional on the alive set* $\mathcal{A}_t$
2. **Averaging over jumps:** Show that the jump kernel preserves curvature in expectation
3. **Lower-bound only:** Settle for showing $\text{CD}(0, \infty)$ (non-negative curvature) rather than strictly positive

### 3.3. Feasibility Assessment for A2

**Required Steps:**
1. ⚠️ **Difficult:** Modify Villani's hypocoercivity to handle non-uniformly-convex $U$
2. ⚠️ **Very difficult:** Extend Bakry-Émery criterion to jump processes
3. ⚠️ **Open research:** Prove curvature bound for operator composition $\mathcal{L}_{\text{kin}} \circ \mathcal{L}_{\text{clone}}$

**Gaps to Address:**
- **Gap 1:** The potential $U$ is only *coercive*, not uniformly convex
- **Gap 2:** Cloning introduces discrete jumps—requires non-local curvature theory
- **Gap 3:** The swarm state space $\Sigma_N$ is a product space—need tensorization arguments

**Time Estimate:** 2-3 months

**Payoff:** High—if successful, provides the **strongest** convergence result with explicit geometric constants

**Recommendation:** ⚠️ **Defer to future work**—too many open problems for immediate pursuit

---

## 4. Approach A3: Otto Calculus (Gradient Flow)

### 4.1. Strategy Overview

**Main Idea:** Interpret the Euclidean Gas evolution as a **gradient flow** of the relative entropy functional on the Wasserstein manifold $(\mathcal{P}(\Sigma_N), W_2)$ and use the **JKO scheme** (Jordan-Kinderlehrer-Otto) to prove convergence.

**Theoretical Foundation (Otto 1999, Villani 2009):**

The Fokker-Planck equation:

$$
\partial_t \rho = \nabla \cdot (\rho \nabla U) + \Delta \rho
$$

is the **Wasserstein gradient flow** of the free energy functional:

$$
\mathcal{F}[\rho] := \int \rho \log \rho \, dx + \int \rho U \, dx
$$

meaning that $\rho_t$ evolves along the path of steepest descent of $\mathcal{F}$ in the Wasserstein metric.

**Our setting:** Can we interpret:

$$
\mu_{t+1} = (\Psi_{\text{kin}}(\tau) \circ \Psi_{\text{clone}})_* \mu_t
$$

as a discrete-time gradient flow?

**Feasibility:** ★☆☆☆☆ **LOW** — Requires proving that $\Psi_{\text{clone}}$ is a "proximal step"

**Difficulty:** ★★★★★ **VERY HIGH** — Cutting-edge research in optimal transport

### 4.2. The JKO Scheme

:::{prf:definition} JKO Scheme (Jordan-Kinderlehrer-Otto)
:label: def-jko-scheme

A sequence of probability measures $(\mu_k)_{k \ge 0}$ is a **discrete-time gradient flow** of a functional $\mathcal{F}: \mathcal{P}(\mathcal{X}) \to \mathbb{R}$ with step size $\tau > 0$ if:

$$
\mu_{k+1} = \arg\min_{\nu \in \mathcal{P}(\mathcal{X})} \left\{ \mathcal{F}[\nu] + \frac{1}{2\tau} W_2^2(\nu, \mu_k) \right\}
$$

where $W_2$ is the 2-Wasserstein distance.

**Interpretation:** At each step, move toward the minimum of $\mathcal{F}$ while penalizing large displacements.
:::

**For the Euclidean Gas to fit this framework, we need:**

1. **Functional $\mathcal{F}$:** Define a potential on $\mathcal{P}(\Sigma_N)$ such that $\pi_{\text{QSD}}$ is the unique minimizer
2. **Metric:** Use Wasserstein (or Fisher-Rao) metric on the swarm space
3. **Proximal interpretation:** Show that $\Psi_{\text{total}}$ solves (or approximately solves) the JKO minimization

### 4.3. Challenges for the Euclidean Gas

#### Challenge 1: The Cloning Operator is Not Gradient Flow

The cloning operator:

$$
\Psi_{\text{clone}}: (x_i, v_i, s_i) \mapsto (x_{c(i)}, v_{c(i)}, 1) \quad \text{with probability } p_i
$$

is a **discrete jump** based on fitness potential. This is fundamentally different from:
- **Diffusion** (continuous, local, reversible)
- **Gradient flow** (minimizes a potential)

**Possible resolution:** Interpret cloning as a **non-local** gradient flow in the space of measures, but this requires showing that the cloning kernel is a "descent direction" for some entropy-like functional.

#### Challenge 2: The Kinetic Operator Has Two Metrics

The kinetic operator $\Psi_{\text{kin}}$ evolves $(x, v)$ via Langevin dynamics. In the Otto calculus framework:
- **Wasserstein metric $W_2$:** Measures displacement in position space $\mathcal{X}$
- **Fisher-Rao metric:** Measures "information-theoretic" displacement

For underdamped Langevin, both metrics are involved:
- Position diffusion $\Rightarrow$ Wasserstein geometry
- Velocity diffusion $\Rightarrow$ Fisher-Rao geometry (via kinetic energy)

**Resolution:** Use the **Wasserstein-Fisher-Rao (WFR) metric** (Chizat et al. 2018), which unifies both geometries. However, this is a recent development with limited theory.

#### Challenge 3: The Swarm State Space is High-Dimensional

The space $\Sigma_N = (\mathcal{X} \times \mathbb{R}^d \times \{0,1\})^N$ is:
- **High-dimensional** ($N \cdot (2d + 1)$ dimensions)
- **Product space** (need tensorization of Wasserstein/Fisher-Rao metrics)
- **Includes discrete component** (survival status $s_i \in \{0,1\}$)

Defining a gradient flow structure on such a space is **non-trivial**.

### 4.4. Feasibility Assessment for A3

**Required Steps:**
1. ❌ **Open problem:** Prove cloning operator is a gradient flow (or proximal step) of some functional
2. ❌ **Cutting-edge:** Extend WFR metric to swarm spaces with discrete components
3. ❌ **Research frontier:** Prove convergence of JKO scheme for non-smooth functionals

**Gaps to Address:**
- **Gap 1:** No known framework for gradient flows with discrete jumps + diffusion
- **Gap 2:** WFR metric theory is still under development (as of 2025)
- **Gap 3:** Tensorization of optimal transport over product spaces with $N \to \infty$ is an open problem (mean-field limit)

**Time Estimate:** 4-6 months (potentially PhD thesis–level work)

**Payoff:** Very high—would be a **major theoretical contribution** to stochastic optimization and information geometry

**Recommendation:** ❌ **Defer to future work**—too ambitious for immediate pursuit; suitable for a dedicated research project

---

## 5. Approach A4: Hybrid (KL + Wasserstein)

### 5.1. Strategy Overview

**Main Idea:** Combine KL-divergence convergence (already proven) with Wasserstein convergence to **sandwich** the Fisher-Rao distance using the **HWI inequality**.

**Theoretical Foundation:**

The **HWI inequality** (Otto-Villani 2000) relates three notions of distance between probability measures:

:::{prf:theorem} HWI Inequality
:label: thm-hwi-inequality

Let $\mu, \pi$ be probability measures on $\mathbb{R}^d$ with $\pi$ satisfying the curvature condition $\text{CD}(\rho, \infty)$ with $\rho > 0$. Then:

$$
H(\mu \| \pi) \le W_2(\mu, \pi) \sqrt{I(\mu \| \pi)} - \frac{\rho}{2} W_2^2(\mu, \pi)
$$

where:
- $H(\mu \| \pi) := D_{\text{KL}}(\mu \| \pi)$ is the relative entropy
- $W_2(\mu, \pi)$ is the 2-Wasserstein distance
- $I(\mu \| \pi)$ is the Fisher information

:::

**Strategy:**
1. **Already have:** Exponential decay of $H(\mu_t \| \pi)$ (KL divergence)
2. **Prove additionally:** Exponential decay of $W_2(\mu_t, \pi)$ (Wasserstein distance)
3. **Combine via HWI:** Deduce bounds on $I(\mu_t \| \pi)$ (Fisher information)
4. **Integrate Fisher information:** Obtain Fisher-Rao distance via $d_{\text{FR}}(\mu_t, \pi) \sim \int \sqrt{I(\mu_s \| \pi)} \, ds$

**Feasibility:** ★★★★☆ **MEDIUM-HIGH** — Wasserstein convergence is tractable; integration step requires care

**Difficulty:** ★★★☆☆ **MEDIUM** — Technically involved but within reach

### 5.2. Step 1: Wasserstein Convergence for the Euclidean Gas

**Existing foundation:** The kinetic operator $\Psi_{\text{kin}}$ is a contraction in Wasserstein distance under mild conditions.

:::{prf:theorem} Wasserstein Contraction for Langevin (Eberle 2016)
:label: thm-langevin-wasserstein-contraction

For underdamped Langevin dynamics with potential $U$ satisfying:
1. $\nabla U$ is $L$-Lipschitz
2. $U$ is $\kappa$-convex (i.e., $\text{Hess}(U) \ge \kappa I_d$)

the Langevin flow map $\Psi_{\text{kin}}(\tau)$ satisfies:

$$
W_2(\Psi_{\text{kin}}(\tau)_* \mu, \Psi_{\text{kin}}(\tau)_* \nu) \le e^{-\lambda \tau} W_2(\mu, \nu)
$$

where $\lambda = \min(\gamma, \kappa)$ is the contraction rate.
:::

**Adaptation needed:** Our potential $U$ (the confining potential for the Euclidean Gas) is **coercive but not uniformly convex**. We need a weaker version:

:::{prf:conjecture} Wasserstein Contraction with Coercive Potential
:label: conj-wasserstein-coercive

For the Euclidean Gas kinetic operator with potential $U$ satisfying {prf:ref}`axiom-confining-potential`, there exists a constant $\kappa_W > 0$ such that:

$$
W_2(\Psi_{\text{kin}}(\tau)_* \mu, \pi_{\text{kin}}) \le (1 - \kappa_W \tau) W_2(\mu, \pi_{\text{kin}}) + O(\tau^2)
$$

where $\pi_{\text{kin}}$ is the stationary measure of $\Psi_{\text{kin}}$.

**Intuition:** Even without uniform convexity, the coercivity of $U$ ensures that walkers far from the origin are pulled back, creating an effective Wasserstein contraction *on average*.
:::

**How to prove:** Use **coupling methods** (Eberle 2016, Majka 2020). Construct a coupling $(X_t, Y_t)$ of two trajectories such that:

$$
\mathbb{E}[|X_t - Y_t|^2] \le e^{-\lambda t} |X_0 - Y_0|^2
$$

For non-uniformly-convex potentials, this requires **localization arguments** (partition space into regions where $U$ is approximately convex).

### 5.3. Step 2: Wasserstein Behavior of Cloning Operator

**Key question:** Is $\Psi_{\text{clone}}$ a contraction, expansion, or neither in Wasserstein distance?

**Analysis:** The cloning operator:
1. **Contracts within alive walkers:** Moves walkers toward high-fitness regions (decreases spread)
2. **Expands via noise:** Adds Gaussian jitter $\mathcal{N}(0, \delta^2 I_d)$ (increases spread)

**Net effect:** Depends on parameters $\delta$ and the fitness landscape.

:::{prf:lemma} Cloning Operator Wasserstein Bound
:label: lem-cloning-wasserstein

For the cloning operator $\Psi_{\text{clone}}$ with noise scale $\delta$:

$$
W_2^2(\Psi_{\text{clone},*} \mu, \Psi_{\text{clone},*} \nu) \le (1 - \kappa_{\text{clone}}) W_2^2(\mu, \nu) + C_{\text{noise}} \delta^2
$$

where:
- $\kappa_{\text{clone}} \in (0,1)$ captures the contraction from fitness-based selection
- $C_{\text{noise}}$ is a constant depending on $N$ and the fitness landscape

:::

**Proof strategy:** Use the **triangle inequality** for Wasserstein distance:

$$
W_2(\Psi_{\text{clone},*} \mu, \Psi_{\text{clone},*} \nu) \le W_2(\Psi_{\text{clone},*} \mu, \Psi_{\text{clone, no noise},*} \mu) + W_2(\Psi_{\text{clone, no noise},*} \mu, \Psi_{\text{clone, no noise},*} \nu)
$$

The first term is bounded by $\delta$ (due to Gaussian noise). The second term contracts due to fitness selection.

### 5.4. Step 3: HWI Inequality Application (REQUIRES CURVATURE)

**Critical Prerequisite:** The HWI inequality ({prf:ref}`thm-hwi-inequality`) requires that $\pi$ satisfies the curvature condition $\text{CD}(\rho, \infty)$ for some $\rho > 0$. This is the **same requirement as Approach A2** (Bakry-Émery curvature).

:::{important}
**Circularity Warning:** Approach A4 does NOT bypass the difficulty of Approach A2. Proving that $\pi_{\text{QSD}}$ satisfies $\text{CD}(\rho, \infty)$ is the central challenge of A2, rated as "HIGH" difficulty. A4 inherits this same difficulty as a prerequisite for applying the HWI inequality.
:::

**Assuming the curvature condition holds**, we can combine Wasserstein and KL bounds via HWI:

From {prf:ref}`thm-hwi-inequality`:

$$
D_{\text{KL}}(\mu_t \| \pi) \le W_2(\mu_t, \pi) \sqrt{I(\mu_t \| \pi)} - \frac{\rho}{2} W_2^2(\mu_t, \pi)
$$

Rearrange to solve for Fisher information:

$$
\sqrt{I(\mu_t \| \pi)} \ge \frac{D_{\text{KL}}(\mu_t \| \pi) + \frac{\rho}{2} W_2^2(\mu_t, \pi)}{W_2(\mu_t, \pi)}
$$

**Scenario 1:** If $W_2(\mu_t, \pi)$ decays faster than $\sqrt{D_{\text{KL}}(\mu_t \| \pi)}$, then Fisher information is dominated by the KL term.

**Scenario 2:** If $W_2(\mu_t, \pi)$ decays slower, then the Wasserstein term dominates.

In both cases, we get an **upper and lower bound** on $I(\mu_t \| \pi)$.

:::{prf:conjecture} Fisher-Rao Distance from Fisher Information
:label: conj-fisher-rao-integration

If $I(\mu_t \| \pi) \le C e^{-\lambda t}$, then along the evolution path $\{\mu_s\}_{s \in [0,t]}$:

$$
d_{\text{FR}}(\mu_t, \pi) \le \int_0^t \sqrt{I(\mu_s \| \pi)} \, ds \le \frac{\sqrt{C}}{\sqrt{\lambda/2}} (1 - e^{-\lambda t/2})
$$

**Proof requirements:** Show that the actual evolution path does not deviate significantly from the geodesic, controlling the "wiggliness" of the trajectory in Fisher-Rao geometry.
:::

### 5.5. Feasibility Assessment for A4

**Required Steps:**
1. ⚠️ **Moderate difficulty:** Prove {prf:ref}`conj-wasserstein-coercive` using coupling methods
2. ✅ **Tractable:** Establish {prf:ref}`lem-cloning-wasserstein` via triangle inequality
3. ❌ **CRITICAL BLOCKER:** Prove $\text{CD}(\rho, \infty)$ curvature condition (same as Approach A2)
4. ⚠️ **Difficult:** Prove {prf:ref}`conj-fisher-rao-integration` to bound distance from Fisher information

**Gaps to Address:**
- **Gap 1 (CRITICAL):** The HWI inequality requires $\text{CD}(\rho, \infty)$ with $\rho > 0$—this is the **central challenge of Approach A2**, which is rated as "HIGH" difficulty
- **Gap 2:** Integrating Fisher information to get Fisher-Rao distance requires controlling the "path" of convergence
- **Gap 3:** Tensorization of HWI to the $N$-particle swarm space
- **Gap 4:** For non-uniformly convex potentials, the curvature bound may not exist or may require localized analysis

**Time Estimate:** 2-3 months (comparable to A2 due to curvature requirement)

**Payoff:** High—**IF** curvature can be established, provides dual metric convergence (KL + Wasserstein) with Fisher-Rao bounds

**Recommendation:** ⚠️ **Do NOT pursue until curvature condition is resolved**—A4 inherits the core difficulty of A2 and cannot proceed independently. Consider a modified strategy:
- **Alt-A4a:** Focus only on Wasserstein convergence (bypass HWI, no Fisher-Rao result)
- **Alt-A4b:** Search for weaker versions of HWI that work with $\text{CD}(0, \infty)$ (non-negative curvature)

---

## 6. Strategic Roadmap and Recommendations

### 6.1. Recommended Sequence (REVISED)

**IMPORTANT:** The initial recommended sequence has been **substantially revised** after identifying critical mathematical errors and hidden circularities in the original plan.

**Phase 1 (Immediate, 3-4 weeks):** Curvature Investigation
- **Goal:** Determine if $\pi_{\text{QSD}}$ satisfies any curvature conditions (strong or weak)
- **Deliverable:** Technical report on curvature properties; proof or counterexample for $\text{CD}(\rho, \infty)$
- **Risk:** Medium—exploratory, may yield negative results
- **Outcome:** **Gates all subsequent approaches**—determines feasibility of A2, A4, and informs A1 strategy

**Phase 2 (Short-term, 1-2 months):** Alt-A4a — Wasserstein Convergence (Standalone)
- **Goal:** Prove Wasserstein convergence WITHOUT requiring HWI or Fisher-Rao bounds
- **Deliverable:** Theorem showing $W_2(\mu_t, \pi_{\text{QSD}}) \le Ce^{-\lambda_W t}$ with explicit constants
- **Risk:** Low-Medium—coupling methods are well-established
- **Outcome:** **Publishable standalone result**; useful even without Fisher-Rao convergence

**Phase 3 (Conditional, 1-2 months):** Approach A1 — Reverse LSI + Fisher Information
- **Goal:** Prove Conjecture {prf:ref}`conj-reverse-lsi` (upper bound on Fisher information)
- **Deliverable:** Reverse LSI theorem; Fisher information decay bounds
- **Risk:** Medium-High—requires proving non-standard functional inequality
- **Outcome:** Enables Fisher information control; partial progress toward Fisher-Rao convergence
- **Condition:** Only pursue if Phase 1 reveals favorable curvature properties

**Phase 4 (Future work, 2-3 months):** Approach A2/A4 — Full Fisher-Rao Convergence
- **Goal:** Combine curvature bounds + Wasserstein + reverse LSI for complete Fisher-Rao result
- **Deliverable:** Theorem showing $d_{\text{FR}}(\mu_t, \pi_{\text{QSD}}) \le Ce^{-\lambda_{\text{FR}} t}$
- **Risk:** High—requires success in all previous phases
- **Outcome:** Strongest possible convergence result with geometric interpretation
- **Condition:** Only feasible if $\text{CD}(\rho, \infty)$ holds with $\rho > 0$

**Phase 5 (Long-term research, 4-6 months):** Approach A3 — Otto Calculus
- **Goal:** Interpret Euclidean Gas as Wasserstein/Fisher-Rao gradient flow
- **Deliverable:** JKO scheme formulation and convergence proof
- **Risk:** Very high—research frontier
- **Outcome:** Major theoretical contribution, suitable for PhD thesis
- **Condition:** Pursue only if Phases 1-4 are successful and reveal gradient flow structure

### 6.2. Detailed Work Breakdown for Phase 1 (Curvature Investigation)

#### Task 1.1: Literature Review on Curvature for Non-Convex Systems (1 week)

**Objective:** Survey existing results on curvature bounds for measures with non-uniformly-convex potentials

**Steps:**
1. Review Bakry-Émery theory for coercive (but not convex) potentials
2. Study localization techniques (Hairer-Mattingly, Majka)
3. Investigate "weak curvature" conditions: $\text{CD}(0, \infty)$, local curvature bounds
4. Examine curvature for product measures on $\Sigma_N$

**Deliverable:** Annotated bibliography + summary of applicable techniques

#### Task 1.2: Analyze the Confining Potential $U$ (1 week)

**Objective:** Characterize geometric properties of the Euclidean Gas confining potential

**Steps:**
1. Extract the explicit form of $U(x)$ from {prf:ref}`axiom-confining-potential`
2. Compute Hessian $\text{Hess}(U)$ and identify regions of:
   - Positive definiteness (locally convex)
   - Zero eigenvalues (flat directions)
   - Negative eigenvalues (saddle regions, if any)
3. Determine if $U$ satisfies any **displacement convexity** conditions in Wasserstein space
4. Check for logarithmic concavity: $e^{-U}$ log-concave?

**Deliverable:** Technical report on potential geometry

#### Task 1.3: Attempt Curvature Bound Proof (1-2 weeks)

**Objective:** Prove or disprove that $\pi_{\text{QSD}}$ satisfies $\text{CD}(\rho, \infty)$

**Steps:**
1. For the kinetic operator: Use hypocoercivity + potential geometry to bound:

$$
\mathcal{L}_{\text{kin}}(\Gamma(f)) \ge \rho_{\text{eff}} \Gamma(f) + \text{error terms}
$$

2. For the cloning operator: Analyze whether discrete jumps preserve or destroy curvature
3. For the composition: Apply tensorization and seesaw arguments from [10_kl_convergence.md](10_kl_convergence/10_kl_convergence.md)
4. If strong curvature fails, attempt to prove weak curvature $\text{CD}(0, \infty)$ or **local curvature bounds**

**Deliverable:** Proof, counterexample, or partial result with explicit obstacles

#### Task 1.4: Decision Gate (1-2 days)

**Objective:** Based on Task 1.3 results, determine the feasibility of each approach

**Decision Tree:**
- **If $\text{CD}(\rho, \infty)$ with $\rho > 0$ holds:** ✅ Proceed with Phases 2-4 (A4 becomes feasible)
- **If only $\text{CD}(0, \infty)$ holds:** ⚠️ Proceed with Phase 2 (Wasserstein only); investigate weaker HWI inequalities for Phase 4
- **If no curvature bounds exist:** ❌ Skip A2/A4 entirely; focus on Phase 2 (Wasserstein) and Phase 3 (reverse LSI) as independent results

**Deliverable:** Strategic decision memo + updated project timeline

### 6.3. Success Criteria (REVISED)

**Minimum Success (Phase 1-2):**
- [ ] Technical report on curvature properties of $\pi_{\text{QSD}}$ (Phase 1)
- [ ] Theorem proving Wasserstein convergence: $W_2(\mu_t, \pi) \le Ce^{-\lambda_W t}$ (Phase 2)
- [ ] Explicit constants in terms of $\gamma, \kappa, \delta, N$
- [ ] All proofs pass Gemini mathematical review
- [ ] **Outcome:** Publishable result on Wasserstein convergence for particle swarm systems

**Target Success (Phase 1-3):**
- [ ] Proof or counterexample for $\text{CD}(\rho, \infty)$ curvature condition (Phase 1)
- [ ] Reverse LSI theorem (Conjecture {prf:ref}`conj-reverse-lsi`) with explicit constants (Phase 3)
- [ ] Fisher information decay bound: $I(\mu_t \| \pi) \le Ce^{-\lambda_I t}$
- [ ] Wasserstein + Fisher information dual-metric convergence
- [ ] **Outcome:** Publishable paper on information-geometric convergence

**Stretch Goal (Phase 1-4):**
- [ ] Curvature bound $\text{CD}(\rho, \infty)$ with $\rho > 0$ (Phase 1)
- [ ] Complete Fisher-Rao distance convergence: $d_{\text{FR}}(\mu_t, \pi) \le Ce^{-\lambda_{\text{FR}} t}$ (Phase 4)
- [ ] Unified theorem combining KL + Wasserstein + Fisher-Rao metrics
- [ ] Explicit connection to Hessian-based diffusion in Adaptive Gas
- [ ] **Outcome:** Major theoretical contribution to stochastic optimization literature

**Blue-Sky Goal (Phase 1-5):**
- [ ] JKO gradient flow formulation (Approach A3, Phase 5)
- [ ] Prove Euclidean Gas is Wasserstein/Fisher-Rao gradient flow of entropy functional
- [ ] **Outcome:** PhD thesis–level contribution bridging particle methods and optimal transport

### 6.4. Open Questions and Future Directions

**Question 1:** Can the cloning operator be interpreted as a **gradient flow** in a non-standard metric (e.g., unbalanced optimal transport)?

**Question 2:** Does the Euclidean Gas satisfy a **modified Bakry-Émery condition** that accounts for the discrete status variable $s_i \in \{0,1\}$?

**Question 3:** In the **mean-field limit** ($N \to \infty$), does the Fisher-Rao geometry simplify to a standard Wasserstein gradient flow?

**Question 4:** Can the **Adaptive Gas** (with Hessian-based anisotropic diffusion) achieve **faster Fisher-Rao convergence** by exploiting local curvature?

**Question 5:** Is there a **discrete-time Otto calculus** that directly handles the operator composition $\Psi_{\text{kin}} \circ \Psi_{\text{clone}}$ without taking limits?

### 6.5. Literature Review Checklist

**Essential References:**
- [ ] **Villani (2009):** *Hypocoercivity*, Memoirs of the AMS
- [ ] **Otto & Villani (2000):** "Generalization of an inequality by Talagrand and links with the logarithmic Sobolev inequality"
- [ ] **Otto (2001):** "The geometry of dissipative evolution equations: the porous medium equation"
- [ ] **Bakry & Émery (1985):** "Diffusions hypercontractives"
- [ ] **Eberle (2016):** "Reflection couplings and contraction rates for diffusions"
- [ ] **Chizat et al. (2018):** "An interpolating distance between optimal transport and Fisher-Rao"
- [ ] **Majka (2020):** "Transportation inequalities for non-globally dissipative SDEs"

**Supplementary References:**
- [ ] **Jordan, Kinderlehrer, Otto (1998):** "The variational formulation of the Fokker-Planck equation"
- [ ] **Ambrosio, Gigli, Savaré (2008):** *Gradient Flows in Metric Spaces and in the Space of Probability Measures*
- [ ] **Pavliotis (2014):** *Stochastic Processes and Applications* (Chapter on hypocoercivity)

---

## 7. Conclusion

### 7.1. Summary of Findings

We have analyzed four distinct approaches to proving Fisher-Rao convergence for the Euclidean Gas:

1. **A1 (Direct LSI Implications):** ✅ Feasible, low-risk, moderate payoff
2. **A2 (Bakry-Émery Curvature):** ⚠️ Technically challenging, high payoff if successful
3. **A3 (Otto Calculus):** ❌ Currently infeasible, research frontier
4. **A4 (KL + Wasserstein Hybrid):** ✅ Tractable, high payoff

### 7.2. Recommended Path Forward

**Immediate action (next 4 weeks):**
- Pursue **Approach A1** to establish Fisher information decay
- Document results in new section of [10_kl_convergence.md](10_kl_convergence/10_kl_convergence.md)

**Medium-term (next 2 months):**
- Extend to **Approach A4** for Wasserstein convergence
- Apply HWI inequality to obtain dual-metric bounds
- Target publication in optimization/probability journal

**Long-term (future research):**
- Investigate **Approach A2** if curvature bounds emerge from other analyses
- Keep **Approach A3** as a "blue-sky" research direction for PhD-level work

### 7.3. Expected Outcomes

**By end of Phase 1:**
- Theorem proving exponential decay of Fisher information
- Foundation for Fisher-Rao distance bounds
- Deeper understanding of LSI structure

**By end of Phase 2:**
- Dual-metric convergence theorem (KL + Wasserstein)
- Geometric interpretation of Euclidean Gas dynamics
- Publishable results

**Long-term impact:**
- Establish Fragile framework as a **geometrically principled** optimization method
- Enable extensions to Riemannian state spaces (Adaptive Gas)
- Connect to broader optimal transport and information geometry literature

---

## References

:::{note}
This roadmap document will be continuously updated as progress is made. Consult [10_kl_convergence.md](10_kl_convergence/10_kl_convergence.md) for the current state of convergence proofs.
:::

**Framework Documents:**
- [01_fragile_gas_framework.md](01_fragile_gas_framework.md) — Foundational axioms and definitions
- [02_euclidean_gas.md](02_euclidean_gas.md) — Euclidean Gas specification
- [04_convergence.md](04_convergence.md) — Total variation convergence
- [10_kl_convergence.md](10_kl_convergence/10_kl_convergence.md) — KL-divergence convergence

**Key Papers:**
1. Villani, C. (2009). *Hypocoercivity*. Memoirs of the American Mathematical Society.
2. Otto, F., & Villani, C. (2000). Generalization of an inequality by Talagrand and links with the logarithmic Sobolev inequality. *Journal of Functional Analysis*, 173(2), 361-400.
3. Otto, F. (2001). The geometry of dissipative evolution equations: the porous medium equation. *Communications in Partial Differential Equations*, 26(1-2), 101-174.
4. Bakry, D., & Émery, M. (1985). Diffusions hypercontractives. *Séminaire de probabilités de Strasbourg*, 19, 177-206.
5. Eberle, A. (2016). Reflection couplings and contraction rates for diffusions. *Probability Theory and Related Fields*, 166(3-4), 851-886.
6. Chizat, L., et al. (2018). An interpolating distance between optimal transport and Fisher-Rao. *Foundations of Computational Mathematics*, 18(1), 1-44.
