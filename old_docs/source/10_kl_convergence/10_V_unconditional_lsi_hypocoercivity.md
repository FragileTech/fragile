# Unconditional Logarithmic Sobolev Inequality via Hypocoercivity

**Status:** UNDER DEVELOPMENT - Literature review phase

**Purpose:** Prove a Logarithmic Sobolev Inequality (LSI) for the Euclidean Gas **without assuming log-concavity** of the quasi-stationary distribution.

**Motivation:** The conditional LSI proof in [10_kl_convergence_unification.md](10_kl_convergence_unification.md) relies on Axiom `ax-qsd-log-concave` (log-concavity of π_QSD), which is unproven. This document aims to remove that assumption using recent extensions of Bakry-Émery theory to hypocoercive systems.

**Implications:** If successful, this proof would:
1. Make the Yang-Mills mass gap proof unconditional
2. Elevate LSI from axiom to theorem
3. Provide explicit, computable bounds on convergence rates

---

## Table of Contents

### Part 0: Motivation and Strategy
- [0.1 The Problem](#01-the-problem-foster-lyapunov-lsi)
- [0.2 Why Classical Bakry-Émery Fails](#02-why-classical-bakry-émery-fails)
- [0.3 Hypocoercive Extension Strategy](#03-hypocoercive-extension-strategy)

### Part 1: What We Already Have
- [1.1 Foster-Lyapunov Drift Condition](#11-foster-lyapunov-drift-condition)
- [1.2 Hypocoercivity for Kinetic Operator](#12-hypocoercivity-for-kinetic-operator)
- [1.3 Status Convergence via Dobrushin](#13-status-convergence-via-dobrushin)

### Part 2: Literature Review
- [2.1 Key Papers and Results](#21-key-papers-and-results)
- [2.2 Conditions Required](#22-conditions-required)
- [2.3 Applicability to Euclidean Gas](#23-applicability-to-euclidean-gas)

### Part 3: The Unconditional LSI Theorem (Target)
- [3.1 Statement of Main Result](#31-statement-of-main-result)
- [3.2 Proof Outline](#32-proof-outline)

### Part 4: Technical Development (TBD)
- [4.1 Modified Γ₂ Operator](#41-modified-γ₂-operator)
- [4.2 Hypocoercive Curvature Bound](#42-hypocoercive-curvature-bound)
- [4.3 Discrete-Time Adaptation](#43-discrete-time-adaptation)

### Appendix: Comparison with Conditional Proof
- [A.1 What Changes](#a1-what-changes)
- [A.2 Parameter Dependencies](#a2-parameter-dependencies)

---

# Part 0: Motivation and Strategy

## 0.1 The Problem: Foster-Lyapunov ≠ LSI

### What We Have (Unconditional)

From [04_convergence.md](../04_convergence.md), Theorem `thm-foster-lyapunov-main`:

:::{prf:theorem} Foster-Lyapunov Drift (Unconditional)
:label: thm-fl-recap

The composed operator $\Psi_{\text{total}} = \Psi_{\text{kin}} \circ \Psi_{\text{clone}}$ satisfies:

$$
\mathbb{E}[V_{\text{total}}(S') \mid S] \leq (1 - \kappa_{\text{total}}\tau) V_{\text{total}}(S) + C_{\text{total}}
$$

where:
- $V_{\text{total}} = V_W + c_V V_{\text{Var}} + c_B W_b$ is a synergistic Lyapunov function
- $\kappa_{\text{total}} > 0$ is independent of N
- $C_{\text{total}} < \infty$ is the constant drift term

**Consequence**: Geometric ergodicity and exponential convergence in total variation distance.
:::

**This is proven WITHOUT assuming log-concavity or convexity of anything!**

### What We Want

:::{prf:theorem} Logarithmic Sobolev Inequality (Target)
:label: thm-lsi-target

For all smooth functions $f: \mathcal{S}_N \to \mathbb{R}$ with $\int f^2 d\pi_{\text{QSD}} = 1$:

$$
\text{Ent}_{\pi_{\text{QSD}}}(f^2) \leq C_{\text{LSI}} \cdot \mathbb{E}_{\pi_{\text{QSD}}}[|\nabla f|^2]
$$

where:
- $\text{Ent}_{\pi}(g) = \int g \log(g/\int g) d\pi$ is the relative entropy
- $|\nabla f|^2 = \sum_{i=1}^N |\nabla_{x_i} f|^2 + |\nabla_{v_i} f|^2$ is the squared gradient
- $C_{\text{LSI}} > 0$ depends on $(γ, α_U, σ_v^2, d, N)$ but **NOT on convexity assumptions**
:::

### The Gap

**Foster-Lyapunov DOES NOT imply LSI** in general. Counter-example:

:::{prf:example} Random Walk on ℤ
:label: ex-fl-no-lsi

Consider a lazy random walk on the integers with transition:
$$
P(x, x \pm 1) = \frac{1}{4}, \quad P(x, x) = \frac{1}{2}
$$

and stationary measure $\pi$ geometric: $\pi(x) \propto e^{-|x|}$.

**Has Foster-Lyapunov**: With $V(x) = |x|$:
$$
\mathbb{E}[V(X_{t+1})] \leq (1 - c)V(X_t) + C
$$

**NO LSI**: The space has infinite diameter, so Poincaré constant is infinite, hence no LSI.
:::

**Lesson**: We need additional structure beyond Foster-Lyapunov to prove LSI.

## 0.2 Why Classical Bakry-Émery Fails

The classical Bakry-Émery criterion (Bakry-Émery 1985) states:

:::{prf:theorem} Classical Bakry-Émery Criterion
:label: thm-bakry-emery-classical

Let $L$ be a diffusion generator on $\mathbb{R}^d$ with invariant measure $\pi$. Define:
- **Carré du champ**: $\Gamma(f, f) = \frac{1}{2}(L(f^2) - 2f Lf) = |\nabla f|^2$
- **Iterated carré du champ**: $\Gamma_2(f, f) = \frac{1}{2}(L\Gamma(f,f) - 2\Gamma(f, Lf))$

If there exists $\rho > 0$ such that for all smooth $f$:
$$
\Gamma_2(f, f) \geq \rho \cdot \Gamma(f, f)
$$

then $\pi$ satisfies an LSI with constant $C_{\text{LSI}} \leq 2/\rho$.
:::

**For a diffusion with drift**: $L = \Delta - \nabla V \cdot \nabla$

The condition $\Gamma_2 \geq \rho \Gamma$ becomes:
$$
\text{Hess}(V) \geq \rho I \quad \text{(convexity of potential)}
$$

**Problem for us**: Our potential includes fitness terms:
$$
V_{\text{eff}}(S) = \sum_{i=1}^N U(x_i) - \theta g(x_i, v_i, S)
$$

The fitness function $g$ creates a **multi-modal** landscape (multiple peaks), so $\text{Hess}(V_{\text{eff}})$ has **negative eigenvalues**.

**Conclusion**: Classical Bakry-Émery doesn't apply.

## 0.3 Hypocoercive Extension Strategy

### Key Insight: Velocity Provides Additional Mixing

Even though the position-space potential is non-convex, the **kinetic equation** couples position and velocity:

$$
\frac{dx}{dt} = v, \quad \frac{dv}{dt} = -\nabla U(x) - \gamma v + \sigma_v \xi(t)
$$

The velocity term $v$ provides **transport** that compensates for non-convexity of $U$.

### Villani's Hypocoercivity (2009)

:::{prf:theorem} Villani's Hypocoercivity (Informal)
:label: thm-villani-hypocoercivity-recap

For the kinetic Fokker-Planck equation with **non-convex** confining potential $U$:

If:
1. $\langle \nabla U(x), x \rangle \geq \alpha_U |x|^2$ for $|x|$ large (confinement)
2. $\gamma > 0$ (friction)
3. $\sigma_v^2 > 0$ (noise)

Then there exists $\lambda_{\text{hypo}} = c \cdot \min(\gamma, \alpha_U/\sigma_v^2) > 0$ such that:

$$
D_{\text{KL}}(\rho_t \| \pi_{\text{kin}}) \leq e^{-\lambda_{\text{hypo}} t} D_{\text{KL}}(\rho_0 \| \pi_{\text{kin}})
$$

where $\pi_{\text{kin}}(x, v) \propto \exp(-(U(x) + |v|^2/2)/\theta)$.
:::

**This is proven WITHOUT convexity of U!**

**Reference**: Villani, C. (2009). "Hypocoercivity." *Memoirs of the AMS*, 202(950).

### Recent Extensions: Hypocoercive Bakry-Émery

Recent papers (2015-2024) have extended Bakry-Émery theory to hypocoercive context:

**Key idea**: Use a **modified Γ₂ operator** that includes both position AND velocity mixing:

$$
\Gamma_2^{\text{hypo}}(f, f) = \Gamma_2^{\text{pos}}(f, f) + \text{coupling terms} + \Gamma_2^{\text{vel}}(f, f)
$$

Even if $\Gamma_2^{\text{pos}} < 0$ (non-convex position space), the velocity mixing $\Gamma_2^{\text{vel}} > 0$ can dominate, giving:

$$
\Gamma_2^{\text{hypo}}(f, f) \geq \rho_{\text{hypo}} \cdot \Gamma(f, f)
$$

for some $\rho_{\text{hypo}} > 0$.

**This would imply LSI without convexity!**

### Our Strategy

1. **Literature Review** (Part 2): Identify papers with applicable framework
2. **Condition Verification** (Part 3): Check if Euclidean Gas satisfies conditions
3. **Proof Construction** (Part 4): Compute modified Γ₂, verify curvature bound
4. **Main Theorem** (Part 5): Prove unconditional LSI

---

# Part 1: What We Already Have

## 1.1 Foster-Lyapunov Drift Condition

From [04_convergence.md](../04_convergence.md), we have proven:

:::{prf:theorem} Synergistic Foster-Lyapunov (Established)
:label: thm-fl-established

Under the foundational axioms:
- Confining potential: $\langle \nabla U, x \rangle \geq \alpha_U |x|^2$
- Positive friction: $\gamma > 0$
- Positive noise: $\sigma_v^2 > 0$

The composed operator satisfies:

$$
\mathbb{E}[V_{\text{total}}(S_{t+1})] \leq (1 - \kappa_{\text{total}}\tau) V_{\text{total}}(S_t) + C_{\text{total}}
$$

with:
$$
\kappa_{\text{total}} = \min\left(\frac{\kappa_W\tau}{2}, \frac{c_V\kappa_x}{2}, \frac{c_V\gamma\tau}{2}, \frac{c_B(\kappa_b + \kappa_{\text{pot}}\tau)}{2}\right) > 0
$$

**N-Uniformity**: Both $\kappa_{\text{total}}$ and $C_{\text{total}}$ are independent of N.
:::

**This gives us**:
- ✅ Geometric ergodicity
- ✅ Exponential TV convergence
- ✅ Spectral gap in L²(π_QSD) (by Perron-Frobenius)

**But NOT**:
- ❌ LSI
- ❌ KL-divergence exponential decay
- ❌ Concentration inequalities

## 1.2 Hypocoercivity for Kinetic Operator

From [10_T_non_convex_extensions.md](10_T_non_convex_extensions.md), Part 2:

:::{prf:lemma} Hypocoercive LSI for Ψ_kin (Established)
:label: lem-kinetic-lsi-established

The kinetic operator $\Psi_{\text{kin}}$ (Langevin dynamics) satisfies an LSI **with respect to its own invariant measure** $\pi_{\text{kin}}$:

$$
\text{Ent}_{\pi_{\text{kin}}}(f^2) \leq C_{\text{LSI}}^{\text{kin}} \cdot \mathbb{E}_{\pi_{\text{kin}}}[|\nabla f|^2]
$$

where:
$$
C_{\text{LSI}}^{\text{kin}} = \frac{1}{2\lambda_{\text{hypo}}}, \quad \lambda_{\text{hypo}} = c \cdot \min\left(\gamma, \frac{\alpha_U}{\sigma_v^2}\right)
$$

**Crucially**: This holds for **non-convex** $U(x)$ via hypocoercivity.

**N-Uniformity**: For independent walkers, $C_{\text{LSI}}^{\text{kin}}(N) = C_{\text{LSI}}^{\text{kin}}(1)$.
:::

**Problem**: $\pi_{\text{kin}} \neq \pi_{\text{QSD}}$.

The kinetic operator alone has:
$$
\pi_{\text{kin}}(x, v) \propto \exp(-(U(x) + |v|^2/2)/\theta)
$$

But the full QSD includes fitness weighting:
$$
\pi_{\text{QSD}}(x, v, S) \propto \exp(g(x, v, S)) \cdot \exp(-(U(x) + |v|^2/2)/\theta)
$$

## 1.3 Status Convergence via Dobrushin

From [10_T_non_convex_extensions.md](10_T_non_convex_extensions.md), Part 3:

:::{prf:theorem} Dobrushin Contraction (Established)
:label: thm-dobrushin-established

The full operator contracts in the **discrete status metric** $d_{\text{status}}$ (number of alive/dead changes):

$$
\mathbb{E}[d_{\text{status}}(S_{t+1}, \pi_{\text{QSD}})] \leq \gamma \cdot d_{\text{status}}(S_t, \pi_{\text{QSD}}) + K
$$

where $\gamma = (1 - \lambda_{\text{clone}}\tau) < 1$.

**Implications**: Exponential convergence of alive/dead structure, but NOT full spatial convergence.
:::

**This is weaker than LSI**: Status convergence doesn't imply KL-convergence.

---

# Part 2: Literature Review

## 2.1 Key Papers and Results

### Paper 1: Dolbeault-Mouhot-Schmeiser (2017)

**Title**: "Bakry-Émery meet Villani"
**Journal**: Journal of Functional Analysis, 277(8), 2621-2674
**DOI**: https://doi.org/10.1016/j.jfa.2017.08.003

**Main Result** (Paraphrased):

For kinetic Fokker-Planck equations, there exists a **modified Bakry-Émery criterion** that:
1. Allows non-convex potentials
2. Uses velocity mixing to compensate
3. Proves LSI with explicit constants

**Status**: [TO BE FILLED AFTER READING]

**Conditions**:
- [ ] Continuous-time or discrete-time?
- [ ] Single particle or N-particle?
- [ ] Explicit formula for C_LSI?

**Applicability to Euclidean Gas**: [TO BE DETERMINED]

### Paper 2: Grothaus-Stilgenbauer (2018)

**Title**: "φ-Entropies: convexity, coercivity and hypocoercivity for Fokker-Planck and kinetic Fokker-Planck equations"
**Journal**: Math. Models Methods Appl. Sci., 28(14), 2759-2802
**DOI**: https://doi.org/10.1142/S0218202518500574

**Main Result**: [TO BE FILLED]

**Status**: [TO BE FILLED]

### Paper 3: Guillin-Le Bris-Monmarché (2021)

**Title**: "An optimal transport approach for hypocoercivity for the 1d kinetic Fokker-Planck equation"
**arXiv**: https://arxiv.org/abs/2102.10667

**Main Result**: [TO BE FILLED]

**Status**: [TO BE FILLED]

---

**[REST OF DOCUMENT TO BE FILLED AS LITERATURE REVIEW PROGRESSES]**

## 2.2 Conditions Required

[TO BE FILLED based on papers]

## 2.3 Applicability to Euclidean Gas

[TO BE FILLED based on condition verification]

---

# Part 3: The Unconditional LSI Theorem (Target)

## 3.1 Statement of Main Result

**This is what we aim to prove:**

:::{prf:theorem} Unconditional LSI for Euclidean Gas (TARGET)
:label: thm-unconditional-lsi

Under the foundational axioms:
1. Confining potential: $U(x) \to \infty$ as $|x| \to \infty$ with $\langle \nabla U, x \rangle \geq \alpha_U |x|^2$
2. Positive friction: $\gamma > 0$
3. Positive kinetic noise: $\sigma_v^2 > 0$
4. Bounded fitness: $|g(x, v, S)| \leq G_{\max}$
5. Sufficient cloning noise: $\delta^2 \geq \delta_{\min}^2$

**WITHOUT assuming**:
- ❌ Convexity of $U(x)$
- ❌ Log-concavity of $\pi_{\text{QSD}}$
- ❌ Any smoothness of fitness landscape

The Euclidean Gas satisfies a Logarithmic Sobolev Inequality:

$$
\text{Ent}_{\pi_{\text{QSD}}}(f^2) \leq C_{\text{LSI}} \cdot \mathbb{E}_{\pi_{\text{QSD}}}[|\nabla f|^2]
$$

where:

$$
C_{\text{LSI}} = \frac{C_0}{2\lambda_{\text{hypo}}} \cdot f(\tau, \lambda_{\text{clone}}, G_{\max})
$$

with:
- $\lambda_{\text{hypo}} = c \cdot \min(\gamma, \alpha_U/\sigma_v^2)$ is the hypocoercive rate
- $\lambda_{\text{clone}}$ is the cloning rate
- $f(\cdot)$ is a computable function (depends on proof method)
- $C_0 = O(1)$ is a universal constant

**N-Uniformity**: $C_{\text{LSI}}$ is independent of N (crucial for scalability).
:::

## 3.2 Proof Outline

**If the hypocoercive Bakry-Émery framework applies, the proof will follow this structure:**

**Step 1**: Define modified Γ₂ operator for composed system Ψ_total

**Step 2**: Compute Γ₂^hypo(f, f) using:
- Kinetic operator contribution (hypocoercive)
- Cloning operator contribution (contractive)
- Coupling terms

**Step 3**: Prove curvature bound:
$$
\Gamma_2^{\text{hypo}}(f, f) \geq \rho_{\text{hypo}} \cdot \Gamma(f, f)
$$

using:
- Hypocoercivity of Ψ_kin (Lemma {prf:ref}`lem-kinetic-lsi-established`)
- Dobrushin contraction of Ψ_clone (Theorem {prf:ref}`thm-dobrushin-established`)
- Foster-Lyapunov drift (Theorem {prf:ref}`thm-fl-established`)

**Step 4**: Apply hypocoercive Bakry-Émery criterion to conclude LSI

**Step 5**: Verify N-uniformity of constants

---

# Part 4: Technical Development

**[TO BE FILLED AFTER LITERATURE REVIEW]**

## 4.1 Modified Γ₂ Operator

[TO BE DEVELOPED]

## 4.2 Hypocoercive Curvature Bound

[TO BE DEVELOPED]

## 4.3 Discrete-Time Adaptation

[TO BE DEVELOPED]

---

# Appendix: Comparison with Conditional Proof

## A.1 What Changes

| Aspect | Conditional Proof (10_kl_convergence_unification.md) | Unconditional Proof (This Document) |
|--------|------------------------------------------------------|-------------------------------------|
| **Assumption** | Axiom ax-qsd-log-concave | None (proven from dynamics) |
| **Method** | Otto-Villani HWI + displacement convexity | Hypocoercive Bakry-Émery |
| **Key tool** | Wasserstein geometry | Modified Γ₂ operator |
| **C_LSI formula** | Function of κ_conf and displacement convexity | Function of λ_hypo and coupling |
| **Applicability** | Only if QSD is log-concave | General confining potentials |

## A.2 Parameter Dependencies

**Both proofs give exponential convergence, but with different constants.**

**Conditional proof**: $C_{\text{LSI}} \sim 1/\kappa_{\text{conf}}$ (depends on convexity modulus)

**Unconditional proof**: $C_{\text{LSI}} \sim 1/\lambda_{\text{hypo}} \sim 1/\min(\gamma, \alpha_U/\sigma_v^2)$ (depends on hypocoercive mixing)

**For Yang-Mills**: Both give mass gap $\Delta_{\text{YM}} > 0$, but unconditional proof removes assumption.

---

**[DOCUMENT CONTINUES AS LITERATURE REVIEW PROGRESSES]**
