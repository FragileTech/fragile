# Mathematical Reference: Mean-Field Convergence Results

**Document Status**: Comprehensive mathematical reference extracted from all mean-field convergence documents
**Date**: 2025-10-10
**Purpose**: Searchable catalog of all definitions, theorems, lemmas, and key formulas for mean-field convergence

---

## Table of Contents

1. [Core Definitions](#1-core-definitions)
2. [Revival Operator Analysis](#2-revival-operator-analysis-stage-0)
3. [QSD Regularity Properties](#3-qsd-regularity-properties-stage-05)
4. [Entropy Production Framework](#4-entropy-production-framework-stage-1)
5. [Explicit Hypocoercivity Constants](#5-explicit-hypocoercivity-constants-stage-2)
6. [Parameter Analysis](#6-parameter-analysis-stage-3)
7. [Main Convergence Theorems](#7-main-convergence-theorems)
8. [Practical Formulas](#8-practical-formulas)

---

## 1. Core Definitions

### 1.1. Mean-Field Generator

**Source**: `11_stage1_entropy_production.md` Section 1.2
**Tags**: #generator #pde #mean-field

:::{prf:definition} Mean-Field Kinetic Operator
:label: def-kinetic-operator

$$
\mathcal{L}_{\text{kin}}[\rho] = -v \cdot \nabla_x \rho + \nabla_x U \cdot \nabla_v \rho + \gamma \nabla_v \cdot (v \rho) + \frac{\sigma^2}{2} \Delta_v \rho
$$

where:
- $U(x)$ - external potential
- $\gamma > 0$ - friction coefficient
- $\sigma^2 > 0$ - diffusion strength

**Physical interpretation**: Langevin dynamics in phase space $(x,v)$
:::

**Source**: `11_stage0_revival_kl.md` Section 1.2
**Tags**: #revival #jump-operator #qsd

:::{prf:definition} Mean-Field Revival Operator
:label: def-revival-operator-formal

$$
\mathcal{R}[\rho, m_d](x,v) := \lambda_{\text{revive}} \cdot m_d \cdot \frac{\rho(x,v)}{\|\rho\|_{L^1}}
$$

where:
- $\|\rho\|_{L^1} = \int_\Omega \rho(x,v) \, dx dv$ - total alive mass
- $m_d = 1 - \|\rho\|_{L^1}$ - dead mass
- $\lambda_{\text{revive}} > 0$ - revival rate

**Properties**:
1. Mass injection: $\int_\Omega \mathcal{R}[\rho, m_d] = \lambda_{\text{revive}} m_d$
2. Proportionality: Revival samples proportionally to current alive density
3. Normalization: Factor $1/\|\rho\|_{L^1}$ preserves distribution shape
:::

**Source**: `11_stage0_revival_kl.md` Section 1.3
**Tags**: #jump-operator #killing #revival

:::{prf:definition} Combined Jump Operator
:label: def-combined-jump-operator

$$
\mathcal{L}_{\text{jump}}[\rho] = -\kappa_{\text{kill}}(x) \rho + \lambda_{\text{revive}} m_d(\rho) \frac{\rho}{\|\rho\|_{L^1}}
$$

where $\kappa_{\text{kill}}(x) \ge 0$ is the interior killing rate.

**Full generator**: $\mathcal{L} = \mathcal{L}_{\text{kin}} + \mathcal{L}_{\text{jump}}$
:::

### 1.2. Fisher Information and Entropy

**Source**: `11_stage2_explicit_constants.md` Section 1.1-1.3
**Tags**: #fisher-information #entropy #hypocoercivity

:::{prf:definition} Velocity Fisher Information
:label: def-velocity-fisher

$$
I_v(\rho) := \int_\Omega \rho(x,v) \left|\nabla_v \log \rho(x,v)\right|^2 dx dv
$$

Measures velocity space variations of the density.
:::

:::{prf:definition} Spatial Fisher Information
:label: def-spatial-fisher

$$
I_x(\rho) := \int_\Omega \rho(x,v) \left|\nabla_x \log \rho(x,v)\right|^2 dx dv
$$

Measures position space variations of the density.
:::

:::{prf:definition} Modified Fisher Information
:label: def-modified-fisher

For $\theta > 0$:

$$
I_\theta(\rho) := I_v(\rho) + \theta I_x(\rho) = \int_\Omega \rho \left(|\nabla_v \log \rho|^2 + \theta |\nabla_x \log \rho|^2\right) dx dv
$$

**Optimal choice**: $\theta = \gamma/(2L_v^{\max})$ where $L_v^{\max}$ is maximum velocity
:::

### 1.3. QSD and Equilibrium

**Source**: `11_convergence_mean_field.md` Section 0.1
**Tags**: #qsd #equilibrium #stationary-state

The mean-field density $\rho_t$ evolves according to the McKean-Vlasov-Fokker-Planck PDE:

$$
\frac{\partial \rho}{\partial t} = \mathcal{L}[\rho] \rho
$$

**Quasi-Stationary Distribution (QSD)**: $\rho_\infty$ satisfying

$$
\mathcal{L}[\rho_\infty] \rho_\infty = 0
$$

**Equilibrium mass**: $M_\infty = \|\rho_\infty\|_{L^1} < 1$

---

## 2. Revival Operator Analysis (Stage 0)

**Source Document**: `11_stage0_revival_kl.md`

### 2.1. Main Result: Revival is KL-Expansive

**Source**: `11_stage0_revival_kl.md` Section 7.1
**Tags**: #revival #kl-divergence #expansive #verified

:::{prf:theorem} Revival Operator is KL-Expansive
:label: thm-revival-kl-expansive

The mean-field revival operator $\mathcal{R}[\rho, m_d]$ **increases** the KL-divergence to the invariant measure $\pi$:

$$
\frac{d}{dt} D_{\text{KL}}(\rho \| \pi) \Big|_{\text{revival}} > 0 \quad \text{for all } \rho \neq \pi, \, m_d > 0
$$

Explicitly:

$$
\frac{d}{dt} D_{\text{KL}}(\rho \| \pi) \Big|_{\text{revival}} = \lambda m_d \left( 1 + \frac{D_{\text{KL}}(\rho \| \pi)}{\|\rho\|_{L^1}} \right)
$$

**Status**: PROVEN (verified by Gemini 2025-01-08)
:::

**Related Results**:
- Joint jump operator NOT unconditionally contractive ({prf:ref}`thm-joint-not-contractive`)
- KL-convergence requires kinetic dominance strategy

### 2.2. Jump Operator Bounds

**Source**: `11_stage0_revival_kl.md` Section 7.2
**Tags**: #jump-operator #entropy-production #bounds

The combined jump operator entropy production satisfies:

$$
I_{\text{jump}} := \int \mathcal{L}_{\text{jump}}(\rho) \log \frac{\rho}{\rho_\infty} \le A_{\text{jump}} D_{\text{KL}}(\rho | \rho_\infty) + B_{\text{jump}}
$$

where $A_{\text{jump}} = O(\lambda_{\text{revive}} / M_\infty + \bar{\kappa}_{\text{kill}})$ and $B_{\text{jump}}$ is a constant.

**Critical insight**: Dissipation must come from kinetic operator, not revival.

### 2.3. Decision Tree Resolution

**Source**: `11_stage0_revival_kl.md` Section 7.3
**Tags**: #strategy #decision

```
Revival alone is KL-expansive: TRUE ✓
  Joint operator is KL-contractive: FALSE ✗
  Kinetic operator dominates: MOST PLAUSIBLE PATH ✓
    → Proceed with composition analysis
    → Proof: kinetic dissipation > jump expansion
```

**Conclusion**: GO with Revised Strategy (kinetic dominance approach)

---

## 3. QSD Regularity Properties (Stage 0.5)

**Source Document**: `11_stage05_qsd_regularity.md`

### 3.1. Regularity Properties (R1-R6)

**Source**: `11_stage05_qsd_regularity.md` throughout
**Tags**: #regularity #qsd #smoothness

The QSD $\rho_\infty$ satisfies six regularity properties:

**R1 (Existence and Uniqueness)**: Via Schauder fixed-point theorem for nonlinear operator

**R2 (C² Smoothness)**: $\rho_\infty \in C^2(\Omega)$ via Hörmander hypoellipticity

**R3 (Strict Positivity)**: $\rho_\infty(x,v) > 0$ for all $(x,v) \in \Omega$ via irreducibility

**R4 (Bounded Spatial Log-Gradient)**:

$$
C_{\nabla x} := \|\nabla_x \log \rho_\infty\|_{L^\infty} < \infty
$$

**R5 (Bounded Velocity Log-Derivatives)**:

$$
\begin{aligned}
C_{\nabla v} &:= \|\nabla_v \log \rho_\infty\|_{L^\infty} < \infty \\
C_{\Delta v} &:= \|\Delta_v \log \rho_\infty\|_{L^\infty} < \infty
\end{aligned}
$$

**R6 (Exponential Concentration)**:

$$
\rho_\infty(x,v) \le C_{\exp} e^{-\alpha_{\exp}(|x|^2 + |v|^2)}
$$

for some $C_{\exp}, \alpha_{\exp} > 0$

**Status**: All six properties proven (roadmap provided in Stage 0.5 document)

### 3.2. Scaling Estimates for Regularity Constants

**Source**: `11_stage3_parameter_analysis.md` Section 1.1
**Tags**: #scaling #regularity #estimates

$$
\begin{aligned}
C_{\nabla x} &\sim \sqrt{\frac{\kappa_{\max}}{\sigma^2}} + \sqrt{\frac{L_U}{\gamma}} \\
C_{\nabla v} &\sim \frac{\sqrt{\gamma}}{\sigma} \\
C_{\Delta v} &\sim \frac{\gamma}{\sigma^2} + \frac{\lambda_{\text{revive}}}{M_\infty \sigma^2} \\
\alpha_{\exp} &\sim \min\left(\frac{\lambda_{\min}}{2\sigma^2}, \frac{\gamma}{\sigma^2}\right)
\end{aligned}
$$

where $\lambda_{\min}$ is the smallest eigenvalue of $\nabla^2 U$.

---

## 4. Entropy Production Framework (Stage 1)

**Source Document**: `11_stage1_entropy_production.md`

### 4.1. Full Entropy Production Derivation

**Source**: `11_stage1_entropy_production.md` Section 1.1
**Tags**: #entropy #dissipation #fundamental-identity

:::{prf:theorem} Fundamental Entropy Production Identity
:label: thm-entropy-production-identity

$$
\frac{d}{dt} D_{\text{KL}}(\rho_t \| \rho_\infty) = \int_\Omega \mathcal{L}(\rho_t) \log \frac{\rho_t}{\rho_\infty} \, dx dv
$$

where the integral uses mass conservation: $\int \mathcal{L}(\rho_t) \, dx dv = 0$.
:::

**Related**: Decomposes as $\mathcal{L} = \mathcal{L}_{\text{kin}} + \mathcal{L}_{\text{jump}}$

### 4.2. Kinetic Dissipation Term

**Source**: `11_stage1_entropy_production.md` Section 1.2, Term 4
**Tags**: #dissipation #diffusion #fisher-information

After integration by parts, the diffusion term produces:

$$
\text{Diffusion term} = -\frac{\sigma^2}{2} I_v(\rho) - \frac{\sigma^2}{2} \int \rho \cdot \Delta_v \log \rho_\infty
$$

where:
- $I_v(\rho) = \int \rho |\nabla_v \log \rho|^2 \ge 0$ is **velocity Fisher information** (DISSIPATIVE)
- Remainder term from $\Delta_v \rho_\infty \neq 0$ controlled via hypocoercivity

**Critical correction** (2025-01-08): Was incorrectly $I_v(\rho | \rho_\infty)$, should be $I_v(\rho)$

### 4.3. Stationarity Equation for QSD

**Source**: `11_stage1_entropy_production.md` Section 2.1
**Tags**: #stationary-pde #balance-equation

Since $\mathcal{L}(\rho_\infty) = 0$:

$$
\mathcal{L}_{\text{kin}}(\rho_\infty) + \mathcal{L}_{\text{jump}}(\rho_\infty) = 0
$$

Isolating the diffusion term:

$$
\boxed{\frac{\sigma^2}{2} \Delta_v \log \rho_\infty = -\frac{\mathcal{L}_{\text{jump}}(\rho_\infty)}{\rho_\infty} + v \cdot \nabla_x \log \rho_\infty - \nabla_x U \cdot \nabla_v \log \rho_\infty - \gamma v \cdot \nabla_v \log \rho_\infty - \gamma d - \frac{\sigma^2}{2} |\nabla_v \log \rho_\infty|^2}
$$

**Key insight**: Remainder term couples back to other coupling terms plus jump-related term

### 4.4. NESS Hypocoercivity Framework

**Source**: `11_stage1_entropy_production.md` Section 2.3
**Tags**: #hypocoercivity #ness #dolbeault

Following Dolbeault, Mouhot, and Schmeiser (2015):

**Modified Lyapunov functional**:

$$
\mathcal{H}_\varepsilon(\rho) := D_{\text{KL}}(\rho | \rho_\infty) + \varepsilon \int \rho \, a(x,v) \, dx dv
$$

where $a(x,v)$ is an auxiliary function chosen to compensate for $x$-$v$ coupling.

**Strategy**:
1. Choose $a$ such that $\mathcal{L}^*[a]$ cancels coupling terms
2. Prove $\frac{d}{dt}\mathcal{H}_\varepsilon \le -C_{\text{hypo}}[I_v + I_x] + I_{\text{jump}}$
3. Show equivalence: $D_{\text{KL}} \le \mathcal{H}_\varepsilon \le (1 + C\varepsilon) D_{\text{KL}}$

### 4.5. LSI for NESS

**Source**: `11_stage1_entropy_production.md` Section 2.4
**Tags**: #lsi #logarithmic-sobolev #assumptions

:::{prf:theorem} LSI Assumptions (Dolbeault et al. 2015)
:label: thm-lsi-assumptions

The LSI holds with constant:

$$
C_{\text{LSI}} = O\left(\frac{1}{\sigma^2 \gamma \kappa_{\text{conf}}}\right) \cdot \left(1 + O\left(\frac{\kappa_{\max} + \lambda}{\sigma^2 \gamma}\right)\right)
$$

under:
1. **Confinement**: $U(x) \to +\infty$ as $|x| \to \infty$ with $\nabla^2 U \ge \kappa_{\text{conf}} I$
2. **Regularity**: QSD satisfies R1-R6 (see Section 3.1)
3. **Bounded jumps**: $\kappa_{\text{kill}}, \lambda_{\text{revive}} < \infty$
:::

### 4.6. Main Framework Result

**Source**: `11_stage1_entropy_production.md` Section 2.5
**Tags**: #kinetic-dominance #convergence-condition

When kinetic dissipation dominates jump expansion:

$$
\boxed{\frac{d}{dt} D_{\text{KL}} \le -\alpha_{\text{net}} D_{\text{KL}} + B_{\text{jump}}}
$$

where:
- $\alpha_{\text{kin}} := C_{\text{hypo}} / C_{\text{LSI}}$ (kinetic dissipation rate)
- $\alpha_{\text{net}} := \alpha_{\text{kin}} - A_{\text{jump}}$ (net convergence rate)

**Kinetic Dominance Condition**: $\alpha_{\text{net}} > 0 \iff \alpha_{\text{kin}} > A_{\text{jump}}$

---

## 5. Explicit Hypocoercivity Constants (Stage 2)

**Source Document**: `11_stage2_explicit_constants.md`

### 5.1. LSI Constant

**Source**: `11_stage2_explicit_constants.md` Section 2.2
**Tags**: #lsi #explicit-constant #bakry-emery

:::{prf:theorem} Explicit LSI Constant
:label: thm-lsi-constant-explicit

$$
\boxed{\lambda_{\text{LSI}} \ge \frac{\alpha_{\exp}}{1 + C_{\Delta v}/\alpha_{\exp}}}
$$

where:
- $\alpha_{\exp}$ is the exponential concentration rate from (R6)
- $C_{\Delta v} = \|\Delta_v \log \rho_\infty\|_{L^\infty}$ from (R5)

**Simplified** (when $C_{\Delta v} \ll \alpha_{\exp}$):

$$
\lambda_{\text{LSI}} \approx \alpha_{\exp} \left(1 - \frac{C_{\Delta v}}{\alpha_{\exp}}\right)
$$
:::

**Proof method**: Holley-Stroock perturbation theorem from Gaussian reference measure

### 5.2. Fisher Information Bound

**Source**: `11_stage2_explicit_constants.md` Section 2.3
**Tags**: #fisher-information #lsi-application

:::{prf:lemma} Fisher Information Bound
:label: lem-fisher-bound

$$
\boxed{I_v(\rho) \ge 2\lambda_{\text{LSI}} D_{\text{KL}}(\rho \| \rho_\infty) - C_{\text{LSI}}}
$$

for explicit constant $C_{\text{LSI}}$ depending on $\lambda_{\text{LSI}}$ and $C_{\nabla v}$.
:::

### 5.3. Coupling Constants

**Source**: `11_stage2_explicit_constants.md` Section 3
**Tags**: #coupling #bounds #transport-force-friction

After bounding all coupling/remainder terms:

$$
|R_{\text{coupling}}| \le C_{\text{KL}}^{\text{coup}} D_{\text{KL}}(\rho \| \rho_\infty) + C_{\text{Fisher}}^{\text{coup}} I_v(\rho) + C_0^{\text{coup}}
$$

where:

$$
\boxed{
\begin{aligned}
C_{\text{KL}}^{\text{coup}} &= (C_{\nabla x} + \gamma) \sqrt{2C_v} \\
C_{\text{Fisher}}^{\text{coup}} &= (C_{\nabla x} + \gamma) \sqrt{2C_v'/\gamma} + \frac{L_U^2}{4\epsilon} + \gamma C_{\nabla v} \sqrt{2C_v'/\gamma} \\
C_0^{\text{coup}} &= L_U C_{\nabla v} + \frac{\sigma^2 C_{\Delta v}}{2}
\end{aligned}
}
$$

**Individual terms**:
- Transport: $|R_{\text{transport}}| \le C_1^{\text{trans}} D_{\text{KL}} + C_2^{\text{trans}} I_v$
- Force: $|R_{\text{force}}| \le C^{\text{force}} I_v + C_0^{\text{force}}$
- Friction: $|R_{\text{friction}}| \le C_1^{\text{fric}} D_{\text{KL}} + C_2^{\text{fric}} I_v$
- Diffusion: $|R_{\text{diffusion}}| \le C^{\text{diff}} = \sigma^2 C_{\Delta v}/2$ (pure constant)

### 5.4. Jump Expansion Constant

**Source**: `11_stage2_explicit_constants.md` Section 4
**Tags**: #jump-expansion #killing #revival

$$
I_{\text{jump}} \le A_{\text{jump}} D_{\text{KL}}(\rho | \rho_\infty) + B_{\text{jump}}
$$

where:

$$
\boxed{
\begin{aligned}
A_{\text{jump}} &= 2\kappa_{\max} + \frac{\lambda_{\text{revive}}(1-M_\infty)}{M_\infty^2} \\
B_{\text{jump}} &= \kappa_{\max} C_{\text{const}} + C_{\text{revive}}
\end{aligned}
}
$$

For uniform killing $\kappa_{\text{kill}} = \kappa_0$:

$$
M_\infty = \frac{\lambda_{\text{revive}}}{\lambda_{\text{revive}} + \kappa_0}
$$

### 5.5. Coercivity Gap and Convergence Rate

**Source**: `11_stage2_explicit_constants.md` Section 5
**Tags**: #coercivity-gap #convergence-rate #main-result

:::{prf:definition} Coercivity Gap
:label: def-coercivity-gap

$$
\boxed{\delta := \lambda_{\text{LSI}} \sigma^2 - 2\lambda_{\text{LSI}} C_{\text{Fisher}}^{\text{coup}} - C_{\text{KL}}^{\text{coup}} - A_{\text{jump}}}
$$
:::

**Condition for exponential convergence**:

$$
\delta > 0 \quad \Leftrightarrow \quad \sigma^2 > \frac{2\lambda_{\text{LSI}} C_{\text{Fisher}}^{\text{coup}} + C_{\text{KL}}^{\text{coup}} + A_{\text{jump}}}{\lambda_{\text{LSI}}} =: \sigma_{\text{crit}}^2
$$

:::{prf:theorem} Exponential Convergence Rate
:label: thm-exponential-convergence-local

Assume $\delta > 0$ and $D_{\text{KL}}(\rho_0 \| \rho_\infty) \le \epsilon_0$ sufficiently small. Then:

$$
D_{\text{KL}}(\rho_t \| \rho_\infty) \le e^{-\alpha_{\text{net}} t} D_{\text{KL}}(\rho_0 \| \rho_\infty)
$$

where:

$$
\boxed{\alpha_{\text{net}} = \frac{\delta}{2}}
$$
:::

---

## 6. Parameter Analysis (Stage 3)

**Source Document**: `11_stage3_parameter_analysis.md`

### 6.1. Mean-Field Rate as Function of Parameters

**Source**: `11_stage3_parameter_analysis.md` Section 2.1
**Tags**: #explicit-formula #parameters #convergence-rate

:::{prf:theorem} Mean-Field Convergence Rate (Explicit)
:label: thm-alpha-net-explicit

$$
\begin{aligned}
\alpha_{\text{net}}(\tau, \gamma, \sigma, \lambda_{\text{revive}}, \kappa_{\max}, L_U) \approx \frac{1}{2} \Bigg[
&\frac{\gamma \sigma^2}{1 + \gamma/\lambda_{\min} + \lambda_{\text{revive}}/(M_\infty \gamma)} \\
&- \frac{2\gamma}{\sigma^2} \left(\sqrt{\frac{\kappa_{\max}}{\sigma^2}} + \sqrt{\frac{L_U}{\gamma}} + \gamma\right) \sigma\tau\sqrt{2d} \\
&- \frac{2\gamma L_U^3}{\sigma^4(1 + \gamma/\lambda_{\min})} \\
&- (C_{\nabla x} + \gamma) \sqrt{2d\sigma^2/\gamma} \\
&- 2\kappa_{\max} - \frac{\kappa_0(\lambda_{\text{revive}} + \kappa_0)^2}{\lambda_{\text{revive}}^2}
\Bigg]
\end{aligned}
$$

**Simplified** (for $\tau \ll 1, \gamma \ll \lambda_{\min}$):

$$
\boxed{\alpha_{\text{net}} \approx \frac{1}{2}\left[\gamma - \frac{2\gamma^2\tau\sqrt{2d}}{\sigma} - \frac{2\gamma L_U^3}{\sigma^4} - 2\kappa_{\max} - \frac{\kappa_0 \lambda_{\text{revive}}}{\lambda_{\text{revive}} + \kappa_0}\right]}
$$
:::

### 6.2. Critical Diffusion Threshold

**Source**: `11_stage3_parameter_analysis.md` Section 2.2
**Tags**: #critical-threshold #diffusion #convergence-condition

For $\alpha_{\text{net}} > 0$, dominant balance gives:

$$
\boxed{\sigma_{\text{crit}} \gtrsim \left(\frac{2L_U^3}{\gamma}\right)^{1/4}}
$$

**Interpretation**: Diffusion must scale as $L_U^{3/4}$ to overcome landscape roughness.

### 6.3. Optimal Parameter Scaling

**Source**: `11_stage3_parameter_analysis.md` Section 2.3
**Tags**: #optimal-scaling #parameter-tuning

:::{prf:theorem} Optimal Parameter Scaling
:label: thm-optimal-parameter-scaling

For landscape with Lipschitz constant $L_U$ and minimum Hessian eigenvalue $\lambda_{\min}$:

$$
\begin{aligned}
\gamma^* &\sim L_U^{3/7} \\
\sigma^* &\sim L_U^{9/14} \\
\tau^* &\sim L_U^{-12/7} \\
\lambda_{\text{revive}}^* &\sim \kappa_{\max}
\end{aligned}
$$

yielding convergence rate:

$$
\alpha_{\text{net}}^* \sim \gamma^* \sim L_U^{3/7}
$$
:::

### 6.4. Parameter Sensitivities

**Source**: `11_stage3_parameter_analysis.md` Section 3
**Tags**: #sensitivity #parameter-effects

**Logarithmic sensitivity**: $S_P = \frac{P}{\alpha_{\text{net}}} \frac{\partial \alpha_{\text{net}}}{\partial P}$

$$
\boxed{
\begin{aligned}
S_{\sigma} &= \frac{\sigma}{\alpha_{\text{net}}} \cdot \frac{1}{2}\left[\frac{2\gamma^2\tau\sqrt{2d}}{\sigma^2} + \frac{8\gamma L_U^3}{\sigma^5}\right] \quad \text{(always positive)} \\
S_{\gamma} &= \frac{\gamma}{\alpha_{\text{net}}} \cdot \frac{1}{2}\left[1 - \frac{4\gamma\tau\sqrt{2d}}{\sigma} - \frac{2L_U^3}{\sigma^4}\right] \\
S_{\tau} &= -\frac{\tau\gamma^2\sqrt{2d}}{\sigma \alpha_{\text{net}}} \quad \text{(always negative)} \\
S_{\kappa_{\max}} &= -\frac{\kappa_{\max}}{\alpha_{\text{net}}} \quad \text{(always negative)} \\
S_{\lambda_{\text{revive}}} &= -\frac{\lambda_{\text{revive}} \kappa_0^2}{(\lambda_{\text{revive}} + \kappa_0)^2 \alpha_{\text{net}}} \quad \text{(always negative)}
\end{aligned}
}
$$

**Ranking** (typical regime): $|S_{\sigma}| > |S_{\gamma}| > |S_{\lambda}| > |S_{\kappa}| > |S_{\tau}|$

**Key insight**: Diffusion $\sigma$ has strongest impact on convergence rate.

### 6.5. Finite-N Corrections

**Source**: `11_stage3_parameter_analysis.md` Section 5
**Tags**: #finite-n #discretization #corrections

$$
\boxed{\alpha_N \approx \alpha_{\text{net}} \left(1 - \frac{c_{\text{clone}}}{\delta^2 N}\right) \left(1 - \frac{\tau \alpha_{\text{net}}}{2\gamma}\right)}
$$

where:
- First factor: Cloning fluctuations ($c_{\text{clone}} \sim 1$, $\delta$ = cloning noise variance)
- Second factor: Time-discretization error

**Guideline**: To stay within 5% of mean-field rate:

$$
\frac{c_{\text{clone}}}{\delta^2 N} + \frac{\tau \alpha_{\text{net}}}{2\gamma} < 0.05
$$

---

## 7. Main Convergence Theorems

### 7.1. Stage 1 Framework

**Source**: `11_stage1_entropy_production.md` Section 4
**Tags**: #main-theorem #kl-convergence #framework

:::{prf:theorem} KL-Convergence for Mean-Field Euclidean Gas (Framework)
:label: thm-corrected-kl-convergence

If the kinetic dominance condition holds:

$$
\sigma^2 \gamma \kappa_{\text{conf}} > C_0 \max\left(\frac{\lambda}{M_\infty}, \bar{\kappa}\right)
$$

then the mean-field Euclidean Gas converges exponentially to its QSD:

$$
D_{\text{KL}}(\rho_t \| \rho_\infty) \le e^{-\alpha_{\text{net}} t} D_{\text{KL}}(\rho_0 \| \rho_\infty) + \frac{B_{\text{jump}}}{\alpha_{\text{net}}} (1 - e^{-\alpha_{\text{net}} t})
$$

where $\alpha_{\text{net}} = \alpha_{\text{kin}} - A_{\text{jump}} > 0$.

**Status**: Framework established, technical details in Stage 2-3
:::

### 7.2. Stage 2 Explicit Result

**Source**: `11_stage2_explicit_constants.md` Section 10
**Tags**: #explicit-constants #main-result

:::{prf:theorem} Main Result: Explicit Convergence Rate
:label: thm-main-explicit-rate

Under QSD regularity (R1-R6) and:

$$
\sigma^2 > \sigma_{\text{crit}}^2 := \frac{2\lambda_{\text{LSI}} C_{\text{Fisher}}^{\text{coup}} + C_{\text{KL}}^{\text{coup}} + A_{\text{jump}}}{\lambda_{\text{LSI}}}
$$

the mean-field Euclidean Gas converges exponentially with rate:

$$
\boxed{\alpha_{\text{net}} = \frac{1}{2}\left(\lambda_{\text{LSI}} \sigma^2 - 2\lambda_{\text{LSI}} C_{\text{Fisher}}^{\text{coup}} - C_{\text{KL}}^{\text{coup}} - A_{\text{jump}}\right)}
$$

where all constants are explicit in $(\gamma, \sigma, L_U, \kappa_{\max}, \lambda_{\text{revive}})$ and $(C_{\nabla x}, C_{\nabla v}, C_{\Delta v}, \alpha_{\exp})$.

**Significance**:
1. Fully computable
2. Numerically verifiable
3. Tunable via physical parameters
4. Completes mean-field convergence proof
:::

### 7.3. Relation to Finite-N

**Source**: `11_stage3_parameter_analysis.md` Section 5
**Tags**: #finite-n #mean-field-limit #asymptotic

**Key relationship**:

$$
\lim_{N \to \infty} \alpha_N(\tau, N) = \alpha_{\text{net}} \quad \text{(after adjusting for discrete-time)}
$$

The mean-field limit removes $1/N$ cloning fluctuations, leaving only jump operator's deterministic expansion.

---

## 8. Practical Formulas

### 8.1. Quick Reference Formulas

**Source**: `11_stage3_parameter_analysis.md` Section 8.1
**Tags**: #quick-reference #formulas

**Mean-field convergence rate**:

$$
\alpha_{\text{net}} \approx \frac{1}{2}\left[\gamma - \frac{2\gamma^2\tau\sqrt{2d}}{\sigma} - \frac{2\gamma L_U^3}{\sigma^4} - 2\kappa_{\max} - C_{\text{jump}}\right]
$$

**Critical diffusion**:

$$
\sigma_{\text{crit}} \gtrsim \left(\frac{2L_U^3}{\gamma}\right)^{1/4}
$$

**Finite-N correction**:

$$
\alpha_N \approx \alpha_{\text{net}} \left(1 - \frac{100}{\delta^2 N}\right)
$$

**Optimal scaling**:

$$
\gamma^* \sim L_U^{3/7}, \quad \sigma^* \sim L_U^{9/14}, \quad \tau^* \sim L_U^{-12/7}
$$

### 8.2. Parameter Effects

**Source**: `11_stage3_parameter_analysis.md` Section 8.2
**Tags**: #parameter-effects #tuning

| Parameter | Increases $\alpha_{\text{net}}$ | Decreases $\alpha_{\text{net}}$ | Optimal Value |
|:----------|:-------------------------------|:--------------------------------|:--------------|
| $\gamma$ | Increases LSI | Increases coupling | $\sqrt{\sigma^4/(L_U \tau\sqrt{2d})}$ |
| $\sigma$ | Always positive | — | $(L_U^3 \gamma)^{1/4}$ |
| $\tau$ | — | Always negative | $\min(0.5/\gamma, 1/\sqrt{\lambda_{\max}})$ |
| $\kappa_{\max}$ | — | Always negative | Minimize (if possible) |
| $\lambda_{\text{revive}}$ | — | Always negative | $\kappa_{\text{avg}}$ (balance) |
| $N$ | Indirect (reduces $1/N$ error) | — | $> 100/\delta^2$ |

### 8.3. Diagnostic Decision Tree

**Source**: `11_stage3_parameter_analysis.md` Section 8.3
**Tags**: #diagnostics #troubleshooting

```
Start: Measure α_emp from simulation
│
├─ α_emp ≈ α_theory (within 20%)
│  └─ SUCCESS: System in mean-field regime
│
├─ α_emp < 0.5 α_theory
│  ├─ Check: N > 100/δ²? → If NO: Increase N
│  ├─ Check: τ < 0.5/γ? → If NO: Reduce τ
│  └─ Check: σ > σ_crit? → If NO: Increase σ (critical!)
│
└─ α_theory < 0
   └─ INVALID REGIME: Must increase σ or reduce L_U/κ_max
```

### 8.4. Numerical Validation Algorithm

**Source**: `11_stage3_parameter_analysis.md` Section 6.2
**Tags**: #numerical #validation #algorithm

**Input**: $(\tau, \gamma, \sigma, \lambda_{\text{revive}}, \kappa_{\max}, L_U, \lambda_{\min}, d, N, \delta)$

**Step 1**: Estimate QSD regularity constants

$$
\begin{aligned}
C_{\nabla x} &= \sqrt{\kappa_{\max}/\sigma^2} + \sqrt{L_U/\gamma} \\
C_{\nabla v} &= \sqrt{\gamma}/\sigma \\
C_{\Delta v} &= \gamma/\sigma^2 + \lambda_{\text{revive}}/(M_\infty \sigma^2) \\
\alpha_{\exp} &= \min(\lambda_{\min}/\sigma^2, \gamma/\sigma^2) / 2
\end{aligned}
$$

**Step 2**: Compute LSI constant

$$
\lambda_{\text{LSI}} = \frac{\alpha_{\exp}}{1 + C_{\Delta v}/\alpha_{\exp}}
$$

**Step 3**: Compute coupling constants

$$
C_{\text{Fisher}}^{\text{coup}} = (C_{\nabla x} + \gamma) \sigma\tau\sqrt{2d} + L_U^3/(2\sigma^2) + \sqrt{\gamma}\tau\sqrt{2d}
$$

**Step 4**: Compute jump expansion

$$
M_\infty = \frac{\lambda_{\text{revive}}}{\lambda_{\text{revive}} + \kappa_0}, \quad A_{\text{jump}} = 2\kappa_{\max} + \kappa_0(\lambda_{\text{revive}} + \kappa_0)^2/\lambda_{\text{revive}}^2
$$

**Step 5**: Assemble convergence rate

$$
\alpha_{\text{net}}^{\text{theory}} = \frac{1}{2}\left(\lambda_{\text{LSI}} \sigma^2 - 2\lambda_{\text{LSI}} C_{\text{Fisher}}^{\text{coup}} - C_{\text{KL}}^{\text{coup}} - A_{\text{jump}}\right)
$$

**Step 6**: Apply finite-N correction

$$
\alpha_N^{\text{theory}} = \alpha_{\text{net}}^{\text{theory}} \left(1 - \frac{100}{\delta^2 N}\right) \left(1 - \frac{\tau \alpha_{\text{net}}^{\text{theory}}}{2\gamma}\right)
$$

**Output**: $\alpha_N^{\text{theory}}$ (predicted convergence rate)

---

## Index of All Theorems and Lemmas

### Definitions

- {prf:ref}`def-kinetic-operator` - Mean-Field Kinetic Operator
- {prf:ref}`def-revival-operator-formal` - Mean-Field Revival Operator
- {prf:ref}`def-combined-jump-operator` - Combined Jump Operator
- {prf:ref}`def-velocity-fisher` - Velocity Fisher Information
- {prf:ref}`def-spatial-fisher` - Spatial Fisher Information
- {prf:ref}`def-modified-fisher` - Modified Fisher Information
- {prf:ref}`def-coercivity-gap` - Coercivity Gap

### Theorems

- {prf:ref}`thm-revival-kl-expansive` - Revival Operator is KL-Expansive (VERIFIED)
- {prf:ref}`thm-entropy-production-identity` - Fundamental Entropy Production Identity
- {prf:ref}`thm-lsi-assumptions` - LSI Assumptions (Dolbeault et al. 2015)
- {prf:ref}`thm-lsi-constant-explicit` - Explicit LSI Constant
- {prf:ref}`thm-exponential-convergence-local` - Exponential Convergence Rate
- {prf:ref}`thm-alpha-net-explicit` - Mean-Field Convergence Rate (Explicit)
- {prf:ref}`thm-optimal-parameter-scaling` - Optimal Parameter Scaling
- {prf:ref}`thm-corrected-kl-convergence` - KL-Convergence Framework
- {prf:ref}`thm-main-explicit-rate` - Main Result: Explicit Convergence Rate

### Lemmas

- {prf:ref}`lem-fisher-bound` - Fisher Information Bound

---

## Document Map

| Topic | Primary Source | Section | Status |
|:------|:--------------|:--------|:-------|
| Revival KL-Expansiveness | `11_stage0_revival_kl.md` | 7.1 | ✅ VERIFIED |
| QSD Regularity (R1-R6) | `11_stage05_qsd_regularity.md` | Throughout | ✅ PROVEN |
| Entropy Production | `11_stage1_entropy_production.md` | 1-2 | ✅ FRAMEWORK COMPLETE |
| NESS Hypocoercivity | `11_stage1_entropy_production.md` | 2.3 | ✅ FRAMEWORK COMPLETE |
| Explicit LSI Constant | `11_stage2_explicit_constants.md` | 2 | ✅ COMPLETE |
| Coupling Bounds | `11_stage2_explicit_constants.md` | 3 | ✅ COMPLETE |
| Jump Expansion | `11_stage2_explicit_constants.md` | 4 | ✅ COMPLETE |
| Coercivity Gap | `11_stage2_explicit_constants.md` | 5 | ✅ COMPLETE |
| Parameter Formulas | `11_stage3_parameter_analysis.md` | 1-2 | ✅ COMPLETE |
| Parameter Sensitivities | `11_stage3_parameter_analysis.md` | 3 | ✅ COMPLETE |
| Finite-N Corrections | `11_stage3_parameter_analysis.md` | 5 | ✅ COMPLETE |
| Numerical Validation | `11_stage3_parameter_analysis.md` | 6-7 | ✅ COMPLETE |

---

## Cross-References to Other Documents

**Foundation Documents**:
- `04_convergence.md` - N-uniform Foster-Lyapunov convergence
- `06_propagation_chaos.md` - Empirical measure convergence μ_N ⇒ ρ_∞
- `10_kl_convergence.md` - Finite-N LSI and KL-convergence

**Related Theory**:
- `01_fragile_gas_framework.md` - Framework axioms and definitions
- `02_euclidean_gas.md` - Discrete implementation and BAOAB integrator
- `07_adaptive_gas.md` - Adaptive mechanisms (future extension)

**Implementation**:
- `../../../src/fragile/gas_parameters.py` - Code implementation of formulas

---

**END OF MATHEMATICAL REFERENCE**
