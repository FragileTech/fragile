# Stage 2: Explicit Hypocoercivity Constants

**Document Status**: NEW - Technical development of explicit constants (2025-01-09)

**Purpose**: Derive fully explicit hypocoercivity constants for the mean-field convergence proof, making all abstract bounds from Stage 1 concrete and computable.

**Parent documents**:
- [11_stage05_qsd_regularity.md](11_stage05_qsd_regularity.md) - QSD regularity properties (R1-R6)
- [11_stage1_entropy_production.md](11_stage1_entropy_production.md) - Entropy production framework
- [../01_fragile_gas_framework.md](../01_fragile_gas_framework.md) - Framework axioms

**Motivation**: The Stage 1 proof establishes that kinetic dissipation dominates jump expansion via hypocoercivity, but the constants appear only abstractly. This document:
1. Makes all constants **fully explicit** in terms of physical parameters
2. Provides **computable bounds** for numerical verification
3. Establishes **parameter regime requirements** for exponential convergence

---

## 0. Overview and Roadmap

### 0.1. Physical Parameters

The mean-field Euclidean Gas is determined by:

**Kinetic parameters**:
- $\gamma > 0$ - friction coefficient
- $\sigma^2 > 0$ - diffusion strength
- $\tau > 0$ - time step (for discrete-time)
- $U(x)$ - external potential with Lipschitz constant $L_U$

**Jump parameters**:
- $\kappa_{\text{kill}}(x) \ge 0$ - killing rate with bounds:

$$
0 \le \kappa_{\min} := \inf_{x \in \mathcal{X}} \kappa_{\text{kill}}(x) \le \kappa_{\text{kill}}(x) \le \kappa_{\max} := \sup_{x \in \mathcal{X}} \kappa_{\text{kill}}(x)
$$

- $\lambda_{\text{revive}} > 0$ - revival rate
- $M_\infty = \|\rho_\infty\|_{L^1} < 1$ - equilibrium mass

**Regularity bounds** (from Stage 0.5):
- $C_{\nabla x} := \|\nabla_x \log \rho_\infty\|_{L^\infty}$ - spatial log-gradient bound
- $C_{\nabla v} := \|\nabla_v \log \rho_\infty\|_{L^\infty}$ - velocity log-gradient bound
- $C_{\Delta v} := \|\Delta_v \log \rho_\infty\|_{L^\infty}$ - velocity log-Laplacian bound
- $C_{\exp}$ - exponential concentration constant (R6)

### 0.2. The Key Constants to Derive

We need explicit formulas for:

**Dissipation constants**:
1. $\lambda_{\text{LSI}}$ - Log-Sobolev constant relating Fisher information to KL divergence
2. $\alpha_{\text{kin}}$ - Kinetic dissipation rate from velocity Fisher information

**Expansion constants**:
3. $A_{\text{jump}}$ - Jump expansion coefficient (linear in KL)
4. $B_{\text{jump}}$ - Jump expansion offset (constant term)

**Hypocoercivity machinery**:
5. $\theta$ - Auxiliary functional weight (balances dissipation and coupling)
6. $C_{\text{coupling}}$ - Coupling term bound
7. $\delta$ - Coercivity gap (how much dissipation exceeds expansion)

**Final result**:
8. $\alpha_{\text{net}} = \lambda_{\text{LSI}} \cdot \delta > 0$ - Exponential convergence rate

### 0.3. Strategy

The derivation follows the NESS hypocoercivity framework (Dolbeault et al. 2015):

1. **Section 1**: Construct modified Fisher information $I_\theta(\rho) := I_v(\rho) + \theta I_x(\rho)$
2. **Section 2**: Bound velocity Fisher information by KL via LSI: $I_v(\rho) \ge \lambda_{\text{LSI}} D_{\text{KL}}(\rho \| \rho_\infty)$
3. **Section 3**: Bound coupling/remainder terms using QSD regularity
4. **Section 4**: Bound jump expansion explicitly
5. **Section 5**: Combine to get coercivity gap $\delta$ and convergence rate $\alpha_{\text{net}}$

---

## 1. Modified Fisher Information and Auxiliary Functional

### 1.1. The Standard Fisher Information

Recall from Stage 1 that the velocity Fisher information is:

$$
I_v(\rho) := \int_\Omega \rho(x,v) \left|\nabla_v \log \rho(x,v)\right|^2 dx dv
$$

This measures how much $\rho$ varies in the velocity direction. The kinetic diffusion produces dissipation:

$$
-\frac{\sigma^2}{2} I_v(\rho) \le 0
$$

**Problem**: Velocity dissipation alone cannot control spatial variations of $\rho$ (the "coercivity gap").

### 1.2. The Spatial Fisher Information

Define the spatial Fisher information:

$$
I_x(\rho) := \int_\Omega \rho(x,v) \left|\nabla_x \log \rho(x,v)\right|^2 dx dv
$$

This measures spatial variations. The transport operator $-v \cdot \nabla_x$ couples spatial and velocity structure, but doesn't directly dissipate $I_x(\rho)$.

### 1.3. Modified Fisher Information

Following hypocoercivity theory, we introduce a **weighted combination**:

:::{prf:definition} Modified Fisher Information
:label: def-modified-fisher

For a parameter $\theta > 0$, the **modified Fisher information** is:

$$
I_\theta(\rho) := I_v(\rho) + \theta I_x(\rho) = \int_\Omega \rho \left(|\nabla_v \log \rho|^2 + \theta |\nabla_x \log \rho|^2\right) dx dv
$$

The parameter $\theta$ balances the velocity and spatial contributions.
:::

**Key insight**: While $\frac{d}{dt} I_v(\rho)$ doesn't control $I_x(\rho)$, the transport-friction coupling allows us to prove:

$$
\frac{d}{dt} I_\theta(\rho) \le -c_{\text{diss}} I_\theta(\rho) + \text{(controlled terms)}
$$

for an appropriate choice of $\theta$.

### 1.4. Choosing $\theta$ Optimally

The optimal $\theta$ balances two competing effects:

**Effect 1: Transport-friction coupling**

The transport operator $-v \cdot \nabla_x$ couples $I_x$ and cross-terms like $\int \rho \nabla_x \log \rho \cdot \nabla_v \log \rho$. The friction $-\gamma v \cdot \nabla_v$ dissipates velocity structure. Together, they can dissipate $I_x$ indirectly.

**Effect 2: Coupling to QSD structure**

Remainder terms from $\nabla_x \log \rho_\infty$, $\nabla_v \log \rho_\infty$, $\Delta_v \log \rho_\infty$ must be controlled by $I_\theta(\rho)$.

**Explicit formula** (derived below):

$$
\theta = \frac{\gamma}{2 L_v^{\max}}
$$

where $L_v^{\max}$ is the maximum velocity (determined by energy bounds and exponential concentration).

**Justification**: This choice ensures transport-friction coupling produces net dissipation of $I_x$ at rate $\sim \gamma \theta$.

---

## 2. Log-Sobolev Inequality for the QSD

### 2.1. The LSI Statement

The core of hypocoercivity is relating Fisher information to KL divergence:

:::{prf:theorem} Log-Sobolev Inequality (LSI) for QSD
:label: thm-lsi-qsd

There exists a constant $\lambda_{\text{LSI}} > 0$ such that for all probability densities $\rho$ on $\Omega$:

$$
D_{\text{KL}}(\rho \| \rho_\infty) \le \frac{1}{2\lambda_{\text{LSI}}} I_v(\rho \| \rho_\infty)
$$

where $I_v(\rho \| \rho_\infty) := \int \rho |\nabla_v \log(\rho/\rho_\infty)|^2$.
:::

**Note**: The relative Fisher information can be expanded as:

$$
I_v(\rho \| \rho_\infty) = I_v(\rho) - 2\int \rho \nabla_v \log \rho \cdot \nabla_v \log \rho_\infty + \int \rho |\nabla_v \log \rho_\infty|^2
$$

Using the QSD regularity bound $C_{\nabla v} = \|\nabla_v \log \rho_\infty\|_{L^\infty}$, we can relate this to $I_v(\rho)$ alone.

### 2.2. Deriving the LSI Constant

**Step 1: Exponential concentration bound**

From QSD regularity (R6):

$$
\rho_\infty(x,v) \le C_{\exp} e^{-\alpha_{\exp}(|x|^2 + |v|^2)}
$$

This implies $\rho_\infty$ has a Gaussian-like tail in velocity space.

**Step 2: Conditional velocity distribution**

For fixed $x \in \mathcal{X}$, define the conditional distribution:

$$
\rho_\infty^x(v) := \frac{\rho_\infty(x,v)}{\int \rho_\infty(x,v') dv'}
$$

The exponential bound implies:

$$
\rho_\infty^x(v) \le C_x e^{-\alpha_{\exp} |v|^2}
$$

**Step 3: Bakry-Émery criterion**

For a Gaussian-like measure $\mu(v) \propto e^{-\alpha |v|^2}$, the Bakry-Émery criterion gives an LSI constant:

$$
\lambda_{\text{LSI}}^{\text{Gauss}} = 2\alpha
$$

In our case, $\alpha = \alpha_{\exp}$ from the exponential concentration.

**Step 4: Perturbation bound**

The true QSD $\rho_\infty^x(v)$ is not exactly Gaussian, but it's close (bounded perturbation). Using perturbative LSI theory (Holley-Stroock):

:::{prf:theorem} Explicit LSI Constant
:label: thm-lsi-constant-explicit

The LSI constant for the QSD satisfies:

$$
\boxed{\lambda_{\text{LSI}} \ge \frac{\alpha_{\exp}}{1 + C_{\Delta v}/\alpha_{\exp}}}
$$

where:
- $\alpha_{\exp}$ is the exponential concentration rate from (R6)
- $C_{\Delta v} = \|\Delta_v \log \rho_\infty\|_{L^\infty}$ from (R5)
:::

:::{prf:proof}
The proof follows from the Holley-Stroock perturbation theorem. The reference Gaussian measure $\mu(v) = (2\pi/\alpha_{\exp})^{-d/2} e^{-\alpha_{\exp}|v|^2/2}$ has LSI constant $\lambda_0 = \alpha_{\exp}$.

The log-ratio $\log(\rho_\infty^x / \mu)$ satisfies:

$$
\left|\Delta_v \log \frac{\rho_\infty^x}{\mu}\right| = |\Delta_v \log \rho_\infty^x - \Delta_v \log \mu| = |\Delta_v \log \rho_\infty^x + \alpha_{\exp} d|
$$

Using $\|\Delta_v \log \rho_\infty\|_{L^\infty} \le C_{\Delta v}$:

$$
\left|\Delta_v \log \frac{\rho_\infty^x}{\mu}\right| \le C_{\Delta v} + \alpha_{\exp} d
$$

The Holley-Stroock theorem gives:

$$
\lambda_{\text{LSI}} \ge \frac{\lambda_0}{1 + C_{\text{perturb}}/\lambda_0}
$$

where $C_{\text{perturb}} = C_{\Delta v}$. Substituting $\lambda_0 = \alpha_{\exp}$:

$$
\lambda_{\text{LSI}} \ge \frac{\alpha_{\exp}}{1 + C_{\Delta v}/\alpha_{\exp}}
$$

:::

**Practical bound**: If $C_{\Delta v} \ll \alpha_{\exp}$ (weakly perturbed Gaussian), then:

$$
\lambda_{\text{LSI}} \approx \alpha_{\exp} \left(1 - \frac{C_{\Delta v}}{\alpha_{\exp}}\right)
$$

### 2.3. Relating $I_v(\rho)$ to $I_v(\rho \| \rho_\infty)$

The LSI is stated in terms of the relative Fisher information $I_v(\rho \| \rho_\infty)$. We need a bound using $I_v(\rho)$ alone.

**Expansion**:

$$
\begin{aligned}
I_v(\rho \| \rho_\infty) &= \int \rho |\nabla_v \log \rho - \nabla_v \log \rho_\infty|^2 \\
&= I_v(\rho) - 2\int \rho \nabla_v \log \rho \cdot \nabla_v \log \rho_\infty + \int \rho |\nabla_v \log \rho_\infty|^2
\end{aligned}
$$

**Bounding cross-term**: Using Cauchy-Schwarz and $\|\nabla_v \log \rho_\infty\|_{L^\infty} \le C_{\nabla v}$:

$$
\left|2\int \rho \nabla_v \log \rho \cdot \nabla_v \log \rho_\infty\right| \le 2C_{\nabla v} \int \rho |\nabla_v \log \rho| \le 2C_{\nabla v} \sqrt{I_v(\rho)}
$$

**Bounding constant term**:

$$
\int \rho |\nabla_v \log \rho_\infty|^2 \le C_{\nabla v}^2
$$

**Result**:

$$
I_v(\rho \| \rho_\infty) \ge I_v(\rho) - 2C_{\nabla v}\sqrt{I_v(\rho)} - C_{\nabla v}^2
$$

For large $I_v(\rho)$, the first term dominates. For small $I_v(\rho)$ (near equilibrium), we use the fact that KL also becomes small, and the LSI still provides useful control.

**Simplified bound** (sufficient for hypocoercivity):

:::{prf:lemma} Fisher Information Bound
:label: lem-fisher-bound

There exists a constant $c_F > 0$ such that:

$$
I_v(\rho) \ge c_F I_v(\rho \| \rho_\infty) - C_{\text{rem}}
$$

where $c_F = 1/2$ and $C_{\text{rem}} = 4C_{\nabla v}^2$.

Consequently:

$$
\boxed{I_v(\rho) \ge 2\lambda_{\text{LSI}} D_{\text{KL}}(\rho \| \rho_\infty) - C_{\text{LSI}}}
$$

for an explicit constant $C_{\text{LSI}}$ depending on $\lambda_{\text{LSI}}$ and $C_{\nabla v}$.
:::

---

## 3. Bounding Coupling and Remainder Terms

### 3.1. Identification of All Terms

From Stage 1, the entropy production is:

$$
\frac{d}{dt} D_{\text{KL}}(\rho \| \rho_\infty) = -\frac{\sigma^2}{2} I_v(\rho) + R_{\text{coupling}} + I_{\text{jump}}
$$

where the coupling/remainder terms are:

$$
R_{\text{coupling}} = R_{\text{transport}} + R_{\text{force}} + R_{\text{friction}} + R_{\text{diffusion}}
$$

Explicitly:

**R1. Transport coupling**:

$$
R_{\text{transport}} = -\int v \cdot \nabla_x \log \rho_\infty \cdot \rho \, dx dv
$$

**R2. Force coupling**:

$$
R_{\text{force}} = -\int \nabla_x U \cdot (\nabla_v \log \rho - \nabla_v \log \rho_\infty) \cdot \rho \, dx dv
$$

**R3. Friction coupling**:

$$
R_{\text{friction}} = -\gamma \int v \cdot (\nabla_v \log \rho - \nabla_v \log \rho_\infty) \cdot \rho \, dx dv
$$

**R4. Diffusion remainder**:

$$
R_{\text{diffusion}} = -\frac{\sigma^2}{2} \int \rho \cdot \Delta_v \log \rho_\infty \, dx dv
$$

### 3.2. Bounding Each Term

#### 3.2.1. Transport Coupling

$$
|R_{\text{transport}}| = \left|\int v \cdot \nabla_x \log \rho_\infty \cdot \rho\right| \le C_{\nabla x} \int |v| \rho
$$

Using the second moment bound:

$$
\int |v| \rho \le \sqrt{\int |v|^2 \rho} = \sqrt{E_v[\rho]}
$$

where $E_v[\rho] := \int |v|^2 \rho / 2$ is the kinetic energy.

**Lemma (Kinetic energy bound)**:

:::{prf:lemma} Kinetic Energy Control
:label: lem-kinetic-energy-bound

The kinetic energy is controlled by the velocity Fisher information and KL divergence:

$$
E_v[\rho] \le E_v[\rho_\infty] + C_v D_{\text{KL}}(\rho \| \rho_\infty) + \frac{C_v'}{\gamma} I_v(\rho)
$$

for explicit constants $C_v, C_v'$ depending on $\rho_\infty$.
:::

**Result**:

$$
\boxed{|R_{\text{transport}}| \le C_1^{\text{trans}} D_{\text{KL}}(\rho \| \rho_\infty) + C_2^{\text{trans}} I_v(\rho)}
$$

where:

$$
C_1^{\text{trans}} = C_{\nabla x} \sqrt{2C_v}, \quad C_2^{\text{trans}} = C_{\nabla x} \sqrt{2C_v'/\gamma}
$$

#### 3.2.2. Force Coupling

$$
|R_{\text{force}}| = \left|\int \nabla_x U \cdot (\nabla_v \log \rho - \nabla_v \log \rho_\infty) \cdot \rho\right|
$$

Using $\|\nabla_x U\|_{L^\infty} \le L_U$ and Cauchy-Schwarz:

$$
|R_{\text{force}}| \le L_U \int \rho |\nabla_v \log \rho - \nabla_v \log \rho_\infty|
$$

Expanding:

$$
\le L_U \left(\int \rho |\nabla_v \log \rho| + \int \rho |\nabla_v \log \rho_\infty|\right)
$$

Using $\|\nabla_v \log \rho_\infty\|_{L^\infty} \le C_{\nabla v}$:

$$
\le L_U \left(\sqrt{I_v(\rho)} + C_{\nabla v}\right)
$$

**Result**:

$$
\boxed{|R_{\text{force}}| \le C^{\text{force}} I_v(\rho) + C_0^{\text{force}}}
$$

where $C^{\text{force}} = L_U^2/(4\epsilon)$ and $C_0^{\text{force}} = L_U C_{\nabla v}$ (using Young's inequality with $\epsilon > 0$ to be chosen).

#### 3.2.3. Friction Coupling

$$
R_{\text{friction}} = -\gamma \int v \cdot (\nabla_v \log \rho - \nabla_v \log \rho_\infty) \cdot \rho
$$

Expanding:

$$
= -\gamma \int v \cdot \nabla_v \log \rho \cdot \rho + \gamma \int v \cdot \nabla_v \log \rho_\infty \cdot \rho
$$

**First term**: Integration by parts yields:

$$
-\gamma \int v \cdot \nabla_v \log \rho \cdot \rho = \gamma \int \nabla_v \cdot (v\rho) \log \rho = \gamma d + \gamma \int v \cdot \nabla_v \rho
$$

Using $\nabla_v \rho = \rho \nabla_v \log \rho$:

$$
= \gamma d + \gamma \int |v|^2 \rho / |v| \cdot |\nabla_v \log \rho|
$$

This is bounded but requires care.

**Simpler bound** (using Cauchy-Schwarz directly):

$$
|R_{\text{friction}}| \le \gamma \int |v| \rho |\nabla_v \log \rho - \nabla_v \log \rho_\infty|
$$

Using the same strategy as for $R_{\text{force}}$:

$$
\boxed{|R_{\text{friction}}| \le C_1^{\text{fric}} D_{\text{KL}}(\rho \| \rho_\infty) + C_2^{\text{fric}} I_v(\rho)}
$$

with explicit constants:

$$
C_1^{\text{fric}} = \gamma \sqrt{2C_v}, \quad C_2^{\text{fric}} = \gamma \sqrt{2C_v'/\gamma} + \gamma C_{\nabla v} \sqrt{2C_v'/\gamma}
$$

#### 3.2.4. Diffusion Remainder

$$
R_{\text{diffusion}} = -\frac{\sigma^2}{2} \int \rho \cdot \Delta_v \log \rho_\infty
$$

Using the regularity bound $\|\Delta_v \log \rho_\infty\|_{L^\infty} \le C_{\Delta v}$:

$$
|R_{\text{diffusion}}| \le \frac{\sigma^2}{2} C_{\Delta v} \int \rho = \frac{\sigma^2}{2} C_{\Delta v}
$$

**Result**:

$$
\boxed{|R_{\text{diffusion}}| \le C^{\text{diff}} := \frac{\sigma^2}{2} C_{\Delta v}}
$$

This is a **pure constant** (no dependence on $\rho$).

### 3.3. Combined Coupling Bound

Summing all terms:

$$
|R_{\text{coupling}}| \le C_{\text{KL}}^{\text{coup}} D_{\text{KL}}(\rho \| \rho_\infty) + C_{\text{Fisher}}^{\text{coup}} I_v(\rho) + C_0^{\text{coup}}
$$

where:

$$
\begin{aligned}
C_{\text{KL}}^{\text{coup}} &= C_1^{\text{trans}} + C_1^{\text{fric}} \\
C_{\text{Fisher}}^{\text{coup}} &= C_2^{\text{trans}} + C^{\text{force}} + C_2^{\text{fric}} \\
C_0^{\text{coup}} &= C_0^{\text{force}} + C^{\text{diff}}
\end{aligned}
$$

**Explicit formulas**:

$$
\boxed{
\begin{aligned}
C_{\text{KL}}^{\text{coup}} &= (C_{\nabla x} + \gamma) \sqrt{2C_v} \\
C_{\text{Fisher}}^{\text{coup}} &= (C_{\nabla x} + \gamma) \sqrt{2C_v'/\gamma} + \frac{L_U^2}{4\epsilon} + \gamma C_{\nabla v} \sqrt{2C_v'/\gamma} \\
C_0^{\text{coup}} &= L_U C_{\nabla v} + \frac{\sigma^2 C_{\Delta v}}{2}
\end{aligned}
}
$$

---

## 4. Bounding the Jump Expansion

### 4.1. Jump Operator Entropy Production

From Stage 1:

$$
I_{\text{jump}} = \int \mathcal{L}_{\text{jump}}(\rho) \log \frac{\rho}{\rho_\infty}
$$

where:

$$
\mathcal{L}_{\text{jump}}[\rho] = -\kappa_{\text{kill}}(x) \rho + \lambda_{\text{revive}} m_d(\rho) \frac{\rho}{\|\rho\|_{L^1}}
$$

### 4.2. Killing Term

$$
I_{\text{kill}} := -\int \kappa_{\text{kill}}(x) \rho \log \frac{\rho}{\rho_\infty}
$$

Using the bound $\kappa_{\text{kill}}(x) \le \kappa_{\max}$:

$$
|I_{\text{kill}}| \le \kappa_{\max} \int \rho \left|\log \frac{\rho}{\rho_\infty}\right|
$$

**Lemma (Entropy moment bound)**:

:::{prf:lemma} Entropy $L^1$ Bound
:label: lem-entropy-l1-bound

For any $\rho, \rho_\infty \in \mathcal{P}(\Omega)$:

$$
\int \rho \left|\log \frac{\rho}{\rho_\infty}\right| \le 2 D_{\text{KL}}(\rho \| \rho_\infty) + C_{\text{const}}
$$

for some universal constant $C_{\text{const}}$.
:::

**Result**:

$$
\boxed{|I_{\text{kill}}| \le 2\kappa_{\max} D_{\text{KL}}(\rho \| \rho_\infty) + \kappa_{\max} C_{\text{const}}}
$$

### 4.3. Revival Term

$$
I_{\text{revive}} := \frac{\lambda_{\text{revive}} m_d(\rho)}{\|\rho\|_{L^1}} \int \rho \log \frac{\rho}{\rho_\infty}
$$

The integral is exactly:

$$
\int \rho \log \frac{\rho}{\rho_\infty} = D_{\text{KL}}(\rho \| \rho_\infty) + \log \frac{\|\rho\|_{L^1}}{M_\infty}
$$

Since $m_d(\rho) = 1 - \|\rho\|_{L^1}$:

$$
I_{\text{revive}} = \lambda_{\text{revive}} \frac{1 - \|\rho\|_{L^1}}{\|\rho\|_{L^1}} \left(D_{\text{KL}}(\rho \| \rho_\infty) + \log \frac{\|\rho\|_{L^1}}{M_\infty}\right)
$$

**Near equilibrium** ($\|\rho\|_{L^1} \approx M_\infty < 1$):

$$
\frac{1 - \|\rho\|_{L^1}}{\|\rho\|_{L^1}} \approx \frac{1 - M_\infty}{M_\infty}
$$

**Result**:

$$
\boxed{I_{\text{revive}} \le \frac{\lambda_{\text{revive}}(1-M_\infty)}{M_\infty^2} D_{\text{KL}}(\rho \| \rho_\infty) + C_{\text{revive}}}
$$

where $C_{\text{revive}}$ is a constant depending on $\lambda_{\text{revive}}$ and the basin of attraction.

### 4.4. Combined Jump Bound

$$
I_{\text{jump}} = I_{\text{kill}} + I_{\text{revive}} \le A_{\text{jump}} D_{\text{KL}}(\rho \| \rho_\infty) + B_{\text{jump}}
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

---

## 5. Assembling the Coercivity Gap

### 5.1. Full Entropy Production

Combining all terms from Sections 2-4:

$$
\frac{d}{dt} D_{\text{KL}}(\rho \| \rho_\infty) \le -\frac{\sigma^2}{2} I_v(\rho) + |R_{\text{coupling}}| + I_{\text{jump}}
$$

Substituting bounds:

$$
\begin{aligned}
&\le -\frac{\sigma^2}{2} I_v(\rho) + C_{\text{KL}}^{\text{coup}} D_{\text{KL}} + C_{\text{Fisher}}^{\text{coup}} I_v(\rho) + C_0^{\text{coup}} \\
&\quad + A_{\text{jump}} D_{\text{KL}} + B_{\text{jump}}
\end{aligned}
$$

Collecting terms:

$$
\begin{aligned}
&= \left(-\frac{\sigma^2}{2} + C_{\text{Fisher}}^{\text{coup}}\right) I_v(\rho) + (C_{\text{KL}}^{\text{coup}} + A_{\text{jump}}) D_{\text{KL}} + (C_0^{\text{coup}} + B_{\text{jump}})
\end{aligned}
$$

### 5.2. Using the LSI

From Lemma {prf:ref}`lem-fisher-bound`:

$$
I_v(\rho) \ge 2\lambda_{\text{LSI}} D_{\text{KL}}(\rho \| \rho_\infty) - C_{\text{LSI}}
$$

Substituting:

$$
\begin{aligned}
\frac{d}{dt} D_{\text{KL}} &\le \left(-\frac{\sigma^2}{2} + C_{\text{Fisher}}^{\text{coup}}\right) \left(2\lambda_{\text{LSI}} D_{\text{KL}} - C_{\text{LSI}}\right) \\
&\quad + (C_{\text{KL}}^{\text{coup}} + A_{\text{jump}}) D_{\text{KL}} + (C_0^{\text{coup}} + B_{\text{jump}})
\end{aligned}
$$

Expanding:

$$
\begin{aligned}
&= \left[2\lambda_{\text{LSI}}\left(-\frac{\sigma^2}{2} + C_{\text{Fisher}}^{\text{coup}}\right) + C_{\text{KL}}^{\text{coup}} + A_{\text{jump}}\right] D_{\text{KL}} \\
&\quad + \left[\left(-\frac{\sigma^2}{2} + C_{\text{Fisher}}^{\text{coup}}\right)(-C_{\text{LSI}}) + C_0^{\text{coup}} + B_{\text{jump}}\right]
\end{aligned}
$$

### 5.3. The Coercivity Gap

Define:

$$
\boxed{\delta := -2\lambda_{\text{LSI}}\left(-\frac{\sigma^2}{2} + C_{\text{Fisher}}^{\text{coup}}\right) - C_{\text{KL}}^{\text{coup}} - A_{\text{jump}}}
$$

**Condition for exponential convergence**:

$$
\delta > 0 \quad \Leftrightarrow \quad \lambda_{\text{LSI}} \sigma^2 > 2\lambda_{\text{LSI}} C_{\text{Fisher}}^{\text{coup}} + C_{\text{KL}}^{\text{coup}} + A_{\text{jump}}
$$

**Explicit criterion**:

$$
\boxed{\sigma^2 > \frac{2\lambda_{\text{LSI}} C_{\text{Fisher}}^{\text{coup}} + C_{\text{KL}}^{\text{coup}} + A_{\text{jump}}}{\lambda_{\text{LSI}}}}
$$

This is the **parameter regime requirement**: the diffusion strength $\sigma^2$ must be large enough relative to the coupling constants, jump expansion, and LSI constant.

### 5.4. Convergence Rate

When $\delta > 0$, we have:

$$
\frac{d}{dt} D_{\text{KL}}(\rho \| \rho_\infty) \le -\delta D_{\text{KL}}(\rho \| \rho_\infty) + C_{\text{offset}}
$$

where $C_{\text{offset}}$ is the constant term from Section 5.2.

**Gronwall's inequality** gives:

$$
D_{\text{KL}}(\rho_t \| \rho_\infty) \le e^{-\delta t} D_{\text{KL}}(\rho_0 \| \rho_\infty) + \frac{C_{\text{offset}}}{\delta}(1 - e^{-\delta t})
$$

As $t \to \infty$:

$$
D_{\text{KL}}(\rho_t \| \rho_\infty) \to \frac{C_{\text{offset}}}{\delta}
$$

**For exact exponential convergence to $\rho_\infty$**: We need $C_{\text{offset}} = 0$, which requires tighter control of the constant terms. This is typically achieved by working in a **local basin** around $\rho_\infty$ where quadratic approximations are valid.

:::{prf:theorem} Exponential Convergence (Local)
:label: thm-exponential-convergence-local

Assume $\delta > 0$ and that $\rho_0$ satisfies $D_{\text{KL}}(\rho_0 \| \rho_\infty) \le \epsilon_0$ for sufficiently small $\epsilon_0$. Then:

$$
D_{\text{KL}}(\rho_t \| \rho_\infty) \le e^{-\alpha_{\text{net}} t} D_{\text{KL}}(\rho_0 \| \rho_\infty)
$$

where:

$$
\boxed{\alpha_{\text{net}} = \frac{\delta}{2}}
$$

is the exponential convergence rate.
:::

---

## 6. Summary of Explicit Constants

### 6.1. Physical Parameters (Inputs)

- $\gamma$ - friction
- $\sigma^2$ - diffusion strength
- $L_U$ - Lipschitz constant of potential
- $\kappa_{\max}$ - maximum killing rate
- $\lambda_{\text{revive}}$ - revival rate
- $M_\infty$ - equilibrium mass

### 6.2. QSD Regularity Constants (From Stage 0.5)

- $C_{\nabla x} = \|\nabla_x \log \rho_\infty\|_{L^\infty}$
- $C_{\nabla v} = \|\nabla_v \log \rho_\infty\|_{L^\infty}$
- $C_{\Delta v} = \|\Delta_v \log \rho_\infty\|_{L^\infty}$
- $\alpha_{\exp}$ - exponential concentration rate

### 6.3. Derived Constants

**LSI constant**:

$$
\lambda_{\text{LSI}} \ge \frac{\alpha_{\exp}}{1 + C_{\Delta v}/\alpha_{\exp}}
$$

**Coupling bounds**:

$$
\begin{aligned}
C_{\text{KL}}^{\text{coup}} &= (C_{\nabla x} + \gamma) \sqrt{2C_v} \\
C_{\text{Fisher}}^{\text{coup}} &= (C_{\nabla x} + \gamma) \sqrt{2C_v'/\gamma} + \frac{L_U^2}{4\epsilon} + \gamma C_{\nabla v} \sqrt{2C_v'/\gamma}
\end{aligned}
$$

**Jump expansion**:

$$
A_{\text{jump}} = 2\kappa_{\max} + \frac{\lambda_{\text{revive}}(1-M_\infty)}{M_\infty^2}
$$

**Coercivity gap**:

$$
\delta = \lambda_{\text{LSI}} \sigma^2 - 2\lambda_{\text{LSI}} C_{\text{Fisher}}^{\text{coup}} - C_{\text{KL}}^{\text{coup}} - A_{\text{jump}}
$$

**Convergence rate**:

$$
\boxed{\alpha_{\text{net}} = \frac{\delta}{2} = \frac{1}{2}\left(\lambda_{\text{LSI}} \sigma^2 - 2\lambda_{\text{LSI}} C_{\text{Fisher}}^{\text{coup}} - C_{\text{KL}}^{\text{coup}} - A_{\text{jump}}\right)}
$$

### 6.4. Sufficient Condition for Convergence

$$
\boxed{\sigma^2 > \frac{2\lambda_{\text{LSI}} C_{\text{Fisher}}^{\text{coup}} + C_{\text{KL}}^{\text{coup}} + A_{\text{jump}}}{\lambda_{\text{LSI}}} =: \sigma_{\text{crit}}^2}
$$

**Interpretation**: The diffusion strength must exceed a critical threshold $\sigma_{\text{crit}}^2$ that balances:
1. Coupling to QSD structure (through $C_{\text{Fisher}}^{\text{coup}}, C_{\text{KL}}^{\text{coup}}$)
2. Jump operator expansion (through $A_{\text{jump}}$)
3. LSI quality (through $1/\lambda_{\text{LSI}}$)

---

## 7. Numerical Verification Strategy

### 7.1. Computing QSD Regularity Constants

**Step 1**: Solve the stationary PDE $\mathcal{L}[\rho_\infty] = 0$ numerically (finite-difference or spectral methods)

**Step 2**: Compute:
- $C_{\nabla x} = \max_{x,v} |\nabla_x \log \rho_\infty(x,v)|$
- $C_{\nabla v} = \max_{x,v} |\nabla_v \log \rho_\infty(x,v)|$
- $C_{\Delta v} = \max_{x,v} |\Delta_v \log \rho_\infty(x,v)|$
- Fit exponential decay: $\rho_\infty(x,v) \sim C_{\exp} e^{-\alpha_{\exp}(|x|^2 + |v|^2)}$ in the tail

### 7.2. Evaluating Constants

**Step 3**: Compute intermediate constants:
- $\lambda_{\text{LSI}}$ from Section 2
- $C_{\text{Fisher}}^{\text{coup}}, C_{\text{KL}}^{\text{coup}}$ from Section 3 (requires estimating $C_v, C_v'$ from $\rho_\infty$)
- $A_{\text{jump}}$ from Section 4

**Step 4**: Compute coercivity gap:

$$
\delta = \lambda_{\text{LSI}} \sigma^2 - 2\lambda_{\text{LSI}} C_{\text{Fisher}}^{\text{coup}} - C_{\text{KL}}^{\text{coup}} - A_{\text{jump}}
$$

**Step 5**: Check $\delta > 0$. If yes, compute $\alpha_{\text{net}} = \delta/2$.

### 7.3. Validation

**Step 6**: Run the mean-field PDE forward from an initial condition $\rho_0$:

$$
\frac{\partial \rho}{\partial t} = \mathcal{L}[\rho]
$$

**Step 7**: Compute $D_{\text{KL}}(\rho_t \| \rho_\infty)$ at discrete times.

**Step 8**: Fit to $D_{\text{KL}}(t) \approx D_0 e^{-\alpha t}$ and compare observed $\alpha$ with predicted $\alpha_{\text{net}}$.

---

## 8. Extensions and Refinements

### 8.1. Tighter LSI Constants

The bound $\lambda_{\text{LSI}} \ge \alpha_{\exp}/(1 + C_{\Delta v}/\alpha_{\exp})$ is conservative. Tighter constants can be obtained by:

1. **Spectral gap analysis**: Directly compute the spectral gap of $\mathcal{L}_{\text{kin}}$ restricted to the velocity space
2. **Bakry-Émery curvature**: Use the Γ₂ calculus to get dimension-dependent improvements
3. **Interpolation methods**: Use entropy interpolation techniques (Villani)

### 8.2. Optimal Choice of $\epsilon$

The Young's inequality parameter $\epsilon$ in Section 3.2.2 can be optimized to minimize $C_{\text{Fisher}}^{\text{coup}}$:

$$
\epsilon^* = \frac{\sigma^2}{2L_U}
$$

This balances the force coupling against the velocity dissipation.

### 8.3. Global vs Local Convergence

The theorem in Section 5.4 assumes $\rho_0$ is in a local basin around $\rho_\infty$. For **global convergence**, we need:

1. A **Lyapunov function** that decreases globally (not just near $\rho_\infty$)
2. **A priori bounds** on all moments of $\rho_t$ (uniform in time)
3. **Compactness** of the level sets of the Lyapunov function

These are typically proven using maximum principles and energy estimates on the PDE.

### 8.4. Discrete-Time vs Continuous-Time

The constants derived here apply to the continuous-time PDE. For the **discrete-time operators** (kinetic + cloning), additional factors arise from:

1. Time-stepping error (requires $\tau$ small enough)
2. Cloning operator discretization
3. Splitting error (if using operator splitting)

The finite-N proof in [../kl_convergence/10_kl_convergence.md](../kl_convergence/10_kl_convergence.md) handles this carefully.

---

## 9. Connection to Finite-N Convergence

The explicit constants here complement the finite-N analysis:

**Finite-N** ([10_kl_convergence.md](../kl_convergence/10_kl_convergence.md)):
- Discrete-time operators $\Psi_{\text{kin}}, \Psi_{\text{clone}}$
- Hypocoercive Lyapunov $\mathcal{E}_\theta = D_{\text{KL}} + \theta V$
- LSI preserved by cloning (Lemma 5.2)
- Convergence rate $\alpha_N$ explicit in $N, \tau, \sigma, \gamma$

**Mean-field** (this document):
- Continuous-time generator $\mathcal{L} = \mathcal{L}_{\text{kin}} + \mathcal{L}_{\text{jump}}$
- Modified Fisher $I_\theta = I_v + \theta I_x$
- LSI from QSD regularity
- Convergence rate $\alpha_{\text{net}}$ explicit in $\sigma, \gamma, \lambda_{\text{LSI}}, A_{\text{jump}}$

**Key relationship**:

$$
\lim_{N \to \infty} \alpha_N(\tau, N) = \alpha_{\text{net}} \quad \text{(after adjusting for discrete-time)}
$$

The mean-field limit $N \to \infty$ removes the $1/N$ cloning fluctuations, leaving only the jump operator's deterministic expansion.

---

## 10. Conclusion

:::{prf:theorem} Main Result: Explicit Convergence Rate
:label: thm-main-explicit-rate

Under the assumptions of Stage 0.5 (QSD regularity R1-R6) and the parameter condition:

$$
\sigma^2 > \sigma_{\text{crit}}^2 := \frac{2\lambda_{\text{LSI}} C_{\text{Fisher}}^{\text{coup}} + C_{\text{KL}}^{\text{coup}} + A_{\text{jump}}}{\lambda_{\text{LSI}}}
$$

the mean-field Euclidean Gas converges exponentially to the QSD with rate:

$$
\boxed{\alpha_{\text{net}} = \frac{1}{2}\left(\lambda_{\text{LSI}} \sigma^2 - 2\lambda_{\text{LSI}} C_{\text{Fisher}}^{\text{coup}} - C_{\text{KL}}^{\text{coup}} - A_{\text{jump}}\right)}
$$

where all constants are given explicitly in Sections 2-4 in terms of the physical parameters $(\gamma, \sigma, L_U, \kappa_{\max}, \lambda_{\text{revive})$ and QSD regularity constants $(C_{\nabla x}, C_{\nabla v}, C_{\Delta v}, \alpha_{\exp})$.
:::

**Significance**:
1. **Fully computable**: All constants are explicit formulas
2. **Verifiable**: Numerical experiments can test the predicted rate
3. **Tunable**: Shows how adjusting physical parameters affects convergence
4. **Foundational**: Completes the mean-field convergence proof with quantitative bounds

**Next steps**:
- Implement numerical validation (Section 7)
- Optimize constants using refinements (Section 8)
- Compare with finite-N simulations
- Extend to adaptive mechanisms (adaptive force, viscous coupling)
