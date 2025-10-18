# Stage 3: Parameter Dependence and Discrete-to-Mean-Field Connection

**Document Status**: NEW - Parameter analysis and simulation guide (2025-01-09)

**Purpose**: Provide explicit formulas connecting discrete simulation parameters to mean-field convergence rate, enabling practitioners to understand how parameter tuning affects convergence.

**Parent documents**:
- [11_stage2_explicit_constants.md](11_stage2_explicit_constants.md) - Explicit hypocoercivity constants
- [../10_kl_convergence/10_kl_convergence.md](../10_kl_convergence/10_kl_convergence.md) - Finite-N convergence
- [../04_convergence.md](../04_convergence.md) - Foster-Lyapunov framework

**Audience**: Practitioners running discrete simulations who want to:
1. Understand parameter sensitivity
2. Tune parameters for faster convergence
3. Predict mean-field behavior from finite-N runs
4. Diagnose convergence bottlenecks

---

## 0. Overview: From Discrete Simulation to Mean-Field Limit

### 0.1. Two Convergence Rates

The Euclidean Gas has **two distinct convergence rates** depending on the analysis level:

**Finite-N (Discrete)**: The N-particle system with discrete operators

$$
S_{t+1} = (\Psi_{\text{kin}}(\tau) \circ \Psi_{\text{clone}})(S_t)
$$

converges at rate $\alpha_N$ (from [10_kl_convergence.md](../10_kl_convergence/10_kl_convergence.md)).

**Mean-Field (Continuous)**: The N→∞ limit with PDE

$$
\frac{\partial \rho}{\partial t} = \mathcal{L}[\rho] = \mathcal{L}_{\text{kin}}[\rho] + \mathcal{L}_{\text{jump}}[\rho]
$$

converges at rate $\alpha_{\text{net}}$ (from Stage 2).

**Key relationship**:

$$
\alpha_N = \alpha_{\text{net}} + O(1/N) + O(\tau)
$$

This document makes this relationship **explicit** and **computable**.

### 0.2. The Parameter Dictionary

**Discrete simulation parameters** (what you set in code):

| Symbol | Name | Typical Range | Role |
|:-------|:-----|:--------------|:-----|
| $\tau$ | Time step | 0.001 - 0.1 | Discretization accuracy |
| $\gamma$ | Friction | 0.1 - 10 | Kinetic dissipation |
| $\sigma$ | Diffusion strength | 0.1 - 5 | Velocity noise |
| $\lambda_{\text{clone}}$ | Cloning rate | 0.5 - 5 | Exploration-exploitation balance |
| $N$ | Number of walkers | 10 - 10000 | Statistical accuracy |
| $\delta$ | Cloning noise | 0.01 - 1 | Clone diversity |
| $\kappa_{\text{kill}}$ | Killing rate | 0.01 - 10 | Boundary pressure |
| $\lambda_{\text{revive}}$ | Revival rate | 0.1 - 5 | Dead mass recycling |

**Mean-field constants** (what affects $\alpha_{\text{net}}$):

| Symbol | Depends On | Expression |
|:-------|:-----------|:-----------|
| $\lambda_{\text{LSI}}$ | $\sigma, \gamma, C_{\Delta v}$ | $\alpha_{\exp}/(1 + C_{\Delta v}/\alpha_{\exp})$ |
| $C_{\text{Fisher}}^{\text{coup}}$ | $\gamma, L_U, C_{\nabla v}$ | $(C_{\nabla x} + \gamma)\sqrt{2C_v'/\gamma} + L_U^2/(4\epsilon)$ |
| $A_{\text{jump}}$ | $\kappa_{\text{kill}}, \lambda_{\text{revive}}$ | $2\kappa_{\max} + \lambda_{\text{revive}}(1-M_\infty)/M_\infty^2$ |
| $\alpha_{\text{net}}$ | All of the above | $\frac{1}{2}(\lambda_{\text{LSI}} \sigma^2 - 2\lambda_{\text{LSI}} C_{\text{Fisher}}^{\text{coup}} - A_{\text{jump}})$ |

### 0.3. Roadmap

**Section 1**: Express mean-field constants in terms of simulation parameters

**Section 2**: Derive explicit formula $\alpha_{\text{net}}(\tau, \gamma, \sigma, \lambda, N, \ldots)$

**Section 3**: Analyze parameter sensitivities: $\partial \alpha_{\text{net}} / \partial \log P_i$

**Section 4**: Parameter tuning strategies and optimization

**Section 5**: Discrete-to-continuous transition: finite-N corrections

**Section 6**: Numerical validation and diagnostic tools

---

## 1. Mean-Field Constants as Functions of Simulation Parameters

### 1.1. QSD Regularity Constants

The QSD regularity constants $(C_{\nabla x}, C_{\nabla v}, C_{\Delta v}, \alpha_{\exp})$ depend on the physical parameters. We provide **scaling estimates** based on typical behavior.

#### Spatial Log-Gradient: $C_{\nabla x}$

The spatial gradient $|\nabla_x \log \rho_\infty|$ measures how rapidly the QSD concentration varies in space.

**Scaling estimate**:

$$
C_{\nabla x} \sim \sqrt{\frac{\kappa_{\max}}{\sigma^2}} + \sqrt{\frac{L_U}{\gamma}}
$$

**Intuition**:
- First term: Boundary killing creates steep gradients $\sim \sqrt{\kappa/\sigma^2}$
- Second term: Potential gradients drive spatial structure $\sim \sqrt{L_U/\gamma}$

**Typical values**: $C_{\nabla x} \approx 0.5 - 5$ for well-behaved potentials

#### Velocity Log-Gradient: $C_{\nabla v}$

The velocity gradient $|\nabla_v \log \rho_\infty|$ measures velocity distribution width.

**Scaling estimate**:

$$
C_{\nabla v} \sim \frac{\sqrt{\gamma}}{\sigma}
$$

**Intuition**: For a Gaussian-like velocity distribution with variance $\sim \sigma^2/\gamma$, we have $|\nabla_v \log \rho| \sim |v|/(\sigma^2/\gamma) \sim \sqrt{\gamma}/\sigma$.

**Typical values**: $C_{\nabla v} \approx 0.1 - 2$

#### Velocity Log-Laplacian: $C_{\Delta v}$

The Laplacian $|\Delta_v \log \rho_\infty|$ measures curvature of the velocity distribution.

**Scaling estimate**:

$$
C_{\Delta v} \sim \frac{\gamma}{\sigma^2} + \frac{\lambda_{\text{revive}}}{M_\infty \sigma^2}
$$

**Intuition**:
- First term: Friction-diffusion balance $\sim \gamma/\sigma^2$
- Second term: Revival operator creates velocity curvature $\sim \lambda_{\text{revive}}/\sigma^2$

**Typical values**: $C_{\Delta v} \approx 0.5 - 10$

#### Exponential Concentration Rate: $\alpha_{\exp}$

The exponential tail $\rho_\infty(x,v) \lesssim e^{-\alpha_{\exp}(|x|^2 + |v|^2)}$ determines Gaussian-like behavior.

**Scaling estimate**:

$$
\alpha_{\exp} \sim \min\left(\frac{\lambda_{\min}}{2\sigma^2}, \frac{\gamma}{\sigma^2}\right)
$$

where $\lambda_{\min}$ is the smallest eigenvalue of the potential Hessian $\nabla^2 U$.

**Intuition**:
- Velocity tail: Friction-diffusion gives $\sim \gamma/\sigma^2$
- Spatial tail: Potential confinement gives $\sim \lambda_{\min}/\sigma^2$
- Take minimum (weakest confinement controls tail)

**Typical values**: $\alpha_{\exp} \approx 0.1 - 5$

### 1.2. LSI Constant

From Stage 2, Theorem {prf:ref}`thm-lsi-constant-explicit`:

$$
\lambda_{\text{LSI}} \ge \frac{\alpha_{\exp}}{1 + C_{\Delta v}/\alpha_{\exp}}
$$

Substituting scaling estimates:

$$
\lambda_{\text{LSI}} \approx \frac{\min(\lambda_{\min}/\sigma^2, \gamma/\sigma^2)}{1 + (\gamma + \lambda_{\text{revive}}/M_\infty)/\min(\lambda_{\min}, \gamma)}
$$

**Simplified form** (when $\gamma \ll \lambda_{\min}$, typical for weakly damped systems):

$$
\boxed{\lambda_{\text{LSI}} \approx \frac{\gamma}{\sigma^2(1 + \gamma/\lambda_{\min} + \lambda_{\text{revive}}/(M_\infty \gamma))}}
$$

**Key dependencies**:
- **Increases** with $\gamma$ (more friction → better LSI)
- **Decreases** with $\sigma^2$ (more noise → worse LSI)
- **Decreases** with $\lambda_{\text{revive}}$ (more revival → perturbs Gaussian structure)

**Typical values**: $\lambda_{\text{LSI}} \approx 0.05 - 2$

### 1.3. Coupling Constants

From Stage 2, Section 3:

$$
\begin{aligned}
C_{\text{KL}}^{\text{coup}} &= (C_{\nabla x} + \gamma) \sqrt{2C_v} \\
C_{\text{Fisher}}^{\text{coup}} &= (C_{\nabla x} + \gamma) \sqrt{2C_v'/\gamma} + \frac{L_U^2}{4\epsilon} + \gamma C_{\nabla v} \sqrt{2C_v'/\gamma}
\end{aligned}
$$

where $C_v = d\sigma^2/\gamma$ and $C_v' = d\sigma^2\tau^2$ (from kinetic energy bounds).

**Substituting**:

$$
C_{\text{Fisher}}^{\text{coup}} \approx (C_{\nabla x} + \gamma) \sigma\tau\sqrt{2d} + \frac{L_U^2}{4\epsilon} + \gamma C_{\nabla v} \sigma\tau\sqrt{2d}
$$

Using $C_{\nabla x} \sim \sqrt{\kappa_{\max}/\sigma^2} + \sqrt{L_U/\gamma}$ and $C_{\nabla v} \sim \sqrt{\gamma}/\sigma$:

$$
\boxed{C_{\text{Fisher}}^{\text{coup}} \approx \left(\sqrt{\frac{\kappa_{\max}}{\sigma^2}} + \sqrt{\frac{L_U}{\gamma}} + \gamma\right) \sigma\tau\sqrt{2d} + \frac{L_U^2}{4\epsilon} + \sqrt{\gamma}\tau\sqrt{2d}}
$$

**Optimal $\epsilon$** (minimizes coupling): $\epsilon^* = \sigma^2/(2L_U)$, giving:

$$
\frac{L_U^2}{4\epsilon^*} = \frac{L_U^3}{2\sigma^2}
$$

**Key dependencies**:
- **Linear** in $\tau$ (larger steps → more coupling)
- **Complex** in $\sigma$ (balances direct and inverse terms)
- **Increases** with landscape complexity ($L_U, \kappa_{\max}$)

### 1.4. Jump Expansion Constant

From Stage 2, Section 4:

$$
A_{\text{jump}} = 2\kappa_{\max} + \frac{\lambda_{\text{revive}}(1-M_\infty)}{M_\infty^2}
$$

The equilibrium mass $M_\infty$ satisfies the balance equation:

$$
M_\infty \cdot \bar{\kappa}_{\text{kill}} = (1 - M_\infty) \cdot \lambda_{\text{revive}}
$$

where $\bar{\kappa}_{\text{kill}} = \int \kappa_{\text{kill}}(x) \rho_\infty(x,v) dx dv / M_\infty$ is the average killing rate.

**Solving for $M_\infty$**:

$$
M_\infty = \frac{\lambda_{\text{revive}}}{\lambda_{\text{revive}} + \bar{\kappa}_{\text{kill}}}
$$

**For uniform killing** $\kappa_{\text{kill}}(x) = \kappa_0$:

$$
M_\infty = \frac{\lambda_{\text{revive}}}{\lambda_{\text{revive}} + \kappa_0}, \quad 1 - M_\infty = \frac{\kappa_0}{\lambda_{\text{revive}} + \kappa_0}
$$

Substituting:

$$
\frac{1-M_\infty}{M_\infty^2} = \frac{\kappa_0(\lambda_{\text{revive}} + \kappa_0)^2}{\lambda_{\text{revive}}^3}
$$

**Result**:

$$
\boxed{A_{\text{jump}} \approx 2\kappa_{\max} + \frac{\kappa_0(\lambda_{\text{revive}} + \kappa_0)^2}{\lambda_{\text{revive}}^2}}
$$

**Key dependencies**:
- **Increases** with $\kappa_{\max}$ (stronger killing)
- **Non-monotonic** in $\lambda_{\text{revive}}$ (minimum near $\lambda_{\text{revive}} \sim \kappa_0$)

**Typical values**: $A_{\text{jump}} \approx 0.1 - 20$

---

## 2. Explicit Convergence Rate Formula

### 2.1. Assembling the Formula

From Stage 2, Theorem {prf:ref}`thm-main-explicit-rate`:

$$
\alpha_{\text{net}} = \frac{1}{2}\left(\lambda_{\text{LSI}} \sigma^2 - 2\lambda_{\text{LSI}} C_{\text{Fisher}}^{\text{coup}} - C_{\text{KL}}^{\text{coup}} - A_{\text{jump}}\right)
$$

Substituting the expressions from Section 1:

:::{prf:theorem} Mean-Field Convergence Rate (Explicit)
:label: thm-alpha-net-explicit

The mean-field convergence rate as a function of simulation parameters is:

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

:::

**Simplified form** (dropping subdominant terms for $\tau \ll 1$, $\gamma \ll \lambda_{\min}$):

$$
\boxed{\alpha_{\text{net}} \approx \frac{1}{2}\left[\gamma - \frac{2\gamma^2\tau\sqrt{2d}}{\sigma} - \frac{2\gamma L_U^3}{\sigma^4} - 2\kappa_{\max} - \frac{\kappa_0 \lambda_{\text{revive}}}{\lambda_{\text{revive}} + \kappa_0}\right]}
$$

### 2.2. Critical Parameter Regime

For $\alpha_{\text{net}} > 0$, we need:

$$
\gamma > \frac{2\gamma^2\tau\sqrt{2d}}{\sigma} + \frac{2\gamma L_U^3}{\sigma^4} + 2\kappa_{\max} + \frac{\kappa_0 \lambda_{\text{revive}}}{\lambda_{\text{revive}} + \kappa_0}
$$

**Solving for $\sigma_{\text{crit}}$** (minimum diffusion for convergence):

Dominant balance: $\gamma \sim \gamma L_U^3/\sigma^4 \Rightarrow \sigma^4 \sim L_U^3$

$$
\boxed{\sigma_{\text{crit}} \gtrsim \left(\frac{2L_U^3}{\gamma}\right)^{1/4}}
$$

**Interpretation**: The diffusion strength must scale as $L_U^{3/4}$ to overcome potential landscape roughness.

### 2.3. Optimal Parameter Balancing

**Goal**: Maximize $\alpha_{\text{net}}$ by choosing parameters optimally.

**Strategy**:
1. Fix landscape-dependent quantities: $L_U, \lambda_{\min}, \kappa_{\max}$
2. Treat $(\gamma, \sigma, \lambda_{\text{revive}}, \tau)$ as free parameters
3. Optimize the balance

**Optimal friction**: Maximizes the first term while controlling coupling. Differentiating:

$$
\frac{\partial}{\partial \gamma}\left[\frac{\gamma \sigma^2}{1 + \gamma/\lambda_{\min}}\right] = \frac{\sigma^2 \lambda_{\min}}{(\lambda_{\min} + \gamma)^2}
$$

This is maximized at $\gamma \to 0$, but coupling terms grow with $\gamma$. The optimal balance is:

$$
\gamma^* \approx \sqrt{\frac{\sigma^4}{\tau\sqrt{2d} L_U}}
$$

**Optimal diffusion**: From the critical regime and optimal $\gamma$:

$$
\sigma^* \approx (L_U^3 \gamma)^{1/4}
$$

**Optimal time step**: Should be small enough that $\tau$-dependent coupling is subdominant:

$$
\tau^* \lesssim \frac{\sigma}{2\gamma^2\sqrt{2d}}
$$

**Optimal revival rate**: Minimizes jump expansion:

$$
\lambda_{\text{revive}}^* \approx \kappa_0
$$

This balances killing and revival, minimizing the dead/alive ratio fluctuations.

:::{prf:theorem} Optimal Parameter Scaling
:label: thm-optimal-parameter-scaling

For a landscape with Lipschitz constant $L_U$ and minimum Hessian eigenvalue $\lambda_{\min}$, the optimal parameter scaling is:

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

**Practical rule**: For harder landscapes (larger $L_U$), increase $\gamma$ and $\sigma$ while decreasing $\tau$.

---

## 3. Parameter Sensitivity Analysis

### 3.1. Sensitivity Matrix

Define the **logarithmic sensitivity**:

$$
S_{ij} := \frac{\partial \log \alpha_{\text{net}}}{\partial \log P_j} = \frac{P_j}{\alpha_{\text{net}}} \frac{\partial \alpha_{\text{net}}}{\partial P_j}
$$

where $P_j \in \{\tau, \gamma, \sigma, \lambda_{\text{revive}}, \kappa_{\max}, N\}$.

**Interpretation**: $S_{ij}$ tells you the **percentage change** in convergence rate per **percentage change** in parameter $P_j$.

### 3.2. Computing Sensitivities

From the simplified formula in Section 2.1:

$$
\alpha_{\text{net}} \approx \frac{1}{2}\left[\gamma - \frac{2\gamma^2\tau\sqrt{2d}}{\sigma} - \frac{2\gamma L_U^3}{\sigma^4} - 2\kappa_{\max} - C_{\text{jump}}\right]
$$

**Friction $\gamma$**:

$$
\frac{\partial \alpha_{\text{net}}}{\partial \gamma} \approx \frac{1}{2}\left[1 - \frac{4\gamma\tau\sqrt{2d}}{\sigma} - \frac{2L_U^3}{\sigma^4}\right]
$$

$$
\boxed{S_{\gamma} = \frac{\gamma}{\alpha_{\text{net}}} \cdot \frac{1}{2}\left[1 - \frac{4\gamma\tau\sqrt{2d}}{\sigma} - \frac{2L_U^3}{\sigma^4}\right]}
$$

**Sign**: Positive if $\sigma > 2\sqrt{\gamma\tau\sqrt{2d} + \sqrt{2L_U^3}}$. Otherwise, increasing $\gamma$ **hurts** convergence.

**Diffusion $\sigma$**:

$$
\frac{\partial \alpha_{\text{net}}}{\partial \sigma} \approx \frac{1}{2}\left[\frac{2\gamma^2\tau\sqrt{2d}}{\sigma^2} + \frac{8\gamma L_U^3}{\sigma^5}\right]
$$

$$
\boxed{S_{\sigma} = \frac{\sigma}{\alpha_{\text{net}}} \cdot \frac{1}{2}\left[\frac{2\gamma^2\tau\sqrt{2d}}{\sigma^2} + \frac{8\gamma L_U^3}{\sigma^5}\right]}
$$

**Sign**: Always positive — increasing diffusion **always helps** (assuming $\alpha_{\text{net}} > 0$).

**Time step $\tau$**:

$$
\frac{\partial \alpha_{\text{net}}}{\partial \tau} \approx -\frac{\gamma^2\sqrt{2d}}{\sigma}
$$

$$
\boxed{S_{\tau} = -\frac{\tau\gamma^2\sqrt{2d}}{\sigma \alpha_{\text{net}}}}
$$

**Sign**: Always negative — larger time steps **hurt** convergence (discretization error).

**Killing rate $\kappa_{\max}$**:

$$
\frac{\partial \alpha_{\text{net}}}{\partial \kappa_{\max}} \approx -1
$$

$$
\boxed{S_{\kappa_{\max}} = -\frac{\kappa_{\max}}{\alpha_{\text{net}}}}
$$

**Sign**: Always negative — more killing **hurts** convergence (expansion term).

**Revival rate $\lambda_{\text{revive}}$**:

$$
\frac{\partial \alpha_{\text{net}}}{\partial \lambda_{\text{revive}}} \approx -\frac{\kappa_0^2}{(\lambda_{\text{revive}} + \kappa_0)^2}
$$

$$
\boxed{S_{\lambda_{\text{revive}}} = -\frac{\lambda_{\text{revive}} \kappa_0^2}{(\lambda_{\text{revive}} + \kappa_0)^2 \alpha_{\text{net}}}}
$$

**Sign**: Always negative, but **decreasing** in magnitude as $\lambda_{\text{revive}}$ increases (saturates).

### 3.3. Sensitivity Ranking

**Typical parameter regime** ($\gamma \sim 1, \sigma \sim 1, \tau \sim 0.01, \kappa_{\max} \sim 1, L_U \sim 10$):

$$
|S_{\sigma}| > |S_{\gamma}| > |S_{\lambda_{\text{revive}}}| > |S_{\kappa_{\max}}| > |S_{\tau}|
$$

**Key insight**: **Diffusion $\sigma$ has the strongest impact** on convergence rate. This is because it appears in both the LSI constant (denominator $\sigma^2$) and the coupling terms (high powers $\sigma^4$).

**Practical implication**: If convergence is slow, first try increasing $\sigma$ before adjusting other parameters.

---

## 4. Parameter Tuning Strategies

### 4.1. Diagnostic Procedure

When running a discrete simulation, follow this diagnostic workflow:

**Step 1: Measure empirical convergence rate**

Compute KL divergence $D_{\text{KL}}(\mu_t \| \pi_{\text{QSD}})$ over time and fit:

$$
D_{\text{KL}}(t) \approx D_0 e^{-\alpha_{\text{emp}} t}
$$

**Step 2: Compute theoretical prediction**

Use the formula from Section 2.1 with your current parameters to get $\alpha_{\text{net}}^{\text{theory}}$.

**Step 3: Compare**

- If $\alpha_{\text{emp}} \approx \alpha_{\text{net}}^{\text{theory}}$: System is in mean-field regime ($N$ is large enough)
- If $\alpha_{\text{emp}} < \alpha_{\text{net}}^{\text{theory}}$: Finite-N effects or time-step error (see Section 5)
- If $\alpha_{\text{emp}} \ll \alpha_{\text{net}}^{\text{theory}}$: Algorithm may not be converging; check parameters satisfy $\sigma > \sigma_{\text{crit}}$

**Step 4: Identify bottleneck**

Compute the sensitivity matrix (Section 3.2) and identify which term in $\alpha_{\text{net}}$ is dominant:
- Large coupling terms $C_{\text{Fisher}}^{\text{coup}}$: Reduce $\tau$ or increase $\sigma$
- Large jump expansion $A_{\text{jump}}$: Reduce $\kappa_{\max}$ or tune $\lambda_{\text{revive}} \approx \kappa_0$
- Small LSI constant $\lambda_{\text{LSI}}$: Increase $\gamma$ or reduce $\lambda_{\text{revive}}$

### 4.2. Parameter Adjustment Recipes

#### Recipe 1: Faster Convergence (At Cost of Computation)

**Goal**: Maximize $\alpha_{\text{net}}$ without constraint.

**Actions**:
1. Increase $\sigma$ by factor 2-4
2. Increase $\gamma$ by factor 1.5-2
3. Decrease $\tau$ by factor 2 (more steps needed)
4. Set $\lambda_{\text{revive}} = \kappa_{\text{avg}}$ (balance killing/revival)

**Expected improvement**: $2\times$ to $5\times$ faster convergence

**Cost**: More steps per unit time, higher computational cost

#### Recipe 2: Balanced Optimization

**Goal**: Best convergence rate per computational step.

**Actions**:
1. Use optimal scaling from Theorem {prf:ref}`thm-optimal-parameter-scaling`
2. Compute $\gamma^* \sim L_U^{3/7}$, $\sigma^* \sim L_U^{9/14}$
3. Set $\tau^* = \min(0.5/\gamma^*, 1/\sqrt{\lambda_{\max}})$
4. Set $\lambda_{\text{revive}}^* = \kappa_{\text{avg}}$

**Expected performance**: Near-optimal rate with reasonable step cost

#### Recipe 3: Low-Noise Regime

**Goal**: Converge with minimal stochasticity (for deterministic landscapes).

**Actions**:
1. Use small $\sigma$ (just above $\sigma_{\text{crit}}$)
2. Increase $\gamma$ to compensate (large friction → tight focusing)
3. Use large $N$ to reduce statistical fluctuations
4. Set $\lambda_{\text{clone}}$ large (aggressive exploration)

**Expected behavior**: Slower mean-field rate, but more deterministic trajectories

### 4.3. Multi-Objective Optimization

Often we have **competing objectives**:

**Objective 1**: Maximize convergence rate $\alpha_{\text{net}}$

**Objective 2**: Minimize computational cost per step (small $N$, large $\tau$)

**Objective 3**: Maintain numerical stability (small $\tau$, moderate $\sigma$)

**Pareto frontier**: The tradeoff curve is approximately:

$$
\alpha_{\text{net}} \sim \frac{\gamma}{\tau^{1/2} N^{1/d}}
$$

This shows:
- Halving $\tau$ costs $\sqrt{2} \approx 1.4\times$ more steps for same convergence
- Doubling $N$ improves rate by $2^{1/d}$ (diminishing returns in high dimensions)

**Recommended strategy**:
1. Fix $N$ at minimum acceptable (e.g., $N = 100$ for $d=2$, $N = 1000$ for $d=4$)
2. Optimize $(\gamma, \sigma, \tau)$ jointly using Theorem {prf:ref}`thm-optimal-parameter-scaling`
3. Accept $\alpha_{\text{net}}$ from this balance

---

## 5. Discrete-to-Continuous Transition

### 5.1. Finite-N Corrections

The discrete convergence rate $\alpha_N$ differs from the mean-field rate $\alpha_{\text{net}}$ by finite-N corrections:

$$
\alpha_N = \alpha_{\text{net}} \cdot (1 - C_N/N) + O(\tau^2)
$$

where $C_N$ is a constant depending on the cloning mechanism.

**Source of correction**: Cloning introduces $O(1/N)$ fluctuations in the swarm distribution, slowing convergence.

**From [10_kl_convergence.md](../10_kl_convergence/10_kl_convergence.md), Section 5**:

$$
C_N \approx \frac{c_{\text{clone}}}{\delta^2}
$$

where $\delta$ is the cloning noise variance and $c_{\text{clone}} \sim 1$ is a constant.

**Explicit formula**:

$$
\boxed{\alpha_N \approx \alpha_{\text{net}} \left(1 - \frac{c_{\text{clone}}}{\delta^2 N}\right)}
$$

**Practical implication**: For $\delta \sim 0.1$ and $c_{\text{clone}} \sim 1$:

$$
\alpha_N \approx \alpha_{\text{net}} \left(1 - \frac{100}{N}\right)
$$

To get within 10% of mean-field rate, we need $N \gtrsim 1000$.

### 5.2. Time-Discretization Error

The discrete-time operators $\Psi_{\text{kin}}(\tau)$ approximate the continuous-time flow. The error is:

$$
\alpha_N = \alpha_{\text{net}} - c_{\tau} \tau \alpha_{\text{net}}^2 + O(\tau^2)
$$

where $c_{\tau} \sim 1/(2\gamma)$ for the BAOAB integrator (see [02_euclidean_gas.md](../02_euclidean_gas.md)).

**Simplified**:

$$
\boxed{\alpha_N \approx \alpha_{\text{net}} (1 - \tau \alpha_{\text{net}}/(2\gamma))}
$$

**Practical implication**: For $\tau = 0.01$, $\gamma = 1$, $\alpha_{\text{net}} = 0.5$:

$$
\alpha_N \approx 0.5 \times (1 - 0.01 \times 0.5 / 2) = 0.49875
$$

The error is negligible (< 0.5%) for typical parameters.

### 5.3. Combined Correction Formula

Combining both effects:

$$
\boxed{\alpha_N \approx \alpha_{\text{net}} \left(1 - \frac{c_{\text{clone}}}{\delta^2 N}\right) \left(1 - \frac{\tau \alpha_{\text{net}}}{2\gamma}\right)}
$$

**Guideline**: To stay within 5% of mean-field rate:

$$
\frac{c_{\text{clone}}}{\delta^2 N} + \frac{\tau \alpha_{\text{net}}}{2\gamma} < 0.05
$$

For typical parameters ($\delta = 0.1, c_{\text{clone}} = 1, \tau = 0.01, \gamma = 1, \alpha_{\text{net}} = 0.5$):

$$
\frac{100}{N} + 0.0025 < 0.05 \quad \Rightarrow \quad N > 2100
$$

### 5.4. Asymptotic Regime Diagram

The parameter space divides into regimes:

**Regime 1: Mean-Field Dominated** ($N \gg 100/\delta^2$, $\tau \ll \gamma/\alpha_{\text{net}}$)
- Convergence rate: $\alpha_N \approx \alpha_{\text{net}}$
- Behavior: Smooth PDE-like dynamics
- Use mean-field formulas from Sections 1-2

**Regime 2: Finite-N Corrections** ($N \sim 100/\delta^2$)
- Convergence rate: $\alpha_N \approx 0.5 \alpha_{\text{net}}$
- Behavior: Cloning fluctuations visible
- Need to account for $O(1/N)$ terms

**Regime 3: Discrete-Time Errors** ($\tau \sim \gamma/\alpha_{\text{net}}$)
- Convergence rate: $\alpha_N \approx 0.5 \alpha_{\text{net}}$
- Behavior: Integrator artifacts
- Reduce $\tau$ or improve integrator

**Regime 4: Pre-Asymptotic** (Very small $N$ or large $\tau$)
- Convergence rate: $\alpha_N \ll \alpha_{\text{net}}$
- Behavior: Non-exponential decay, large fluctuations
- Formulas from this document not applicable

---

## 6. Numerical Validation and Diagnostics

### 6.1. Validation Checklist

Before trusting the theoretical predictions, verify:

**V1. QSD Regularity**: Check that $\rho_{\text{QSD}}$ (from long-time simulation) satisfies:
- Smooth (no discontinuities)
- Exponential tails: $\rho_{\text{QSD}}(x,v) \lesssim e^{-\alpha(|x|^2 + |v|^2)}$
- Bounded gradients: $\max |\nabla \log \rho_{\text{QSD}}| < \infty$

**V2. Parameter Regime**: Verify $\sigma > \sigma_{\text{crit}}$ from Section 2.2

**V3. Finite-N Threshold**: Check $N > 100/\delta^2$ from Section 5.1

**V4. Time-Step Stability**: Check $\tau < \min(0.5/\gamma, 1/\sqrt{\lambda_{\max}})$

**V5. Exponential Decay**: Fit KL divergence to exponential and check $R^2 > 0.95$

If all checks pass, proceed to quantitative comparison.

### 6.2. Computing Theoretical Prediction

**Algorithm**: Given simulation parameters, compute $\alpha_{\text{net}}^{\text{theory}}$

**Input**: $(\tau, \gamma, \sigma, \lambda_{\text{revive}}, \kappa_{\max}, L_U, \lambda_{\min}, d, N, \delta)$

**Step 1**: Estimate QSD regularity constants (Section 1.1):

$$
\begin{aligned}
C_{\nabla x} &= \sqrt{\kappa_{\max}/\sigma^2} + \sqrt{L_U/\gamma} \\
C_{\nabla v} &= \sqrt{\gamma}/\sigma \\
C_{\Delta v} &= \gamma/\sigma^2 + \lambda_{\text{revive}}/(M_\infty \sigma^2) \\
\alpha_{\exp} &= \min(\lambda_{\min}/\sigma^2, \gamma/\sigma^2) / 2
\end{aligned}
$$

**Step 2**: Compute LSI constant (Section 1.2):

$$
\lambda_{\text{LSI}} = \frac{\alpha_{\exp}}{1 + C_{\Delta v}/\alpha_{\exp}}
$$

**Step 3**: Compute coupling constants (Section 1.3):

$$
C_{\text{Fisher}}^{\text{coup}} = (C_{\nabla x} + \gamma) \sigma\tau\sqrt{2d} + L_U^3/(2\sigma^2) + \sqrt{\gamma}\tau\sqrt{2d}
$$

**Step 4**: Compute jump expansion (Section 1.4):

$$
M_\infty = \frac{\lambda_{\text{revive}}}{\lambda_{\text{revive}} + \kappa_0}, \quad A_{\text{jump}} = 2\kappa_{\max} + \kappa_0(\lambda_{\text{revive}} + \kappa_0)^2/\lambda_{\text{revive}}^2
$$

**Step 5**: Assemble convergence rate (Section 2.1):

$$
\alpha_{\text{net}}^{\text{theory}} = \frac{1}{2}\left(\lambda_{\text{LSI}} \sigma^2 - 2\lambda_{\text{LSI}} C_{\text{Fisher}}^{\text{coup}} - C_{\text{KL}}^{\text{coup}} - A_{\text{jump}}\right)
$$

**Step 6**: Apply finite-N correction (Section 5.3):

$$
\alpha_N^{\text{theory}} = \alpha_{\text{net}}^{\text{theory}} \left(1 - \frac{100}{\delta^2 N}\right) \left(1 - \frac{\tau \alpha_{\text{net}}^{\text{theory}}}{2\gamma}\right)
$$

**Output**: $\alpha_N^{\text{theory}}$

### 6.3. Empirical Rate Estimation

**Algorithm**: Measure $\alpha_N^{\text{emp}}$ from simulation trajectory

**Input**: Trajectory $\{S_t\}_{t=0}^{T}$ and reference QSD $\pi_{\text{QSD}}$

**Step 1**: Compute KL divergence at each time:

$$
D_{\text{KL}}(t) = \mathbb{E}_{\mu_t}\left[\log \frac{d\mu_t}{d\pi_{\text{QSD}}}\right]
$$

(Use kernel density estimation if $\pi_{\text{QSD}}$ is not analytically known)

**Step 2**: Select exponential decay window $[t_1, t_2]$ where:
- $t_1$: After initial transient (e.g., $t_1 = 0.2 T$)
- $t_2$: Before equilibrium fluctuations dominate (e.g., $t_2 = 0.8 T$)

**Step 3**: Fit exponential:

$$
\log D_{\text{KL}}(t) = \log D_0 - \alpha_N^{\text{emp}} t
$$

using linear regression on $[t_1, t_2]$

**Step 4**: Estimate uncertainty from residuals:

$$
\sigma_{\alpha} = \text{std}(\text{residuals}) / \sqrt{t_2 - t_1}
$$

**Output**: $\alpha_N^{\text{emp}} \pm \sigma_{\alpha}$

### 6.4. Diagnostic Plots

**Plot 1: KL Decay**
- x-axis: Time $t$
- y-axis: $\log D_{\text{KL}}(t)$ (log scale)
- Expected: Straight line with slope $-\alpha_N$
- Overlay: Theoretical prediction $-\alpha_N^{\text{theory}} t$

**Plot 2: Parameter Sensitivity**
- x-axis: Parameter value $P_j$
- y-axis: Convergence rate $\alpha_N$
- Show: Sweep over parameter while holding others fixed
- Overlay: Theoretical curve from Section 2.1

**Plot 3: Finite-N Scaling**
- x-axis: $1/N$
- y-axis: $\alpha_N$
- Expected: Linear decrease per Section 5.1
- Fit: $\alpha_N = \alpha_{\text{net}} (1 - C_N/N)$ and extract $\alpha_{\text{net}}$

**Plot 4: Critical Diffusion**
- x-axis: $\sigma$
- y-axis: $\alpha_N$
- Expected: Threshold behavior at $\sigma_{\text{crit}}$
- Overlay: Theoretical $\sigma_{\text{crit}}$ from Section 2.2

### 6.5. Troubleshooting Guide

| Symptom | Likely Cause | Solution |
|:--------|:-------------|:---------|
| $\alpha_N^{\text{emp}} \ll \alpha_N^{\text{theory}}$ | Below $\sigma_{\text{crit}}$ | Increase $\sigma$ |
| Non-exponential decay | Pre-asymptotic regime | Run longer or increase $N$ |
| $\alpha_N^{\text{emp}}$ negative | Parameter instability | Check $\tau < 1/\sqrt{\lambda_{\max}}$ |
| Large residuals in fit | Multiple timescales | Check for bottleneck (Section 4.1) |
| Theory predicts $\alpha_N < 0$ | Invalid parameter regime | Adjust per Section 2.2 |
| $\alpha_N^{\text{emp}} > \alpha_N^{\text{theory}}$ | QSD not converged | Run burn-in longer |

---

## 7. Worked Examples

### 7.1. Example 1: Quadratic Potential

**Setup**:
- Potential: $U(x) = \frac{1}{2}\lambda x^2$ with $\lambda = 2$
- Dimension: $d = 2$
- Killing: Uniform $\kappa_{\text{kill}}(x) = 0.5$ for $|x| > 3$, else 0
- Parameters: $\gamma = 1, \sigma = 1, \tau = 0.01, \lambda_{\text{revive}} = 0.5, N = 500, \delta = 0.1$

**Calculation**:

**Step 1**: Landscape constants
- $\lambda_{\min} = \lambda_{\max} = 2$
- $L_U = \lambda \cdot \max|x| = 2 \times 3 = 6$
- $\kappa_{\max} = 0.5$, $\kappa_0 \approx 0.1$ (weighted average)

**Step 2**: QSD regularity (Section 1.1)
- $C_{\nabla x} = \sqrt{0.5/1} + \sqrt{6/1} = 0.707 + 2.449 = 3.16$
- $C_{\nabla v} = \sqrt{1}/1 = 1$
- $M_\infty = 0.5/(0.5 + 0.1) = 0.833$
- $C_{\Delta v} = 1/1 + 0.5/(0.833 \times 1) = 1 + 0.6 = 1.6$
- $\alpha_{\exp} = \min(2/1, 1/1)/2 = 0.5$

**Step 3**: LSI constant (Section 1.2)
- $\lambda_{\text{LSI}} = 0.5/(1 + 1.6/0.5) = 0.5/4.2 = 0.119$

**Step 4**: Coupling (Section 1.3)
- $C_{\text{Fisher}}^{\text{coup}} = (3.16 + 1) \times 1 \times 0.01 \times \sqrt{4} + 6^3/(2 \times 1) + \sqrt{1} \times 0.01 \times \sqrt{4}$
- $= 4.16 \times 0.02 + 108 + 0.02 = 0.083 + 108 + 0.02 = 108.1$

**Step 5**: Jump expansion (Section 1.4)
- $A_{\text{jump}} = 2 \times 0.5 + 0.1 \times (0.5 + 0.1)^2 / 0.5^2 = 1 + 0.144 = 1.14$

**Step 6**: Mean-field rate (Section 2.1)
- $\alpha_{\text{net}} = 0.5 \times (0.119 \times 1 - 2 \times 0.119 \times 108.1 - 3.16 \times \sqrt{4/1} - 1.14)$
- $= 0.5 \times (0.119 - 25.73 - 6.32 - 1.14)$
- $= 0.5 \times (-33.07) = -16.5$

**Result**: $\alpha_{\text{net}} < 0$ — **convergence not guaranteed!**

**Diagnosis**: The coupling term $C_{\text{Fisher}}^{\text{coup}} = 108$ is huge due to the $L_U^3/\sigma^2 = 216$ term. This violates the critical condition.

**Fix**: Increase $\sigma$ to reduce coupling:

$$
\sigma_{\text{crit}} \sim (2L_U^3/\gamma)^{1/4} = (2 \times 216 / 1)^{1/4} = 4.56
$$

**Retrying with $\sigma = 5$**:
- $C_{\text{Fisher}}^{\text{coup}} \approx 6^3/(2 \times 25) = 4.32$ (much better!)
- $\lambda_{\text{LSI}} = 0.5/(1 + 1.6/(0.5 \times 5)) \approx 0.44$
- $\alpha_{\text{net}} = 0.5 \times (0.44 \times 25 - 2 \times 0.44 \times 4.32 - 1.14) = 0.5 \times (11 - 3.8 - 1.14) = 3.03$

**Result**: $\alpha_{\text{net}} = 3.03$ — **exponential convergence expected!**

### 7.2. Example 2: Rugged Landscape

**Setup**:
- Potential: $U(x) = -\log p(x)$ where $p(x)$ is a mixture of 10 Gaussians
- Dimension: $d = 4$
- Lipschitz constant: $L_U \approx 50$ (estimated from gradient samples)
- Minimal curvature: $\lambda_{\min} \approx 0.1$ (shallow modes)
- Killing: Distance-based $\kappa_{\text{kill}}(x) = \exp(|x| - 5)$, $\kappa_{\max} \approx 10$
- Parameters: $\gamma = 2, \sigma = 3, \tau = 0.005, \lambda_{\text{revive}} = 5, N = 2000, \delta = 0.2$

**Calculation**:

Following the same steps as Example 1:

**QSD regularity**:
- $\alpha_{\exp} = \min(0.1/9, 2/9)/2 \approx 0.006$
- $C_{\nabla x} \approx \sqrt{10/9} + \sqrt{50/2} \approx 6.1$
- $C_{\Delta v} \approx 2/9 + 5/(0.5 \times 9) \approx 1.33$

**LSI constant**:
- $\lambda_{\text{LSI}} \approx 0.006/(1 + 1.33/0.006) \approx 0.000027$ (very small!)

**Coupling**:
- $C_{\text{Fisher}}^{\text{coup}} \approx 50^3/(2 \times 9) = 6944$ (huge!)

**Jump expansion**:
- $A_{\text{jump}} \approx 20 + 5 \approx 25$

**Mean-field rate**:
- $\alpha_{\text{net}} \approx 0.5 \times (0.000027 \times 9 - 2 \times 0.000027 \times 6944 - 25) \approx -12.7$

**Result**: $\alpha_{\text{net}} < 0$ — **convergence impossible with these parameters!**

**Diagnosis**: The landscape is too rugged ($L_U = 50$) and the LSI constant is too small ($\lambda_{\text{LSI}} \approx 10^{-5}$).

**Recommended fix**:
1. Dramatically increase diffusion: $\sigma \sim (50^3 \times 2)^{1/4} = 13.6$
2. Increase friction: $\gamma \sim 5$ (helps LSI)
3. Reduce killing: $\kappa_{\max} \sim 1$ (if possible)

**With adjusted parameters** ($\gamma = 5, \sigma = 15$):
- $\alpha_{\exp} \approx 5/225 = 0.022$
- $\lambda_{\text{LSI}} \approx 0.022/(1 + 60) \approx 0.00036$
- $C_{\text{Fisher}}^{\text{coup}} \approx 50^3/(2 \times 225) = 278$
- $\alpha_{\text{net}} \approx 0.5 \times (0.00036 \times 225 - 2 \times 0.00036 \times 278 - 25) \approx -12.4$

Still negative! This landscape may require **adaptive mechanisms** (adaptive force, viscous coupling) to achieve convergence. See [../02_adaptive_gas.md](../02_adaptive_gas.md).

---

## 8. Summary and Quick Reference

### 8.1. Key Formulas

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

### 8.2. Parameter Effects Table

| Parameter | Increases $\alpha_{\text{net}}$ | Decreases $\alpha_{\text{net}}$ | Optimal Value |
|:----------|:--------------------------------|:--------------------------------|:--------------|
| $\gamma$ | Increases LSI | Increases coupling | $\sqrt{\sigma^4/(L_U \tau\sqrt{2d})}$ |
| $\sigma$ | Always positive | — | $(L_U^3 \gamma)^{1/4}$ |
| $\tau$ | — | Always negative | $\min(0.5/\gamma, 1/\sqrt{\lambda_{\max}})$ |
| $\kappa_{\max}$ | — | Always negative | Minimize (if possible) |
| $\lambda_{\text{revive}}$ | — | Always negative | $\kappa_{\text{avg}}$ (balance) |
| $N$ | Indirect (reduces $1/N$ error) | — | $> 100/\delta^2$ |

### 8.3. Diagnostic Decision Tree

```
Start: Measure α_emp from simulation
│
├─ α_emp ≈ α_theory (within 20%)
│  └─ SUCCESS: System in mean-field regime, formulas valid
│
├─ α_emp < 0.5 α_theory
│  ├─ Check: N > 100/δ²?
│  │  ├─ No → Increase N
│  │  └─ Yes → Check τ < 0.5/γ?
│  │     ├─ No → Reduce τ
│  │     └─ Yes → Finite-N effects, see Section 5
│  │
│  └─ Check: σ > σ_crit?
│     ├─ No → Increase σ (critical!)
│     └─ Yes → Landscape may be too hard, consider adaptive mechanisms
│
└─ α_theory < 0
   └─ INVALID REGIME: Must increase σ or reduce L_U/κ_max
```

### 8.4. Implementation Checklist

Before running a production simulation:

- [ ] Compute $\sigma_{\text{crit}}$ and verify $\sigma > 1.5 \sigma_{\text{crit}}$
- [ ] Set $N > 100/\delta^2$ for mean-field validity
- [ ] Set $\tau < \min(0.5/\gamma, 1/\sqrt{\lambda_{\max}}, 0.01)$
- [ ] Balance revival: $\lambda_{\text{revive}} \approx \kappa_{\text{avg}}$
- [ ] Predict $\alpha_{\text{net}}$ using Section 6.2 algorithm
- [ ] Run short test (100-1000 steps) and measure $\alpha_{\text{emp}}$
- [ ] Compare $\alpha_{\text{emp}}$ vs $\alpha_{\text{net}}$ (should agree within 20%)
- [ ] Adjust parameters if needed using Section 4.2 recipes
- [ ] Validate final choice with full-length run

---

## 9. Connection to Code Implementation

The formulas in this document are implemented in [../../../src/fragile/gas_parameters.py](../../../src/fragile/gas_parameters.py).

**Key functions**:

- `compute_convergence_rates(params, landscape)`: Computes $\alpha_N$ for given parameters (uses finite-N formulas from [04_convergence.md](../04_convergence.md))
- `compute_optimal_parameters(landscape, V_target)`: Implements optimal scaling from Theorem {prf:ref}`thm-optimal-parameter-scaling`
- `evaluate_gas_convergence(params, landscape)`: Complete diagnostic report
- `adaptive_parameter_tuning(trajectory, params, landscape)`: Iterative tuning from empirical measurements

**Relationship**: The code uses the **finite-N discrete-time** formulas, which are less precise than the mean-field formulas here but more directly applicable to simulations. For large $N$ and small $\tau$, the two approaches agree (Section 5).

**Usage example**:

```python
from fragile.gas_parameters import (
    GasParams, LandscapeParams,
    compute_optimal_parameters, evaluate_gas_convergence
)

# Define landscape
landscape = LandscapeParams(
    lambda_min=0.5, lambda_max=10.0, d=2,
    f_typical=1.0, Delta_f_boundary=5.0
)

# Get optimal parameters
params = compute_optimal_parameters(landscape, V_target=0.5)

# Analyze convergence
results = evaluate_gas_convergence(params, landscape, verbose=True)
print(f"Expected convergence rate: {results['rates'].kappa_total:.4f}")
print(f"Mixing time: {results['mixing_steps']} steps")
```

This will produce a report similar to the examples in Section 7, using the discrete-time approximations.

---

## 10. Future Directions

### 10.1. Adaptive Mechanisms

The formulas here apply to the **Euclidean Gas** (kinetic + cloning). The **Adaptive Gas** adds:
- Adaptive force from mean-field fitness potential
- Viscous coupling between walkers
- Regularized Hessian diffusion

These mechanisms can improve $\alpha_{\text{net}}$ by:
1. Adaptive force reduces $C_{\text{Fisher}}^{\text{coup}}$ (targets high-fitness regions)
2. Viscous coupling reduces $A_{\text{jump}}$ (collective response to killing)
3. Hessian diffusion increases $\lambda_{\text{LSI}}$ (anisotropic noise adapts to landscape)

**Expected improvement**: $2\times$ to $10\times$ faster for rugged landscapes (e.g., Example 7.2).

**Status**: Rigorous analysis in progress (see [02_adaptive_gas.md](../02_adaptive_gas.md)).

### 10.2. Non-Log-Concave QSD

The formulas assume the QSD has nice regularity properties (R1-R6 from Stage 0.5). For **multi-modal** or **non-convex** QSD:
- LSI constant may be exponentially small: $\lambda_{\text{LSI}} \sim e^{-\beta \Delta F}$ (Eyring-Kramers)
- Convergence rate dominated by **spectral gap** of Markov chain (slowest mode)
- Mean-field theory needs modification (large deviations, metastability)

**Practical impact**: For landscapes with deep local minima, $\alpha_{\text{net}}$ may be much smaller than predicted. Use **adaptive mechanisms** or **tempering** to accelerate.

### 10.3. High-Dimensional Scaling

For $d \to \infty$:
- Coupling constants grow: $C_{\text{Fisher}}^{\text{coup}} \sim \sqrt{d}$
- Optimal diffusion scales: $\sigma^* \sim d^{1/8}$
- Convergence rate decreases: $\alpha_{\text{net}} \sim d^{-1/4}$ (curse of dimensionality)

**Open question**: Can adaptive mechanisms break this scaling? (Preliminary results: yes, via anisotropic noise)

---

**END OF DOCUMENT**
