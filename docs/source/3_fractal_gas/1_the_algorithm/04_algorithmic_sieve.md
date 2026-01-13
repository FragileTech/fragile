# The Algorithmic Sieve: Parameter Constraints for Convergence

## TLDR

::::{dropdown} One-Page Summary
:open:

**Goal**: Derive the **tightest possible parameter bounds** for Fractal Gas convergence by combining three analysis layers.

**Three-Layer Hierarchy**:
1. **Appendix analysis** (sufficient conditions): Rate formulas with O(1) constants
2. **Algorithm bounds** (necessary conditions): Exact local bounds (fitness, Doeblin)
3. **Hypostructure tightening** (optimal): Certificate propagation sharpens bounds

**Five Master Constraints**:

| Constraint | Tight Bound |
|------------|-------------|
| Phase Control | $\Gamma = T_{\text{kin}}/T_{\text{clone}} \in [0.5, 2.0]$ |
| Acoustic Stability | $\gamma > C \cdot p_{\max} S_{\max} M^2 / h$ |
| Doeblin Minorization | $\epsilon \geq D_{\text{alg}}/4.1$ |
| Timestep (CFL) | $h < \min(1/(2\gamma), \kappa_x/(2\gamma))$ |
| Noise Injection | $\sigma_x^2 \geq C_{\text{KL}}/(\lambda_{\text{LSI}} N)$ |

**Convergence Rate**:

$$\kappa_{\text{total}} = \min\left(\frac{\gamma \rho_{\text{LSI}}}{M^2}, \kappa_W, \kappa_{\text{conf}}, \gamma - C\nu_{\text{clone}}M^2\right) - \frac{C}{\sqrt{N}}$$

**Key Result**: Hypostructure certificate propagation yields bounds that are **provably tighter** than either appendix or algorithm analysis alone.
::::

---

## Introduction

This chapter synthesizes convergence results from three sources to derive the **tightest valid parameter ranges** for the Fractal Gas algorithm:

1. **Appendix analysis** ({prf:ref}`def-hypocoercive-decay-rate`, {prf:ref}`prop-acoustic-limit`): Provides explicit rate formulas but with order-1 constant slack
2. **Algorithm document** ({prf:ref}`def-latent-fractal-gas-constants`, {prf:ref}`lem-latent-fractal-gas-companion-doeblin`): Provides exact local bounds that are necessary but not sufficient
3. **Hypostructure formalism** (Vol. 2, {prf:ref}`def-structural-sieve`): Provides certificate-based constraint propagation that optimally tightens bounds

The key insight is that **none of these analyses alone produces optimal bounds**. The hypostructure framework provides the mathematical machinery to combine and tighten them systematically.

---

(sec-three-layer-hierarchy)=
## Three-Layer Bound Hierarchy

### Layer 1: Appendix Analysis (Sufficient Conditions)

The convergence appendices provide explicit rate formulas:

:::{prf:definition} Hypocoercive Decay Rate
:label: def-alg-sieve-hypocoercive-rate

The hypocoercive decay rate for the kinetic Fokker-Planck operator is:

$$\Lambda_{\text{hypo}} \approx \frac{\gamma \cdot \rho_{\text{LSI}}}{M^2}$$

where:
- $\gamma$: Friction coefficient (dissipation strength)
- $\rho_{\text{LSI}}$: Log-Sobolev inequality constant
- $M = \sup \|\nabla_x^2 V_{\text{eff}}\|$: Hessian bound (landscape roughness)
:::

:::{prf:proposition} Acoustic Limit
:label: prop-alg-sieve-acoustic-limit

For the combined kinetic-cloning system to converge, the friction must dominate cloning shocks:

$$\gamma > C \cdot \nu_{\text{clone}} \cdot M^2$$

where $C$ is a dimension-dependent constant of order 1, and $\nu_{\text{clone}}$ is the cloning rate.

*Proof.* The total convergence rate is $\Lambda \approx \gamma/M^2 - C\nu_{\text{clone}}$. For stability, $\Lambda > 0$. $\square$
:::

**Limitation**: The constant $C$ is not sharp; it represents worst-case analysis.

### Layer 2: Algorithm Bounds (Necessary Conditions)

The algorithm document provides exact local bounds:

:::{prf:lemma} Fitness Bounds
:label: lem-alg-sieve-fitness-bounds

The fitness potential $V_{\text{fit}}$ satisfies:

$$V_{\min} := \eta^{\alpha+\beta} \leq V_{\text{fit}} \leq (A+\eta)^{\alpha+\beta} =: V_{\max}$$

where $\eta > 0$ is the positivity floor, $A > 0$ is the logistic bound, and $\alpha, \beta \geq 0$ are the reward/diversity exponents.

**Default values** ($\alpha = \beta = 1$, $\eta = 0.1$, $A = 2.0$):

$$V_{\min} = 0.01, \quad V_{\max} = 4.41$$
:::

:::{prf:lemma} Cloning Score Bound
:label: lem-alg-sieve-cloning-score-bound

The cloning score satisfies:

$$|S_i| \leq S_{\max} := \frac{V_{\max} - V_{\min}}{V_{\min} + \epsilon_{\text{clone}}}$$

**Default value**: $S_{\max} = (4.41 - 0.01)/(0.01 + 0.01) = 220$
:::

**Limitation**: These bounds are necessary but don't guarantee convergence alone.

### Layer 3: Hypostructure Tightening

The hypostructure framework provides **certificate-based constraint propagation**:

:::{prf:definition} Certificate Propagation
:label: def-alg-sieve-certificate-propagation

Each gate node $i$ in the 60-node sieve transforms input certificates to output constraints:

$$\text{Gate}_i: (\text{state } x, \text{context } \Gamma) \mapsto (\text{outcome } o, \text{certificate } K_i^+, \text{context } \Gamma')$$

The **closure operation** computes all logical consequences:

$$\text{Cl}(\Gamma) = \text{fixed point of promotion rules}$$
:::

The hypostructure produces tighter bounds because:
- It **combines** appendix rates with algorithm bounds
- It **propagates** constraints through the dependency graph
- It **tightens** via promotion rules and barrier certificates

---

(sec-master-constraints)=
## Master Constraint System

Combining all three layers, we derive the **necessary and sufficient conditions** for convergence.

### A. Thermal Balance (Phase Control)

:::{prf:definition} Phase Control Parameter
:label: def-alg-sieve-phase-control

The **phase control parameter** balances kinetic and cloning temperatures:

$$\Gamma := \frac{T_{\text{kin}}}{T_{\text{clone}}} = \frac{\sigma_v^2}{2\gamma} \cdot \frac{\alpha d}{\beta}$$

where:
- $T_{\text{kin}} = \sigma_v^2/(2\gamma)$: Kinetic temperature (fluctuation-dissipation)
- $T_{\text{clone}} = \beta/(\alpha d)$: Effective cloning temperature
- $d$: Latent space dimension
:::

:::{prf:proposition} Optimal Phase Regime
:label: prop-alg-sieve-optimal-phase

For optimal convergence, the phase control parameter should satisfy:

$$\boxed{\Gamma \in [0.5, 2.0]}$$

- $\Gamma \gg 1$ (gas phase): High entropy, uniform coverage, slow optimization
- $\Gamma \approx 1$ (liquid phase): Balanced exploitation/exploration, optimal
- $\Gamma \ll 1$ (crystal phase): Locked in local minima, poor exploration

*Derivation*:
- **Appendix**: $\Gamma \approx 1$ optimal for liquid phase dynamics
- **Algorithm**: $T_{\text{kin}} = c_2^2/(2\gamma)$ where $c_2^2 = (1 - e^{-2\gamma h})T_c$
- **Hypostructure**: Gate 4 (Scale) requires subcriticality $\alpha < \beta + \lambda_c$
:::

### B. Acoustic Stability (Friction Bound)

:::{prf:proposition} Friction Lower Bound
:label: prop-alg-sieve-friction-bound

The friction coefficient must satisfy:

$$\boxed{\gamma > \frac{C \cdot p_{\max} \cdot S_{\max}}{h} \cdot M^2}$$

where:
- $p_{\max}$: Maximum cloning probability scale
- $S_{\max}$: Cloning score bound ({prf:ref}`lem-alg-sieve-cloning-score-bound`)
- $h$: BAOAB timestep
- $M^2$: Hessian bound of effective potential

*Derivation*:
- **Appendix**: $\gamma > C \nu_{\text{clone}} M^2$ (acoustic limit)
- **Algorithm**: $\nu_{\text{clone}} \approx p_{\max} \cdot S_{\max} / h$ (cloning rate)
- **Hypostructure**: BarrierGap requires spectral gap $\inf\sigma(L) > 0$

**Explicit bound** (defaults: $p_{\max} = 1$, $S_{\max} = 220$, $h = 0.01$, $M^2 \sim 1$):

$$\gamma > C \cdot 22000 \cdot M^2 \implies \gamma \gtrsim 1 \text{ (for } C \sim 10^{-4}\text{)}$$
:::

### C. Minorization (Doeblin Condition)

:::{prf:proposition} Kernel Scale Bound
:label: prop-alg-sieve-kernel-bound

The companion kernel scale must satisfy:

$$\boxed{\frac{D_{\text{alg}}^2}{2\epsilon^2} \leq \ln\left(\frac{n_{\text{alive}} - 1}{p_{\min,\text{target}}}\right)}$$

Equivalently:

$$\epsilon \geq \frac{D_{\text{alg}}}{\sqrt{2\ln((n_{\text{alive}} - 1)/p_{\min})}}$$

*Derivation*:
- **Algorithm**: Doeblin minorization requires $m_\epsilon = \exp(-D_{\text{alg}}^2/(2\epsilon^2))$
- **Algorithm**: Companion probability $p_{\min} \geq m_\epsilon/(n_{\text{alive}} - 1)$
- **Hypostructure**: Node 10 (Ergodicity) certifies Doeblin condition

**Explicit bound** (target $p_{\min} = 0.01$, $n_{\text{alive}} = 50$):

$$\epsilon \geq \frac{D_{\text{alg}}}{\sqrt{2\ln(4900)}} \approx \frac{D_{\text{alg}}}{4.1}$$
:::

### D. Timestep Constraint (CFL-like)

:::{prf:proposition} Timestep Upper Bound
:label: prop-alg-sieve-timestep-bound

The BAOAB timestep must satisfy:

$$\boxed{h < \min\left(\frac{1}{2\gamma}, \frac{\kappa_x}{2\gamma}, \frac{1}{\Lambda_{\text{hypo}}}\right)}$$

*Derivation*:
- **Appendix**: $\tau \ll M^2/(\gamma \rho_{\text{LSI}})$ for discretization stability
- **Algorithm**: $c_1 = e^{-\gamma h} \in (0, 1)$ requires $\gamma h < \infty$
- **Hypostructure**: Foster-Lyapunov coupling requires $\gamma h \leq \kappa_x/2$
:::

### E. Noise Injection (LSI Constant)

:::{prf:proposition} Jitter Lower Bound
:label: prop-alg-sieve-jitter-bound

The cloning jitter must satisfy:

$$\boxed{\sigma_x^2 \geq \frac{C_{\text{KL}}}{\lambda_{\text{LSI}}} \cdot \frac{1}{N}}$$

where:

$$\lambda_{\text{LSI}} = \frac{\gamma \cdot \kappa_{\text{conf}} \cdot \kappa_W \cdot \delta^2}{C_0}$$

*Derivation*:
- **Appendix**: LSI constant depends on $\delta^2 = \sigma_x^2$ (jitter scale)
- **Algorithm**: $\sigma_x > 0$ prevents genealogical collapse
- **Hypostructure**: Gate 1 (Energy) + BarrierSat require finite KL bound
:::

---

(sec-quantitative-bounds)=
## Quantitative Bounds Table

:::{prf:definition} Valid Parameter Ranges
:label: def-alg-sieve-parameter-ranges

The following table summarizes the tight bounds on all algorithm parameters:

| Parameter | Symbol | Tight Lower Bound | Tight Upper Bound | Source |
|-----------|--------|-------------------|-------------------|--------|
| Population | $N$ | $\geq 2$ (Doeblin) | $\infty$ | Algorithm |
| Kernel scale | $\epsilon$ | $D_{\text{alg}}/4.1$ | $\infty$ | Hypostructure |
| Friction | $\gamma$ | $C \cdot p_{\max} S_{\max} M^2/h$ | $1/(2h)$ | Appendix + Algorithm |
| Temperature | $T_c$ | $> 0$ | $\sigma_v^2/(2\gamma\Gamma_{\max})$ | Phase control |
| Timestep | $h$ | $> 0$ | $\min(1/(2\gamma), \kappa_x/(2\gamma))$ | Algorithm |
| Cloning jitter | $\sigma_x$ | $\sqrt{C_{\text{KL}}/(\lambda_{\text{LSI}} N)}$ | $\epsilon$ (locality) | Hypostructure |
| Reward exponent | $\alpha$ | $\geq 0$ | $< \beta + \lambda_c$ (subcritical) | Hypostructure |
| Diversity exponent | $\beta$ | $> 0$ (diversity) | $\infty$ | Algorithm |
| Positivity floor | $\eta$ | $> 0$ | $\ll A$ | Algorithm |
| Logistic bound | $A$ | $> 0$ | $\infty$ | Algorithm |
| Clone regularizer | $\epsilon_{\text{clone}}$ | $> 0$ | $V_{\min}$ | Algorithm |
| Max clone prob | $p_{\max}$ | $> 0$ | $1$ | Algorithm |
:::

---

(sec-convergence-rate)=
## Convergence Rate Formula

:::{prf:theorem} Total Convergence Rate
:label: thm-alg-sieve-total-rate

The total discrete-time convergence rate for the Fractal Gas is:

$$\kappa_{\text{total}} = \min\left(\underbrace{\frac{\gamma \rho_{\text{LSI}}}{M^2}}_{\text{hypocoercive}}, \underbrace{\kappa_W}_{\text{Wasserstein}}, \underbrace{\kappa_{\text{conf}}}_{\text{boundary}}, \underbrace{\gamma - C\nu_{\text{clone}}M^2}_{\text{acoustic}}\right) - \underbrace{\frac{C}{\sqrt{N}}}_{\text{finite-}N}$$

where:
- $\kappa_W$: Wasserstein contraction from companion geometry
- $\kappa_{\text{conf}}$: Boundary confinement spectral gap
- $C/\sqrt{N}$: Finite-population correction

*Proof.* Each term bounds a different contraction mechanism:
1. Hypocoercive term: Kinetic Fokker-Planck entropy dissipation
2. Wasserstein term: Companion geometry contraction
3. Boundary term: Dirichlet spectral gap on domain
4. Acoustic term: Stability margin from {prf:ref}`prop-alg-sieve-acoustic-limit`
5. Finite-$N$ correction: Mean-field approximation error

The minimum determines the bottleneck. $\square$
:::

:::{prf:corollary} Mixing Time
:label: cor-alg-sieve-mixing-time

The mixing time to reach error $\varepsilon$ is:

$$T_{\text{mix}}(\varepsilon) = \frac{1}{\kappa_{\text{total}}} \ln\left(\frac{1}{\varepsilon}\right)$$
:::

---

(sec-certificate-propagation)=
## Hypostructure Certificate Propagation

The hypostructure framework tightens bounds through systematic certificate propagation.

### Gate Contributions

| Gate | Predicate | Certificate Output |
|------|-----------|-------------------|
| **Gate 1 (Energy)** | $E[\Phi] < \infty$ | Bounds $V_{\max}$ |
| **Gate 4 (Scale)** | $\alpha < \beta + \lambda_c$ | Subcriticality |
| **Gate 10 (Ergodicity)** | Doeblin + hypoelliptic | Mixing rate $\kappa_{\text{mix}}$ |
| **BarrierGap** | $\inf\sigma(L) > 0$ | Spectral gap $\lambda_1$ |
| **BarrierSat** | Drift $\leq C$ | Energy saturation |

### Closure Operation

:::{prf:proposition} Certificate Closure
:label: prop-alg-sieve-closure

The closure operation computes all logical consequences of certificates:

$$\text{Cl}(\Gamma) = \bigcup_{n=0}^{\infty} \Gamma_n$$

where $\Gamma_0 = \Gamma$ and $\Gamma_{n+1} = \Gamma_n \cup \{\text{promotion rules}\}$.

The closure terminates in finite steps and produces bounds that are **tighter than any single analysis** because:
- Appendix bounds assume worst-case constants
- Algorithm bounds are necessary but not sufficient
- Hypostructure combines both with logical tightening
:::

---

(sec-parameter-selection)=
## Parameter Selection Algorithm

:::{prf:algorithm} Tight Parameter Selection
:label: alg-alg-sieve-parameter-selection

**Input**: Problem parameters $(M^2, d, \varepsilon_{\text{target}}, N)$

**Output**: Tight parameter set satisfying all constraints

```python
def compute_tight_parameters(M2, d, eps_target, N):
    """Compute tightest valid parameters for Fractal Gas."""

    # Default algorithm constants
    alpha, beta = 1.0, 1.0
    eta, A = 0.1, 2.0
    epsilon_clone = 0.01
    p_max = 1.0

    # 1. Compute fitness bounds
    V_min = eta ** (alpha + beta)
    V_max = (A + eta) ** (alpha + beta)
    S_max = (V_max - V_min) / (V_min + epsilon_clone)

    # 2. Phase control (Gamma ~ 1)
    Gamma_target = 1.0

    # 3. Acoustic stability: gamma > C * p_max * S_max * M2 / h
    C_acoustic = 1e-4  # problem-specific
    h_init = 0.01
    gamma_min = C_acoustic * p_max * S_max * M2 / h_init
    gamma = max(1.0, gamma_min)

    # 4. Timestep from stability: h < 1/(2*gamma)
    h = min(h_init, 1.0 / (2.0 * gamma))

    # 5. Temperature from phase control
    T_c = 2.0 * gamma * Gamma_target / (alpha * d / beta)

    # 6. Kernel scale from Doeblin: epsilon >= D_alg / 4.1
    D_z = 10.0  # latent diameter (problem-specific)
    D_v = 5.0   # velocity diameter
    lambda_alg = 0.0
    D_alg = sqrt(D_z**2 + lambda_alg * D_v**2)
    epsilon = D_alg / 4.1

    # 7. Jitter from LSI
    kappa_conf = 1.0  # confinement constant
    kappa_W = 0.1     # Wasserstein constant
    C_0 = 1.0         # interaction complexity
    C_KL = 1.0        # KL constant
    sigma_x = sqrt(C_KL / (gamma * kappa_conf * kappa_W * N))
    sigma_x = max(sigma_x, 0.01)  # minimum jitter

    # 8. Compute convergence rate
    rho_LSI = 1.0  # LSI constant
    kappa_hypo = gamma * rho_LSI / M2
    kappa_acoustic = gamma - C_acoustic * p_max * S_max * M2 / h
    kappa_total = min(kappa_hypo, kappa_W, kappa_conf, kappa_acoustic)
    kappa_total -= 1.0 / sqrt(N)  # finite-N correction

    # 9. Mixing time
    T_mix = log(1.0 / eps_target) / max(kappa_total, 1e-6)

    return {
        'N': N,
        'epsilon': epsilon,
        'gamma': gamma,
        'T_c': T_c,
        'h': h,
        'sigma_x': sigma_x,
        'alpha': alpha,
        'beta': beta,
        'eta': eta,
        'A': A,
        'epsilon_clone': epsilon_clone,
        'p_max': p_max,
        'kappa_total': kappa_total,
        'T_mix': T_mix
    }
```
:::

---

(sec-17-node-verification)=
## Verification: 17-Node Instantiation

The tight bounds are certified by the 17-node sieve from {prf:ref}`def-latent-fractal-gas-sieve-instantiation`:

| Node | Constraint | Certificate | Tight Bound |
|------|------------|-------------|-------------|
| 1 (Energy) | $\Phi \in [0, V_{\max}]$ | $K_E^+$ | $V_{\max} = 4.41$ |
| 2 (Recovery) | Bad set finite | $K_{\text{Rec}}^+$ | Cloning repairs |
| 3 (Confinement) | $S_N$ symmetry | $K_C^+$ | Permutation invariance |
| 4 (Scaling) | $\alpha = \beta = 2$ | $K_{\text{SC}}^+$ | Parabolic confinement |
| 5 (Parameters) | Constants fixed | $K_{\text{Par}}^+$ | Table values |
| 6 (Capacity) | Bad set capacity | $K_{\text{Cap}}^+$ | Finite |
| 7 (Analyticity) | $C^2$ regularity | $K_{\text{An}}^+$ | Bounded derivatives |
| 8 (Topology) | Single sector | $K_{\text{Top}}^+$ | Connected ball |
| 9 (Tameness) | O-minimal | $K_{\text{Tame}}^+$ | Definable |
| 10 (Ergodicity) | Doeblin + hypoelliptic | $K_{\text{Erg}}^+$ | $m_\epsilon > 0$, $T_c > 0$ |
| 11 (Complexity) | Finite precision | $K_{\text{Cx}}^+$ | Float64 |
| 12 (Oscillation) | Bounded | $K_{\text{Osc}}^+$ | Alive core |
| 13 (Boundary) | Open + killing | $K_{\partial}^+$ | Recovery via cloning |
| 14 (Overload) | Controlled | $K_{\text{Ov}}^+$ | Thermostat |
| 15 (Starvation) | QSD conditioning | $K_{\text{St}}^+$ | $n_{\text{alive}} \geq 1$ |
| 16 (Alignment) | Selection pressure | $K_{\text{Al}}^+$ | Mean fitness increase |
| 17 (Lock) | Pattern blocked | $K_{\text{Lock}}^+$ | Invariant mismatch |

---

## References

```{bibliography}
:filter: docname in docnames
```
